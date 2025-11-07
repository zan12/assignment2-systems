import math
import torch
import triton
import triton.language as tl

from einops import einsum


def cdiv(x, y):
    return (x + y - 1) // y


class FlashAttnPytorchAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, Bq=16, Bk=16):
        b, Nq, d = Q.shape
        _, Nk, dk = K.shape
        _, _, dv = V.shape
        assert d == dk == dv, "QKV hidden dimensions should match."
        Tq, Tk = Nq//Bq, Nk//Bk
        O = torch.empty(b, Nq, d, device=Q.device)
        L = torch.empty(b, Nq, device=Q.device)
        for i in range(Tq):
            Qi = Q[:, i*Bq:(i+1)*Bq, :] #[b, Bq, d]
            Oi = torch.zeros(b, Bq, d, device=Q.device) # [b, Bq, d]
            li = torch.zeros(b, Bq, device=Q.device) # [b, Bq]
            mi = -torch.inf * torch.ones(b, Bq, device=Q.device) # [b, Bq]
            for j in range(Tk):
                Kj = K[:, j*Bk:(j+1)*Bk, :] # [b, Bk, d]
                Vj = V[:, j*Bk:(j+1)*Bk, :] # [b, Bk, d]
                Sij = einsum(Qi, Kj, "b t d, b s d -> b t s") / math.sqrt(d)
                new_mi = torch.maximum(mi, torch.max(Sij,dim=-1)[0]) # [b, Bq]
                Pi = torch.exp(Sij-new_mi[:, :, None]) # [b, Bq, Bk]
                new_li = torch.exp(mi-new_mi)*li + torch.sum(Pi, dim=-1)
                new_Oi = einsum(torch.diag_embed(torch.exp(mi-new_mi)), Oi, "b t t1, b t1 d -> b t d") + einsum(Pi, Vj, "b t s, b s d -> b t d")
                # Update
                mi = new_mi
                li = new_li
                Oi = new_Oi
            Oi = einsum(torch.diag_embed(1.0 / li), Oi, "b t t1, b t1 d -> b t d")
            Li = mi + torch.log(li)
            O[:, i*Bq:(i+1)*Bq, :] = Oi
            L[:, i*Bq:(i+1)*Bq] = Li
        ctx.save_for_backward(Q,K,V,O,L)
        return O
    
    @staticmethod
    def backward(ctx, dO, is_causal=False, Bq=16, Bk=16):
        Q, K, V, O, L = ctx.saved_tensors
        b, Nq, d = Q.shape
        _, Nk, dk = K.shape
        _, _, dv = V.shape
        D = torch.sum(dO * O, dim=-1) # [b, Nq]

        Tq, Tk = Nq//Bq, Nk//Bk
        dQ = torch.zeros(Q.shape, device=Q.device)
        dK = torch.zeros(K.shape, device=K.device)
        dV = torch.zeros(V.shape, device=V.device)
        for j in range(Tk):
            Kj = K[:, j*Bk:(j+1)*Bk, :]
            Vj = V[:, j*Bk:(j+1)*Bk, :]
            dKj = torch.zeros(b, Bk, d, device=Q.device)
            dVj = torch.zeros(b, Bk, d, device=Q.device)
            for i in range(Tq):
                Qi = Q[:, i*Bq:(i+1)*Bq, :]
                dOi = dO[:, i*Bq:(i+1)*Bq, :]
                Li = L[:, i*Bq:(i+1)*Bq]
                Di = D[:, i*Bq:(i+1)*Bq]
                Sij = einsum(Qi, Kj, "b t d, b s d -> b t s") / math.sqrt(d)
                Pij = torch.exp(Sij - Li[:,:,None])
                dVj += einsum(Pij, dOi, "b t s, b t d -> b s d")
                dPij = einsum(dOi, Vj, "b t d, b s d -> b t s")
                dSij = Pij * (dPij - Di[:,:,None]) / math.sqrt(d)
                dQi = dQ[:, i*Bq:(i+1)*Bq, :]
                dQi += einsum(dSij, Kj, "b t s, b s d -> b t d")
                dKj += einsum(dSij, Qi, "b t s, b t d -> b s d")
            dK[:, j*Bk:(j+1)*Bk, :] = dKj
            dV[:, j*Bk:(j+1)*Bk, :] = dVj
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), # Same as key shape
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -1e16, dtype=tl.float32) # NOTE: should replace with a very small number
    new_oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    new_li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    new_mi = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Oi = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
    oi = Oi

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        if is_causal:
            q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = tl.where(tl.expand_dims(q_indices, axis=-1) >= tl.expand_dims(k_indices, axis=0), 0, -1e6)
        else:
            mask = 0
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale + mask # (Q_TILE_SIZE, K_TILE_SIZE)
        new_mi = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
        Pi = tl.exp(Sij-tl.expand_dims(new_mi, axis=-1)) # (Q_TILE_SIZE, K_TILE_SIZE)
        new_li = tl.exp(mi-new_mi)*li + tl.sum(Pi, axis=-1)
        new_oi = tl.expand_dims(tl.exp(mi-new_mi), axis=-1) * oi + tl.dot(Pi.to(Vj.dtype), Vj)
        mi = new_mi
        li = new_li
        oi = new_oi
        # Move the pointers to the next tile.
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    oi = tl.expand_dims(1.0 / li, axis=-1) * oi
    li = mi + tl.log(li)
    tl.store(O_block_ptr, oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, li, boundary_check=(0,))


@triton.jit
def flash_bwd_kernel(
    Q_ptr, O_ptr, dO_ptr,
    K_ptr, V_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_ob, stride_oo, stride_od,
    stride_dob, stride_doo, stride_dod,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, DIM),
        strides=(stride_oo, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, DIM),
        strides=(stride_doo, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, DIM),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, DIM),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Since each time we call the kernal, dQ will be re-initialized, we cannot accumulate its value inside the kernel.
    # Thus, we augment it and accumulate outside the kernel.
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES * (N_KEYS // K_TILE_SIZE), DIM),
        strides=(stride_dqq, stride_dqd),
        offsets=(key_tile_index * N_QUERIES, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, DIM),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, DIM),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0),
    )

    dKj = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)
    dVj = tl.zeros((K_TILE_SIZE, DIM), dtype=tl.float32)

    Kj = tl.load(K_block_ptr, boundary_check=(0, 1)) # (K_TILE_SIZE, DIM)
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1)) # (K_TILE_SIZE, DIM)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1)) # (Q_TILE_SIZE, DIM)
        Oi = tl.load(O_block_ptr, boundary_check=(0, 1)) # (Q_TILE_SIZE, DIM)
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1)) # (Q_TILE_SIZE, DIM)
        Li = tl.load(L_block_ptr, boundary_check=(0,)) # (Q_TILE_SIZE,)
        Di = tl.sum(dOi * Oi, axis=-1) # (Q_TILE_SIZE,)
        if is_causal:
            q_index = tl.expand_dims(tl.arange(0, Q_TILE_SIZE) + i * Q_TILE_SIZE, axis=-1)
            k_index = tl.expand_dims(tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE, axis=0)
            mask = tl.where(q_index >= k_index, 0, -1e6)
        else:
            mask = 0
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale + mask # (Q_TILE_SIZE, K_TILE_SIZE)
        Pij = tl.exp(Sij - tl.expand_dims(Li, axis=-1)) # (Q_TILE_SIZE, K_TILE_SIZE)
        dVj += tl.dot(tl.trans(Pij), dOi) # (K_TILE_SIZE, DIM)
        dPij = tl.dot(dOi, tl.trans(Vj)) # (Q_TILE_SIZE, K_TILE_SIZE)
        dSij = Pij * (dPij - tl.expand_dims(Di, axis=-1)) * scale
        dQi = tl.load(dQ_block_ptr, boundary_check=(0, 1)) # (Q_TILE_SIZE, DIM)
        dQi = tl.dot(dSij, Kj)
        tl.store(dQ_block_ptr, dQi, boundary_check=(0, 1))
        dKj += tl.dot(tl.trans(dSij), Qi)

        # Advance the block pointers to prepare for the next iteration.
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))

    tl.store(dK_block_ptr, dKj, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj, boundary_check=(0, 1))
    

class FlashAttnTritonAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        
        assert Q.shape[0] == K.shape[0] and Q.shape[-1] == K.shape[-1], "QK shape mismatch"
        assert K.shape == V.shape, "KV shape mismatch"
        
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal

        O = torch.empty(Q.shape, device=Q.device)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device)

        flash_fwd_kernel[(cdiv(N_QUERIES, ctx.Q_TILE_SIZE), batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=1/math.sqrt(D),
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal
        )
        ctx.save_for_backward(Q,K,V,O,L)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q,K,V,O,L = ctx.saved_tensors
        batch_size, N_QUERIES, DIM = Q.shape
        N_KEYS = K.shape[1]

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        scale = 1 / math.sqrt(DIM)

        dQ = torch.empty((batch_size, (N_KEYS//ctx.K_TILE_SIZE)*N_QUERIES, DIM), device=Q.device)
        dK = torch.empty(K.shape, device=K.device)
        dV = torch.empty(V.shape, device=V.device)

        flash_bwd_kernel[(cdiv(N_KEYS, ctx.K_TILE_SIZE), batch_size)](
            Q, O, dO,
            K, V, L,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES, N_KEYS,
            scale, DIM=DIM, 
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )
        dQ = dQ.view((batch_size, N_KEYS//ctx.K_TILE_SIZE, N_QUERIES, DIM)).sum(dim=1)
        return dQ, dK, dV, None


f_pytorch_flashattention = FlashAttnPytorchAutogradFunc.apply
f_triton_flashattention = FlashAttnTritonAutogradFunc.apply
device = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.randn((1,16,16), requires_grad=True, device=device)
k = torch.randn((1,32,16), requires_grad=True, device=device)
v = torch.randn((1,32,16), requires_grad=True, device=device)
do = torch.randn((1,16,16), requires_grad=False, device=device)
output_pytorch = f_pytorch_flashattention(q,k,v).backward(do)
ref_q_grad = q.grad.data
ref_k_grad = k.grad.data
ref_v_grad = v.grad.data

# print(ref_q_grad)

q.grad.zero_()
k.grad.zero_()
v.grad.zero_()

output_triton = f_triton_flashattention(q,k,v).backward(do)
# print(q.grad)
torch.testing.assert_close(ref_q_grad, q.grad, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(ref_k_grad, k.grad, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(ref_v_grad, v.grad, rtol=1e-2, atol=1e-2)
