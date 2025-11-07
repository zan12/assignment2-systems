import torch
import triton


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

flash = torch.compile(FlashAttention2.apply)

def flash_forward_backward():
    o = flash(q, k, v, True)
    loss = o.sum()
    loss.backward()

results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
print(results)