import argparse
import math

import torch
import torch.nn as nn

from einops import einsum
import torch.cuda.nvtx as nvtx


def update_config(parser: argparse.ArgumentParser, default: dict, argv):
    for key, value in default.items():
        if key in ["d_model", "seq_len", "batch_size"]:
            parser.add_argument(f"--{key}", dest=key, default=value, type=int)
    return parser.parse_args(argv)


def my_softmax(input, dim): # [b t d]
    offset, _ = torch.max(input, dim=dim, keepdim=True)
    input = torch.exp(input - offset)
    scale = torch.sum(input, dim=dim, keepdim=True)
    return input / scale


def my_attention(
    q: torch.Tensor, # [b,t,d]
    k: torch.Tensor, # [b,s,d]
    v: torch.Tensor, # [b,s,d]
    mask: torch.Tensor, # [b,t,s]
):
    assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    d = q.shape[-1]
    qk = einsum(q, k, "b t d, b s d -> b t s") / math.sqrt(d)
    qk += torch.where(mask, 0, -torch.inf)
    qk = my_softmax(qk, dim=-1)
    a = einsum(qk, v, "b t s, b s d -> b t d")
    return a
    
def my_loss(input):
    return torch.mean(torch.square(input))
    

def run(argv=None):
    parser = argparse.ArgumentParser()
    config = dict(
        d_model = 16,
        seq_len = 256,
        batch_size = 8,
    )
    config = update_config(parser, config, argv)
    d_model = config.d_model
    seq_len = config.seq_len
    b = config.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d, t = d_model, seq_len
    q = nn.Parameter(torch.randn((b,t,d), device=device) / math.sqrt(d))
    k = nn.Parameter(torch.randn((b,t,d), device=device) / math.sqrt(d))
    v = nn.Parameter(torch.randn((b,t,d), device=device) / math.sqrt(d))
    mask = torch.tril(torch.ones(t,t)).to(dtype=torch.bool, device=device)
    my_attention_compiled = torch.compile(my_attention)
    for _ in range(5): # warmup
        o = my_attention_compiled(q,k,v,mask)
        loss = my_loss(o)
        loss.backward()

    prof = torch.profiler.profile(activities=[torch.profiler.   ProfilerActivity.CUDA])
    prof.start()
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(100):
        with nvtx.range("forward"):
            o = my_attention_compiled(q,k,v,mask)
            loss = my_loss(o)
            torch.cuda.synchronize()
        with nvtx.range("backward"):
            loss.backward()
            torch.cuda.synchronize()
    prof.stop()
    torch.cuda.memory._dump_snapshot(f"memory_snapshot_2_7b_compiled.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    run()