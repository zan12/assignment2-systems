import argparse
import timeit
import torch
import numpy as np

import torch.cuda.nvtx as nvtx

from contextlib import nullcontext
from .annotated_model import scaled_dot_product_attention as annotated_scaled_dot_product_attention

from cs336_basics import model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy

# Replace sdpa with annotated sdpa.
model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def init_args(config: dict, parser: argparse.ArgumentParser, argv):
    for key, value in config.items():
        if key in ["context_length", "d_model", "d_ff", "num_layers", "num_heads", "num_steps"]:
            parser.add_argument(f"--{key}", dest=key, default=value, type=int)
        elif key in ["use_amp", "profile_memory", "compile_model"]:
            parser.add_argument(f"--{key}", dest=key, action='store_true')
        else:
            parser.add_argument(f"--{key}", dest=key, default=value)
        
    return parser.parse_args(argv)


def run(argv = None):
    config = dict(
        vocab_size = 10_000,
        context_length = 128,
        d_model = 768,
        num_layers = 12,
        num_heads = 12,
        d_ff = 3072,
        rope_theta = 10000.0,
        batch_size = 4,
        num_steps = 15,
        use_amp = False,
        profile_memory = False,
        compile_model = False,
    )
    parser = argparse.ArgumentParser()
    params = init_args(config, parser, argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BasicsTransformerLM(
        params.vocab_size,
        params.context_length,
        params.d_model,
        params.num_layers,
        params.num_heads,
        params.d_ff,
        params.rope_theta,
    ).to(device)
    if params.compile_model:
        model = torch.compile(model)
    opt = AdamW(model.parameters())
    dataset = np.random.choice(params.vocab_size, (params.context_length*100,), replace=True)
    warmups = 5
    forward_time, backward_time = [], []

    for t in range(warmups):
        inputs, targets = get_batch(dataset, params.batch_size, params.context_length, device)
        opt.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        opt.step()

    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if params.use_amp else nullcontext()
    prof = torch.profiler.profile(activities=[torch.profiler.   ProfilerActivity.CUDA])
    prof.start()
    if params.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(params.num_steps):
        inputs, targets = get_batch(dataset, params.batch_size, params.context_length, device)
        opt.zero_grad()
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        with context:
            with nvtx.range("forward"):    
                outputs = model(inputs)
                loss = cross_entropy(outputs, targets)
                torch.cuda.synchronize()
            after_forward_time = timeit.default_timer()
            with nvtx.range("backward"):
                loss.backward()
                torch.cuda.synchronize()
            after_backward_time = timeit.default_timer()
            with nvtx.range("optimizer_step"):
                opt.step()
                torch.cuda.synchronize()
        forward_time.append(after_forward_time - start_time)
        backward_time.append(after_backward_time - after_forward_time)
    prof.stop()
    if params.profile_memory:
        torch.cuda.memory._dump_snapshot(f"memory_snapshot_2_7b_compile_off.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    print("Per-step Forward Time - mean: {:.4f} std: {:.4f}".format(np.mean(np.array(forward_time)), np.std(forward_time)))
    print("Per-step Backward Time - mean: {:.4f} std: {:.4f}".format(np.mean(np.array(backward_time)), np.std(backward_time)))

if __name__ == "__main__":
    run()