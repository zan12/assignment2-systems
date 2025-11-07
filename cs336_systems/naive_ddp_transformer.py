import argparse
import os
import random
import timeit
import torch
import numpy as np

import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from copy import deepcopy
from contextlib import nullcontext
from functools import partial

from cs336_basics import model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy

from cs336_systems.naive_ddp_parity import test_approximate_equality


def init_args(config: dict, parser: argparse.ArgumentParser, argv):
    for key, value in config.items():
        if key in ["context_length", "d_model", "d_ff", "num_layers", "num_heads", "num_steps", "world_size"]:
            parser.add_argument(f"--{key}", dest=key, default=value, type=int)
        elif key in ["use_amp", "profile_memory", "compile_model", "backward_hook"]:
            parser.add_argument(f"--{key}", dest=key, action='store_true')
        else:
            parser.add_argument(f"--{key}", dest=key, default=value)
        
    return parser.parse_args(argv)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def single_process_trainer(config, model):
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if config.compile_model:
        model = torch.compile(model)
    opt = AdamW(model.parameters())
    warmups = 5

    set_seed(5)  # set seed before generating data
    dataset = np.random.choice(config.vocab_size, (config.context_length*100,), replace=True)

    for t in range(warmups):
        set_seed(3+t)
        inputs, targets = get_batch(dataset, config.batch_size, config.context_length, device)
        opt.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        opt.step()
    
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if config.use_amp else nullcontext()
    forward_time, backward_time, opt_step_time = [], [], []
    prof = torch.profiler.profile(activities=[torch.profiler.   ProfilerActivity.CUDA])
    prof.start()
    if config.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for t in range(config.num_steps):
        set_seed(43+t)
        inputs, targets = get_batch(dataset, config.batch_size, config.context_length, device)
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
            after_step_time = timeit.default_timer()
        forward_time.append(after_forward_time - start_time)
        backward_time.append(after_backward_time - after_forward_time)
        opt_step_time.append(after_step_time - after_backward_time)
    prof.stop()
    if config.profile_memory:
        torch.cuda.memory._dump_snapshot(f"memory_snapshot_2_7b_compile_off.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    print("Per-step Forward Time - mean: {:.4f} std: {:.4f}".format(np.mean(np.array(forward_time)), np.std(forward_time)))
    print("Per-step Backward Time - mean: {:.4f} std: {:.4f}".format(np.mean(np.array(backward_time)), np.std(backward_time)))
    print("Per-step Optimizer Step Time - mean: {:.4f} std: {:.4f}".format(np.mean(np.array(opt_step_time)), np.std(opt_step_time)))


def grad_allreduce_hook(param, handles):
    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
    # handles.append((handle, param))
    handles.append((handle))


def setup_hooks(model):
    handles = []
    for param in model.parameters():
        param.register_post_accumulate_grad_hook(
            partial(grad_allreduce_hook, handles=handles)
        )
    return handles


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_ddp_trainer(rank, world_size, config, model, return_list):
    setup(rank, world_size)  # setup backend and device
    rank_device = f"cuda:{rank}"
    model.to(rank_device)
    if config.backward_hook:
        handles = setup_hooks(model)  # setup handles
    if config.compile_model:
        model = torch.compile(model)
    set_seed(5)  # setup seed before generating datasets
    dataset = np.random.choice(config.vocab_size, (config.context_length*100,), replace=True)
    opt = AdamW(model.parameters())
    warmups = 5

    slice_size = config.batch_size // world_size
    
    for t in range(warmups):
        set_seed(3+t)
        inputs, targets = get_batch(dataset, config.batch_size, config.context_length, rank_device)

        rank_inputs = inputs[rank*slice_size:(rank+1)*slice_size, ...]
        rank_targets = targets[rank*slice_size:(rank+1)*slice_size, ...]

        opt.zero_grad()
        loss = cross_entropy(model(rank_inputs), rank_targets)
        loss.backward()

        if config.backward_hook:
            # for handle, param in handles:
            for handle in handles:
                handle.wait()
                # param.grad /= world_size
            handles.clear()
        else:
            for param in model.parameters():
                dist.all_reduce(param.grad)
                param.grad /= world_size
        opt.step()

    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if config.use_amp else nullcontext()
    forward_time, backward_time, allreduce_time, opt_step_time = [], [], [], []
    prof = torch.profiler.profile(activities=[torch.profiler.   ProfilerActivity.CUDA])
    prof.start()
    if config.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for t in range(config.num_steps):
        set_seed(43+t)
        inputs, targets = get_batch(dataset, config.batch_size, config.context_length, rank_device)
        rank_inputs = inputs[rank*slice_size:(rank+1)*slice_size, ...]
        rank_targets = targets[rank*slice_size:(rank+1)*slice_size, ...]

        # handles = setup_hooks(model, world_size)
        opt.zero_grad()
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        with context:
            with nvtx.range("forward"):
                rank_outputs = model(rank_inputs)
                loss = cross_entropy(rank_outputs, rank_targets)
                torch.cuda.synchronize()
            after_forward_time = timeit.default_timer()
            with nvtx.range("backward"):
                loss.backward()
                torch.cuda.synchronize()
            after_backward_time = timeit.default_timer()
            with nvtx.range("all_reduce"):
                if config.backward_hook:
                    # for handle, param in handles:
                    for handle in handles:
                        handle.wait()
                        # param.grad /= world_size
                    handles.clear()
                elif config.ddp == "individual":
                    # All-reduce on gradients.
                    for param in model.parameters():
                        dist.all_reduce(param.grad)
                        param.grad /= world_size
                    # all_reduce is a sync op. there is no need to call barrier.
                    # dist.barrier()
                elif config.ddp == "flat":
                    all_param_grads = []
                    for param in model.parameters():
                        all_param_grads.append(param.grad)
                    flattened_grads = torch._utils._flatten_dense_tensors(tuple(all_param_grads))
                    # Communicate all parameters at once.
                    dist.all_reduce(flattened_grads, async_op=False)
                    all_param_grads = list(torch._utils._unflatten_dense_tensors(flattened_grads, all_param_grads))
                    for param in model.parameters():
                        param.grad = all_param_grads.pop(0) / world_size
                else:
                    raise ValueError("Invalid ddp flag.")

                torch.cuda.synchronize()
            after_allreduce_time = timeit.default_timer()
            with nvtx.range("optimizer_step"):
                opt.step()
                torch.cuda.synchronize()
            after_opt_step_time = timeit.default_timer()
        forward_time.append(after_forward_time - start_time)
        backward_time.append(after_backward_time - after_forward_time)
        allreduce_time.append(after_allreduce_time - after_backward_time)
        opt_step_time.append(after_opt_step_time - after_allreduce_time)
    prof.stop()
    # Return model for parity check.
    if rank == 0:
        return_list.append(model.to(device="cpu"))
    if config.profile_memory:
        torch.cuda.memory._dump_snapshot(f"memory_snapshot_2_7b_compile_off.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    
    forward_time_list = [None for _ in range(world_size)]
    backward_time_list = [None for _ in range(world_size)]
    allreduce_time_list = [None for _ in range(world_size)]
    opt_step_time_list = [None for _ in range(world_size)]
    dist.all_gather_object(forward_time_list, sum(forward_time)/len(forward_time))
    dist.all_gather_object(backward_time_list, sum(backward_time)/len(backward_time))
    dist.all_gather_object(allreduce_time_list, sum(allreduce_time)/len(allreduce_time))
    dist.all_gather_object(opt_step_time_list, sum(opt_step_time)/len(opt_step_time))
    
    if rank == 0:
        print("Per-step Forward Time - mean:", np.mean(np.array(forward_time_list)), " std:", np.std(np.array(forward_time_list)))
        print("Per-step Backward Time - mean:", np.mean(np.array(backward_time_list)), " std:", np.std(np.array(backward_time_list)))
        print("Per-step All-Reduce Time - mean:", np.mean(np.array(allreduce_time_list)), " std:", np.std(np.array(allreduce_time_list)))
        print("Per-step Optimizer Step Time - mean:", np.mean(np.array(opt_step_time_list)), " std:", np.std(np.array(opt_step_time_list)))


def ddp_trainer(config, model):
    world_size = config.world_size
    manager = mp.Manager()
    return_list = manager.list()
    mp.spawn(
        fn = run_ddp_trainer,
        args = (world_size, config, model, return_list),
        nprocs = world_size,
    )
    return return_list[0]


def run(argv = None):
    config = dict(
        vocab_size = 10_000,
        context_length = 128,
        d_model = 1600,
        num_layers = 48,
        num_heads = 25,
        d_ff = 6400,
        rope_theta = 10000.0,
        batch_size = 32,
        num_steps = 15,
        world_size = 4,
        use_amp = False,
        profile_memory = False,
        compile_model = False,
        backward_hook = False,
        ddp = None,
    )
    parser = argparse.ArgumentParser()
    config = init_args(config, parser, argv)
    model_single = BasicsTransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.rope_theta,
    )
    model_ddp = deepcopy(model_single)
    single_process_trainer(config, model_single) # in-place modification
    model_ddp = ddp_trainer(config, model_ddp)
    test_approximate_equality(model_single.to("cpu"), model_ddp)

if __name__ == "__main__":
    run()