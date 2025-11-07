import itertools
import os
import pandas as pd
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def distributed_demo(rank, world_size, data_size, return_dict):
    setup(rank, world_size)
    num_elements = data_size // 4  # float32
    warmup = 5
    # Warmups
    for _ in range(warmup):
        data = torch.randn((num_elements,), dtype=torch.float32, device=f"cuda:{rank}")
        dist.all_reduce(data, async_op=False)
    # After warmups
    data = torch.randn((num_elements,), dtype=torch.float32, device=f"cuda:{rank}")
    torch.cuda.synchronize()
    start = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()
    end = timeit.default_timer()

    object_list = [None for _ in range(world_size)]
    dist.all_gather_object(object_list, end-start)
    if rank == 0:
        return_dict["world_size"] = world_size
        return_dict["num_elements"] = num_elements
        return_dict["time_cost"] = sum(object_list) / len(object_list)


if __name__ == "__main__":
    world_size = [2,4,6]
    data_size = [2**20, 10*2**20, 100*2**20, 2**30]
    manager = mp.Manager()
    result_table = []
    for w, d in itertools.product(world_size, data_size):
        return_dict = manager.dict()
        mp.spawn(fn=distributed_demo, args=(w, d, return_dict), nprocs=w, join=True)
        result_table.append(dict(return_dict))
    df = pd.DataFrame(result_table)
    print(df)