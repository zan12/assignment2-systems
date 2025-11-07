import os
import torch
import torch.nn as nn

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.optim import SGD
from copy import deepcopy


class ToyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.relu(self.linear1(x)))


def single_process_trainer(batch_size, input_dim, model, f_loss):
    opt = SGD(model.parameters(), lr=1.)
    device = "cuda"
    model.to(device)
    for t in range(100):
        torch.cuda.manual_seed(43+t)
        data = torch.randn((batch_size, input_dim), dtype=torch.float32, device=device)
        label = 2*data
        pred = model(data) 
        loss = f_loss(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_distributed_trainer(rank, world_size, batch_size, input_dim, model, f_loss, return_list):
    setup(rank, world_size)
    rank_device = f"cuda:{rank}"
    model.to(rank_device)
    opt = SGD(model.parameters(), lr=1.)
    for t in range(100):
        torch.manual_seed(43+t)
        data = torch.randn((batch_size, input_dim), dtype=torch.float32, device=rank_device)
        label = 2*data
        slice_size = data.shape[0] // world_size
        # Slice data and label.
        data_slice = data[rank*slice_size:(rank+1)*slice_size, ...].to(device=rank_device)
        label_slice = label[rank*slice_size:(rank+1)*slice_size, ...].to(device=rank_device)
        pred = model(data_slice)
        loss = f_loss(pred, label_slice)
        opt.zero_grad()
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad.data, async_op=False)
            param.grad.data /= world_size
        opt.step()
    if rank == 0:
        # model to cpu before returning
        return_list.append(model.to(device="cpu"))


def distributed_trainer(world_size, batch_size, input_dim, model2, f_loss, return_list):
    mp.spawn(
        fn=run_distributed_trainer, 
        args=(world_size, batch_size, input_dim, model2, f_loss, return_list), 
        nprocs=world_size, 
        join=True,
    )


def test_approximate_equality(model1, model2, rtol=1e-2, atol=1e-2):
    all_close = True
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), 
                                                model2.named_parameters()):
        if name1 != name2:
            raise ValueError(f"Parameter names don't match: {name1} vs {name2}")
        assert torch.allclose(param1, param2, rtol=rtol, atol=atol)
    

if __name__ == "__main__":
    input_dim, hidden_dim, output_dim, batch_size = 16, 32, 16, 32
    model1 = ToyModel(input_dim, hidden_dim, output_dim)
    model2 = deepcopy(model1)
    f_loss = nn.MSELoss()
    manager = mp.Manager()
    return_list = manager.list()
    world_size = 4
    
    # single process trainer
    single_process_trainer(batch_size, input_dim, model1, f_loss)
    # distributed trainer requires world size and return list
    distributed_trainer(world_size, batch_size, input_dim, model2, f_loss, return_list)

    model2 = return_list[0]
    test_approximate_equality(model1.to("cpu"), model2)
