import torch
import torch.nn as nn
import torch.distributed as dist
from functools import partial


class DDPIndividualParameters(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for p in self.module.state_dict().values():
            dist.broadcast(p, src=0)
        self.handles = self.setup_hooks()

    def gradient_allreduce_hook(self, param, handles):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        handles.append((handle, param))

    def setup_hooks(self):
        handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    partial(self.gradient_allreduce_hook, handles=handles)
                )
        return handles

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, param in self.handles:
            handle.wait()
            param.grad /= dist.get_world_size()
        self.handles.clear()


class DDPBucketedParameters(nn.Module):
    
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        for p in self.module.state_dict().values():
            dist.broadcast(p, src=0)
        for name, p in self.module.named_parameters():
            p._name = name
        self.param_grad_bucket = []
        self.param_name_bucket = []
        self.bucket_occupied_mb = 0
        self.bucket_size_mb = bucket_size_mb
        self.handles = []
        self.setup_hooks()

    def flush_buckets(self):
        flattened_grads = torch._utils._flatten_dense_tensors(tuple(self.param_grad_bucket))
        handle = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, flattened_grads, self.param_grad_bucket, self.param_name_bucket))
        self.param_grad_bucket, self.param_name_bucket, self.bucket_occupied_mb = [], [], 0
        

    def gradient_allreduce_hook(self, param):
        # current parameter gradient size in mb
        param_grad_mb = param.grad.nbytes / 2 ** 20

        # put the paramter gradient into the bucket
        self.param_grad_bucket.append(param.grad)
        self.param_name_bucket.append(param._name)

        # update gradient bucket size
        self.bucket_occupied_mb += param_grad_mb

        # flush when bucket overflow
        if self.bucket_occupied_mb > self.bucket_size_mb:
            self.flush_buckets()

    def setup_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self.gradient_allreduce_hook
                )
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):

        def get_param_obj(module, name: str):
            parts = name.split(".")
            obj = module
            for part in parts[:-1]:  # walk through submodules
                obj = getattr(obj, part)
            return getattr(obj, parts[-1])
        
        # Flush buckets after backward, when opt step has not started.
        if self.param_grad_bucket:
            self.flush_buckets()

        for handle, flattened_grads, param_grads_shape, param_names in self.handles:
            handle.wait()  # wait for all-reduce finish
            bucket_param_grads = list(torch._utils._unflatten_dense_tensors(flattened_grads, param_grads_shape))
            for name, param_grads in zip(param_names, bucket_param_grads):
                param = get_param_obj(self.module, name)
                param.grad = param_grads / dist.get_world_size()
        self.handles.clear()