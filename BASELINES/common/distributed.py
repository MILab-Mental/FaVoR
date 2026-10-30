"""Two-GPU runtime, exact evaluation shards and differentiable training steps."""
from datetime import timedelta
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


def active():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank():
    return torch.distributed.get_rank() if active() else 0


def world_size():
    return torch.distributed.get_world_size() if active() else 1


def initialize(test=False):
    local = int(os.environ.get('LOCAL_RANK', 0))
    if int(os.environ.get('WORLD_SIZE', 1)) > 1 and not active():
        torch.cuda.set_device(local)
        timeout=timedelta(minutes=5) if test else timedelta(hours=2)
        if os.environ.get('BASELINES_TCPSTORE_CLIENT')=='1':
            world=int(os.environ['WORLD_SIZE'])
            store=torch.distributed.TCPStore(os.environ['MASTER_ADDR'],int(os.environ['MASTER_PORT']),
                world+1,False,timeout=timeout)
            torch.distributed.init_process_group('nccl',store=store,rank=int(os.environ['RANK']),world_size=world,timeout=timeout)
        else:torch.distributed.init_process_group('nccl', timeout=timeout)
    return torch.device('cuda', local) if active() else None


def gather(value):
    if not active():
        return [value]
    values = [None] * world_size()
    torch.distributed.all_gather_object(values, value)
    return values


def barrier():
    if active():
        torch.distributed.barrier()


class EvaluationSampler(Sampler):
    """Never pad evaluation with another speaker's duplicate prediction."""
    def __init__(self, dataset):
        self.size = len(dataset)

    def __iter__(self):
        return iter(range(rank(), self.size, world_size()))

    def __len__(self):
        return len(range(rank(), self.size, world_size()))


class TrainingStep(nn.Module):
    def __init__(self, model, forward):
        super().__init__()
        self.model, self.compute = model, forward

    def forward(self, batch, step=0):
        return self.compute(self.model, batch, step)


def training_step(model, forward):
    step = TrainingStep(model, forward)
    if active():
        # Raw waveforms/frames must remain on CPU for official preprocessing;
        # adapters explicitly move their prepared inputs to the local GPU.
        step = DistributedDataParallel(step, device_ids=None,
            broadcast_buffers=any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()),
            find_unused_parameters=True, gradient_as_bucket_view=True)
    return step


@torch.no_grad()
def broadcast_batchnorm_buffers(model):
    """Distributed evaluation uses the same running statistics as rank-0 save."""
    if active():
        for module in model.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):
                for value in module.buffers(recurse=False):torch.distributed.broadcast(value,src=0)


def checkpoint_blocks(model):
    """Non-reentrant activation checkpointing without changing weight keys."""
    from functools import partial
    from torch.utils.checkpoint import checkpoint
    for module in model.modules():
        if hasattr(module,'use_checkpoint'):module.use_checkpoint=False
        if hasattr(module,'use_activation_checkpointing'):module.use_activation_checkpointing=False
        if hasattr(module,'gradient_checkpointing'):module.gradient_checkpointing=False
    for module in model.modules():
        block=any(cls.__name__.endswith('Block') or 'EncoderLayer' in cls.__name__ for cls in type(module).__mro__)
        # Recomputing a CNN block would update BatchNorm running stats twice.
        stateful=any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in module.modules()) if block else False
        if block and not stateful and not getattr(module,'_baseline_checkpoint',False) and any(p.requires_grad for p in module.parameters()):
            module.forward=partial(checkpoint,module.forward,use_reentrant=False)
            module._baseline_checkpoint=True


def cpu_copy(value):
    """An immutable RAM checkpoint; no temporary checkpoint file is needed."""
    import copy
    if torch.is_tensor(value):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def readonly_test():
    """Reject application writes, including hidden downloads and cache creation."""
    import sys
    import resource
    import tempfile
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    # Torch 2.7 generates a temporary RemoteModule Python file at import. Keep
    # mandatory runtime IPC/templates in RAM; application/cache writes stay denied.
    tempfile.tempdir='/dev/shm'
    os.environ['TMPDIR']='/dev/shm'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', CUDA_CACHE_DISABLE='1', TORCHDYNAMO_DISABLE='1')
    sys.dont_write_bytecode = True
    def audit(event, args):
        if event in ('open','os.mkdir','os.rename','os.remove','os.rmdir','os.utime'):
            filename=args[0]
            if isinstance(filename,str):
                if filename.startswith('/dev/shm/'):return
                if event in ('os.remove','os.rmdir') and len(args)>1 and isinstance(args[1],int) and args[1]>=0:
                    if os.readlink(f'/proc/self/fd/{args[1]}').startswith('/dev/shm/'):return
        if event == 'open':
            filename, mode, flags = args
            if isinstance(filename, int) and filename in (0, 1, 2):
                return
            if isinstance(filename,int):
                target=os.readlink(f'/proc/self/fd/{filename}')
                if target.startswith(('pipe:','socket:','/dev/shm/','/memfd:')):return
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise PermissionError(f'File-free test rejected write: {filename}')
        elif event in ('os.mkdir', 'os.rename', 'os.remove', 'os.rmdir', 'os.utime'):
            raise PermissionError(f'File-free test rejected {event}: {args[0]}')
    sys.addaudithook(audit)
