"""Launch local DDP ranks without elastic log directories or temporary files."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
import subprocess
import sys

from app.main import load_config
from BASELINES.common.runtime import ROOT,apply_overrides
from BASELINES.common.experiments import resource_errors,task_catalog


def python_environment(python,test=False):
    from pathlib import Path
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
    if '/environments/' in python:
        env['MPLCONFIGDIR']=str(Path(python).parents[1]/'.matplotlib')
    if test:env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1',TORCHDYNAMO_DISABLE='1',TMPDIR='/dev/shm')
    return env


def stop_children(children):
    for process in children:
        if process.poll() is None:process.terminate()
    for process in children:
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_ranks(command,gpus,test):
    from datetime import timedelta
    import torch
    # Keep the rendezvous port bound for the whole run. Choosing an unused
    # port then closing its socket races with other jobs on this shared host.
    store=torch.distributed.TCPStore('127.0.0.1',0,gpus+1,True,
        timeout=timedelta(minutes=5) if test else timedelta(hours=2),wait_for_workers=False)
    port=store.port
    children=[]
    try:
        for rank in range(gpus):
            env=python_environment(command[0],test)
            env.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),WORLD_SIZE=str(gpus),
                       RANK=str(rank),LOCAL_RANK=str(rank),BASELINES_TCPSTORE_CLIENT='1')
            children.append(subprocess.Popen(command,cwd=ROOT,env=env))
        with ThreadPoolExecutor(max_workers=gpus) as pool:
            futures=[pool.submit(p.wait) for p in children]
            for future in as_completed(futures):
                if future.result()!=0:
                    stop_children(children)
                    return 1
        return 0
    finally:
        stop_children(children)
        del store


def matrix(configs,args):
    if args.gpus!=2:
        raise ValueError('The comparison protocol uses two GPUs; pass --gpus 2')
    if args.resume or args.evaluate or args.checkpoint or args.smoke:
        raise ValueError('Matrix mode uses --test for file-free checks; use a single experiment for resume/evaluate')
    tasks=args.tasks or ['all']
    print('MATRIX',args.stage,'33 ordinary + 38 mental task definitions; two DDP ranks; test=',args.test,flush=True)
    task_catalog()
    resources=[(config,resource_errors(apply_overrides(load_config(config),args.set),args.test)) for config in configs]
    failures=[]
    for config,issues in resources:
        if issues:
            failures.append(config.stem)
            print('BLOCKED',config.stem,'; '.join(issues),flush=True)
    if failures and not args.test:
        raise RuntimeError('Resource preflight failed before training: '+', '.join(failures))
    for config,issues in resources:
        if issues:continue
        cfg=apply_overrides(load_config(config),args.set)
        command=[cfg.get('python',sys.executable),'-B','-m','BASELINES.worker','--stage',args.stage,'--config',str(config),'--tasks',*tasks]
        if args.test:command.append('--test')
        if args.set:command.extend(['--set',*args.set])
        code=run_ranks(command,args.gpus,args.test)
        print('MODEL_RESULT',config.stem,'FAILED' if code else 'PASSED',flush=True)
        if code:failures.append(config.stem)
        if code and not args.test:break
    print('MATRIX_RESULT',args.stage,'models=',len(configs),'failed_or_blocked=',failures,flush=True)
    if failures:sys.exit(1)
