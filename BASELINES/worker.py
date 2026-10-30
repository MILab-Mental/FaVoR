"""A rank-local experiment worker; --test keeps every training state in RAM."""
import argparse
import copy
import gc
import json
from types import SimpleNamespace

import torch

from app.main import load_config
from BASELINES.common.runtime import ROOT,OUTPUT_ROOT,path,setup,apply_overrides,strict_load,load_file,finalize_finetune_checkpoint
from BASELINES.common import distributed
from BASELINES.common.data import configure_instances,make_loaders,read_rows
from BASELINES.common.experiments import task_catalog,task_config
from BASELINES.build_model import build_model,TaskModel


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['pretrain','finetune'],required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--tasks',nargs='+',default=['all'])
    p.add_argument('--test',action='store_true')
    p.add_argument('--set',nargs='*',default=[])
    return p.parse_args()


def checked_parameters(model):
    tensors=[p for p in model.parameters() if p.requires_grad]
    tensors.extend(b for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
                   for b in m.buffers(recurse=False))
    sums=torch.stack([p.detach().float().sum() for p in tensors])
    if distributed.active():
        low,high=sums.clone(),sums.clone()
        torch.distributed.all_reduce(low,op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(high,op=torch.distributed.ReduceOp.MAX)
        if not torch.equal(low,high):raise RuntimeError('Two ranks have different model parameters or BatchNorm buffers')


def empty_args():
    return SimpleNamespace(resume=None,evaluate=None,checkpoint=None)


def pretrain(base,test):
    from BASELINES.pretrain.trainer import main
    cfg=copy.deepcopy(base)
    cfg['stage']='pretrain'
    cfg['no_save']=test
    if test:
        accumulation=int(cfg['optimization'].get('gradient_accumulation',1))
        cfg['data'].update(limit=2*distributed.world_size()*accumulation,batch_size=2,num_workers=int(cfg.get('test_workers',0)))
        cfg['optimization'].update(epochs=2)
        cfg.update(max_batches=accumulation,stop_after_epoch=2,verify_resume=True)
    setup(cfg)
    args=empty_args()
    last=path(cfg['output'])/'latest.pt'
    if not test and last.exists():
        saved=load_file(last)
        for key in ('model','data','optimization','seed','precision'):
            if saved['config'].get(key)!=cfg.get(key):raise ValueError('Existing pretraining run differs in '+key+': '+str(last))
        args.resume=str(last)
        if saved['epoch']>=int(cfg['optimization']['epochs']):
            if distributed.rank()==0:print('COMPLETE PRETRAIN',cfg['model']['id'],flush=True)
            return None,cfg
        del saved
    model=build_model(cfg,pretrain=True,initialize=not bool(args.resume))
    distributed.checkpoint_blocks(model)
    main(cfg,args,model=model)
    checked_parameters(model)
    if distributed.rank()==0:
        print('PASS PRETRAIN',cfg['model']['id'],
              json.dumps(cfg.get('test_result',{}).get('history',[])),flush=True)
    encoder=distributed.cpu_copy(model.export_encoder()) if test else None
    del model
    gc.collect()
    torch.cuda.empty_cache()
    distributed.barrier()
    return encoder,cfg


def selected_tasks(selectors):
    catalog=task_catalog()
    if selectors==['all']:return list(catalog)
    selected=[]
    for selector in selectors:
        matches=[name for name in catalog if name==selector or name.startswith(selector+'/') or name.rsplit('/',1)[-1]==selector]
        if not matches:raise ValueError('Unknown task: '+selector)
        selected.extend(name for name in matches if name not in selected)
    return selected


def load_transfer(adapter,state,cfg):
    if hasattr(adapter,'load_encoder'):adapter.load_encoder(state)
    else:strict_load(adapter.encoder,state,
        allow_missing=('fc_norm.',) if cfg['model']['kind'] in ('audiomae','videomae') else ())


def finetune(base,tasks,test,encoder_state=None):
    from BASELINES.finetune.trainer import main,run_epoch
    from app.finetune_a.train import _criterion
    outcomes=[]
    backbone=initial=None
    upstream_code=None
    for name in tasks:
        if test:
            import faulthandler
            faulthandler.dump_traceback_later(300,exit=True)
        cfg=task_config(base,name,'audio',test)
        # The first build establishes the native modality; later tasks reuse
        # only its architecture and restore its initial weights before fitting.
        if backbone is None:
            cfg['model'].pop('instance_aggregator',None)
            setup(cfg)
            initial_model=build_model(cfg,initialize=encoder_state is None)
            upstream_code=copy.deepcopy(cfg.get('upstream_code'))
            if encoder_state is not None:load_transfer(initial_model.encoder,encoder_state,cfg)
            modality=initial_model.input_spec['modality']
            backbone=initial_model.encoder
            initial=distributed.cpu_copy(backbone.state_dict())
            del initial_model
        cfg=task_config(base,name,modality,test)
        if upstream_code is not None:cfg['upstream_code']=copy.deepcopy(upstream_code)
        if test:
            cfg['model'].setdefault('encoder_batch_size',1)
            accumulation=int(cfg['optimization'].get('gradient_accumulation',1))
            cfg['data']['batch_size']=int(cfg.get('test_batch_size',1))
            cfg['data']['num_workers']=int(cfg.get('test_workers',0))
            cfg['data']['limit']=max(4,distributed.world_size()*accumulation*cfg['data']['batch_size'])
            cfg['optimization'].update(epochs=1)
            cfg.update(max_batches=accumulation,stop_after_epoch=1)
        rows,missing=read_rows(cfg['data'],modality,max_records=1)
        if not rows and missing:
            outcomes.append(dict(task=name,status='not_applicable',reason='no real '+modality+' inputs'))
            continue
        configure_instances(cfg,modality)
        if not test and cfg['data'].get('num_sentence'):
            cfg['data']['batch_size']=min(2,int(cfg['data'].get('batch_size',8)))
            cfg['optimization']['gradient_accumulation']=8//cfg['data']['batch_size']
        if cfg['data']['task']=='regression':cfg['data']['num_class']=1
        folder=OUTPUT_ROOT/'finetune'/cfg['model']['id']/cfg['experiment']
        args=empty_args()
        if not test and (folder/'best.pt').exists():
            saved=load_file(folder/'best.pt')['config']
            for key in ('model','data','optimization','seed'):
                if saved.get(key)!=cfg.get(key):raise ValueError('Existing run differs in '+key+': '+str(folder))
            report=json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').exists() else {}
            if 'test' in report.get('evaluations',{}):
                finalize_finetune_checkpoint(folder/'best.pt')
                outcomes.append(dict(task=name,status='completed'))
                if distributed.rank()==0:print('COMPLETE FINETUNE',cfg['model']['id'],name,flush=True)
                continue
            args.resume=str(folder/'best.pt')
        if distributed.rank()==0:print('START FINETUNE',base['model']['id'],name,flush=True)
        backbone.load_state_dict(initial,strict=True)
        setup(cfg)
        model=TaskModel(backbone,1 if cfg['data']['task']=='regression' else int(cfg['data']['num_class']),
                        cfg['model'].get('instance_aggregator'),encoder_batch_size=cfg['model'].get('encoder_batch_size',8))
        distributed.checkpoint_blocks(model) if model is not None else None
        first=not any(r['status']=='passed' for r in outcomes)
        if test and first:
            cfg['optimization']['epochs']=2
            cfg.update(stop_after_epoch=2,verify_resume=True)
        try:
            main(cfg,args,model=model)
            if test:
                checked_parameters(model)
                loaders=make_loaders(cfg,model.input_spec)
                device=torch.device('cuda',torch.cuda.current_device())
                criterion=_criterion(cfg['data']['task'],cfg['optimization'],loaders['train'].dataset,int(cfg['data']['num_class']),device)
                model.eval()
                with distributed_eval(loaders.get('test')):
                    evaluation=run_epoch(model,loaders['test'],criterion,cfg,device) if 'test' in loaders else {}
                history=cfg['test_result']['history']
                result=dict(task=name,status='passed',steps=cfg['test_result']['optimizer_steps'],
                    train_loss=history[-1]['train']['loss'],test_samples=evaluation.get('samples'),
                    instance_level=bool(cfg['data'].get('num_sentence')),synchronized=True,
                    resume=cfg.get('resume_verified_in_memory',False))
                if first:result['io_check']=cfg.get('io_check',{})
            else:
                gc.collect()
                torch.cuda.empty_cache()
                main(cfg,SimpleNamespace(resume=None,evaluate='test',checkpoint=str(folder/'best.pt')),model=model)
                finalize_finetune_checkpoint(folder/'best.pt')
                result=dict(task=name,status='passed')
            outcomes.append(result)
            if distributed.rank()==0:print('PASS FINETUNE',base['model']['id'],json.dumps(result),flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        distributed.barrier()
    return outcomes


def distributed_eval(loader):
    return torch.no_grad() if loader is None else _evaluation(loader)


def _evaluation(loader):
    from contextlib import contextmanager
    from BASELINES.common.runtime import evaluation_rng
    @contextmanager
    def context():
        with evaluation_rng(loader),torch.no_grad():yield
    return context()


def main():
    args=arguments()
    if args.test:distributed.readonly_test()
    distributed.initialize(test=args.test)
    base=apply_overrides(load_config(path(args.config)),args.set)
    base['stage']=args.stage
    base['no_save']=args.test
    tasks=selected_tasks(args.tasks)
    try:
        if args.stage=='pretrain':
            state,cfg=pretrain(base,args.test)
            outcomes=[]
            if args.test:
                name=cfg['model']['id']+('-continued' if cfg['route']=='continue_pretrain' else '')
                downstream=load_config(ROOT/'BASELINES/finetune/configs'/f'{name}.yaml')
                outcomes=finetune(downstream,['ordinary/RAVDESS-emotion'],True,state)
        else:
            state=None
            if args.test and base['model'].get('encoder_checkpoint'):
                source=load_config(ROOT/'BASELINES/pretrain/configs'/f'{base["model"]["id"]}.yaml')
                state,_=pretrain(source,True)
            outcomes=finetune(base,tasks,args.test,state)
        if distributed.rank()==0:
            print('RESULT',json.dumps(dict(model=base['model']['id'],stage=args.stage,test=args.test,
                  world_size=distributed.world_size(),outcomes=outcomes)),flush=True)
    except BaseException:
        # A failed rank must exit before NCCL teardown can wait on a peer that
        # is still in backward. The parent then terminates the other rank.
        import os
        import sys
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        if args.test:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        if distributed.active():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
