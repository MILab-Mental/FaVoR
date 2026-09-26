import pytest
import torch
from test_va_lejepa import tiny_va, batch_data
from models.va_lejepa.finetune import VAFinetuner
from app.finetune_va.train import _loss, task_metrics, EvaluationSampler


@pytest.mark.parametrize('strategy',['separate_logits','transformer'])
@pytest.mark.parametrize('task',['classification','regression','multi_label_classification'])
def test_task_heads_forward_backward(task,strategy):
    model=VAFinetuner(tiny_va(('feature','late')),1 if task=='regression' else 3,
                      {'branch_aggregation':strategy,'aggregator_nhead':3,'aggregator_depth':1})
    output=model(batch_data())
    if task=='classification':
        labels=torch.tensor([0,2]); criterion=torch.nn.CrossEntropyLoss()
    elif task=='regression':
        labels=torch.tensor([.1,.9]); criterion=torch.nn.MSELoss()
    else:
        labels=torch.tensor([[1.,0.,1.],[0.,1.,0.]]); criterion=torch.nn.BCEWithLogitsLoss()
    loss=_loss(output,labels,criterion,task,.1)
    loss.backward()
    assert model.heads['feature'].weight.grad is not None
    metrics,_,_,_=task_metrics(labels.tolist(),output['logits'].detach().tolist(),task,3)
    assert 'mae' in metrics if task=='regression' else 'f1_macro' in metrics


@pytest.mark.parametrize('freeze_encoders,freeze_fusion',[(True,True),(True,False),(False,False)])
def test_probe_fusion_only_end_to_end(freeze_encoders,freeze_fusion):
    model=VAFinetuner(tiny_va(),3,{'freeze_video_encoder':freeze_encoders,'freeze_audio_encoder':freeze_encoders,
                                 'freeze_fusion_adapter':freeze_fusion})
    model.train()
    assert model.va.video_encoder.training != freeze_encoders
    loss=model(batch_data())['logits'].square().mean()
    loss.backward()
    assert (model.va.video_encoder.patch_embed.proj.weight.grad is None)==freeze_encoders
    assert (model.va.input_adapters['feature'].video[0].weight.grad is None)==freeze_fusion
    assert all(not p.requires_grad for p in model.va.loss_projectors.parameters())


def test_eval_sampler_no_duplicate_padding():
    indices=[list(EvaluationSampler(list(range(5)),rank,2)) for rank in range(2)]
    assert sorted(indices[0]+indices[1])==list(range(5))


@pytest.mark.parametrize('task',['classification','regression','multi_label_classification'])
def test_finetune_entry_best_latest_predictions_resume(tmp_path,monkeypatch,task):
    import app.finetune_va.train as train
    from test_va_lejepa import view
    from torch.utils.data import Dataset
    from models.va_lejepa import save_checkpoint
    class Samples(Dataset):
        epoch=0
        def __init__(self,*args):
            self.labels=[0,1] if task=='classification' else [.1,.9] if task=='regression' else [[1.,0.],[0.,1.]]
        def __len__(self): return 2
        def __getitem__(self,index):
            sample=view(1)
            return {'global':{k:v[0] for k,v in sample.items()},'local':[],
                'label':torch.as_tensor(self.labels[index],dtype=torch.long if task=='classification' else torch.float32),
                'pair_id':str(index),'source_id':str(index),'video_path':'v','audio_path':'a','sync_diagnostics':{}}
    pretrained=tmp_path/'pretrain.pt'
    save_checkpoint(pretrained,tiny_va())
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(train,'build_va_lejepa',lambda *args,**kwargs:tiny_va())
    monkeypatch.setattr(train,'AVCSVDataset',Samples)
    cfg=dict(folder=str(tmp_path/'ft'),model={},data=dict(task=task,num_class=2,num_workers=0,batch_size=2),
        optimization=dict(epochs=1,lr=.001),finetune=dict(pretrained_checkpoint=str(pretrained)),meta=dict(auto_resume=True))
    train.main(cfg)
    cfg['optimization']['epochs']=2
    train.main(cfg)
    state=torch.load(tmp_path/'ft/latest.pt',weights_only=False)
    assert state['epoch']==2 and state['step']==2
    assert (tmp_path/'ft/best.pt').exists() and (tmp_path/'ft/logs/predictions.csv').exists()
