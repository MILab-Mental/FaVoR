"""Published AVSiam Base+ and the table's random-initialized Large candidate."""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from BASELINES.common.official import official
from BASELINES.common.runtime import load_file, path, strict_load
from .audio import fbank
from .video import frames_tensor


def avsiam_source():
    import ast
    import sys
    import types
    from BASELINES.common.official import checkout
    directory = checkout('AVSiam') / 'src/models'
    filename = directory / 'cav_mae_base.py'
    tree = ast.parse(filename.read_text())
    # The release imports debugging/ToMe files that are absent, but their
    # only call sites are commented out. Verify that before removing ONLY
    # these unused imports; execute all backbone/fusion code unchanged.
    unused = {'set_trace', 'bipartite_soft_matching', 'merge_source', 'merge_wavg', 'yb_bipartite_soft_matching'}
    if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in unused for n in ast.walk(tree)):
        raise ValueError('AVSiam now uses formerly dead imports; review the source adaptation')
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and
                 node.module in ('ipdb', 'tome.merge', 'yb_tome'))]
    package_name = '_favor_baseline_avsiam'
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    module = types.ModuleType(package_name + '.cav_mae_base')
    module.__file__, module.__package__ = str(filename), package_name
    sys.modules[module.__name__] = module
    with official('AVSiam','src'):
        exec(compile(tree,str(filename),'exec'),module.__dict__)
    return module


class AVSiam(nn.Module):
    """Published Base+ only; two shared streams followed by native fusion blocks."""
    def __init__(self, cfg, initialize=True):
        super().__init__()
        import timm
        module=avsiam_source()
        with official('AVSiam', 'src'):
            create_model = timm.create_model
            def no_redundant_download(name, **kwargs):
                if name != 'vit_base_patch16_224.augreg_in21k':
                    raise ValueError('Unexpected AVSiam backbone')
                kwargs['pretrained'] = False  # full Base+ checkpoint replaces these tensors
                return create_model(name, **kwargs)
            timm.create_model = no_redundant_download
            try:
                self.encoder = module.CAVMAEFT_BASE(label_dim=1)
            finally:
                timm.create_model = create_model
        for name in ('my_blocks', 'my_patch_embed', 'my_patch_embed_a'):
            delattr(self.encoder, name)
        for name in ('mlp_head', 'mlp_head_a', 'mlp_head_mm', 'mlp_head_mm_v2'):
            setattr(self.encoder, name, nn.Identity())
        self.encoder.vit_base.head = nn.Identity()
        if initialize:
            from huggingface_hub import hf_hub_download
            weights = cfg.get('weights') or hf_hub_download(cfg['hub_id'], cfg['filename'], revision=cfg['revision'])
            checkpoint = load_file(weights)
            strict_load(self.encoder, checkpoint.get('model', checkpoint), discard=(
                'my_blocks.', 'my_patch_embed.', 'my_patch_embed_a.', 'ast_base.',
                'decoder_', 'mask_token', 'vit_base.head.', 'mlp_head', 'predictor.'))
        self.output_dim = 1536
        self.input_spec = dict(modality='audio-video', sample_rate=16000, seconds=10.24, frames=1, size=224)

    def prepare(self, clip):
        audio, valid = fbank(clip['audio'], mean=-5.081, std=4.4849)
        return dict(audio=audio[None], valid=torch.tensor([valid]), video=frames_tensor(clip['video'], training=self.training, square_resize=True)[None])

    def forward(self, batch):
        # Native training path returns fused, audio, video readouts. Only the
        # fused representation feeds our new task head in both train and eval.
        fused, _, _ = self.encoder(batch['audio'], batch['video'], mode='mm_grad', is_eval=False)
        return fused

    def gradient_components(self):
        return dict(audio_input=self.encoder.vit_base.patch_embed_a,
                    video_input=self.encoder.vit_base.patch_embed)


class AVSiamLarge(AVSiam):
    """Paper-based Large candidate with explicitly random initialization.

    Reuse the published blocks, masking, decoder and losses. The released PT
    constructor hardcodes Base and a private ImageNet file; its reconstruction
    path also creates a second encoder. Construct the paper's shared Large
    instead, retaining two fusion blocks and a six-block MAE decoder.
    """
    def __init__(self,cfg,initialize=False,pretrain=False):
        nn.Module.__init__(self)
        if initialize:raise ValueError('AVSiam Large has no specified public checkpoint; pretrain it from scratch')
        if cfg.get('size')!='large':raise ValueError('The AVSiam candidate must be Large')
        import timm
        native=avsiam_source()
        cls=native.CAVMAE_BASE if pretrain else native.CAVMAEFT_BASE
        model=cls.__new__(cls)
        nn.Module.__init__(model)
        backbone=timm.create_model('vit_large_patch16_224',pretrained=False,num_classes=0)
        if len(backbone.blocks)!=24 or backbone.embed_dim!=1024:
            raise ValueError('Expected the standard 24-layer, 1024-dimension ViT-L/16')
        blocks=[]
        for original in backbone.blocks:
            for key in ('norm1','norm2'):
                setattr(original,key+'_a',copy.deepcopy(getattr(original,key)))
                setattr(original,key+'_v',copy.deepcopy(getattr(original,key)))
            block=native.Block(dim=1024,num_heads=16,qkv_bias=True)
            block.load_state_dict(original.state_dict(),strict=True)
            blocks.append(block)
        backbone.blocks=nn.Sequential(*blocks)
        patch=native.PatchEmbed(embed_dim=1024)
        patch.load_state_dict(backbone.patch_embed.state_dict(),strict=True)
        backbone.patch_embed=patch
        backbone.patch_embed_a=native.PatchEmbed(in_chans=1,embed_dim=1024)
        backbone.patch_embed_a.load_state_dict(
            {'proj.weight':patch.proj.weight.mean(1,keepdim=True),'proj.bias':patch.proj.bias},strict=True)
        backbone.pos_embed_a=nn.Parameter(F.interpolate(backbone.pos_embed[:,1:].transpose(1,2),size=512).transpose(1,2))
        backbone.norm_a=copy.deepcopy(backbone.norm)
        backbone.norm_pre_a=copy.deepcopy(backbone.norm_pre)
        model.vit_base=backbone
        model.mm_layer_1=copy.deepcopy(blocks[-2])
        model.mm_layer_2=copy.deepcopy(blocks[-1])
        if pretrain:
            model.decoder_embed=nn.Linear(1024,512)
            model.mask_token=nn.Parameter(torch.empty(1,1,512))
            nn.init.normal_(model.mask_token,std=.02)
            model.decoder_blocks=nn.Sequential(*[native.Block(dim=512,num_heads=16,qkv_bias=True) for _ in range(6)])
            model.decoder_norm=nn.LayerNorm(512)
            model.decoder_pred_a=nn.Linear(512,256)
            model.decoder_pred_v=nn.Linear(512,768)
            for mode,height,width in [('a',8,64),('v',14,14)]:
                # Same fixed sine/cosine coordinates as the referenced MAE.
                with official('AVSiam','src'):
                    positions=torch.as_tensor(native.get_2d_sincos_pos_embed(512,height,width)).float()[None]
                setattr(model,'decoder_pos_embed_'+mode,nn.Parameter(positions,requires_grad=False))
                embedding=nn.Parameter(torch.empty(1,1,512))
                nn.init.normal_(embedding,std=.02)
                setattr(model,'decoder_modality_'+mode,embedding)
        self.encoder=model
        self.pretraining=pretrain
        self.output_dim=2048
        self.input_spec=dict(modality='audio-video',sample_rate=16000,seconds=10.24,frames=1,size=224)
        if pretrain:self.input_spec['minimum_batch_size']=2

    def _embed(self,batch):
        model=self.encoder.vit_base
        audio=model.patch_embed_a(batch['audio'].unsqueeze(1).transpose(2,3))+model.pos_embed_a
        video=model.patch_embed(batch['video'].flatten(0,1))+model.pos_embed[:,1:]
        return audio+model.norm_pre_a(audio),video+model.norm_pre(video)

    def _encode(self,tokens,mode):
        model=self.encoder.vit_base
        for block in model.blocks:tokens=block(tokens,mode)
        return (model.norm_a if mode=='a' else model.norm)(tokens)

    def _fuse(self,audio,video):
        tokens=torch.cat([audio,video],1)
        return self.encoder.mm_layer_2(self.encoder.mm_layer_1(tokens,'a'),'a')

    def forward(self,batch):
        audio,video=self._embed(batch)
        audio,video=self._encode(audio,'a'),self._encode(video,'v')
        tokens=self._fuse(audio,video)
        mask=torch.arange(64,device=tokens.device).repeat(8)[None]<(batch['valid'][:,None]+15)//16
        audio=(tokens[:,:512]*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return torch.cat([audio,tokens[:,512:].mean(1)],-1)

    def _multi_ratio(self,tokens,mode):
        # Paper Fig.3 / Sec.3.2: independently random 0..50% in 5% increments.
        # Group equal ratios to preserve dense batched attention, including B<11.
        ratios=torch.randint(11,(len(tokens),),device=tokens.device)
        features,order=[],[]
        for ratio in ratios.unique(sorted=True).tolist():
            indices=(ratios==ratio).nonzero().flatten()
            masked,_,_=self.encoder.random_masking_unstructured(tokens[indices],ratio*.05)
            features.append(self._encode(masked,mode).mean(1))
            order.append(indices)
        return torch.cat(features).index_select(0,torch.cat(order).argsort())

    def pretrain_loss(self,batches,step=0):
        batch={key:torch.cat([b[key] for b in batches]) for key in batches[0]}
        audio,video=self._embed(batch)
        model=self.encoder
        a,mask_a,restore_a=model.random_masking_unstructured(audio,.75)
        v,mask_v,restore_v=model.random_masking_unstructured(video,.75)
        fused=self._fuse(self._encode(a,'a'),self._encode(v,'v'))
        predicted_a,predicted_v=model.forward_decoder(fused,mask_a,restore_a,mask_v,restore_v)
        reconstruction=model.forward_mae_loss(batch['audio'],predicted_a,mask_a,'a')
        reconstruction=reconstruction+model.forward_mae_loss(batch['video'][:,0],predicted_v,mask_v,'v')
        a,v=self._multi_ratio(audio,'a'),self._multi_ratio(video,'v')
        from BASELINES.common import distributed
        if distributed.active():
            from torch.distributed.nn.functional import all_gather
            a,v=torch.cat(all_gather(a)),torch.cat(all_gather(v))
        contrastive,_=model.forward_contrastive(a,v,bidirect_contrast=True)
        return reconstruction+contrastive

    def export_encoder(self):
        return {k:v for k,v in self.encoder.state_dict().items() if not k.startswith(('decoder_','mask_token'))}

    def gradient_components(self):
        modules=super().gradient_components()
        modules.update(multimodal_first=self.encoder.mm_layer_1,multimodal_second=self.encoder.mm_layer_2)
        if self.pretraining:
            modules.update(audio_reconstruction=self.encoder.decoder_pred_a,video_reconstruction=self.encoder.decoder_pred_v)
        return modules
