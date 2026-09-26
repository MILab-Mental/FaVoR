import copy
import logging
import torch
from torch import nn
from models.video_lejepa import Projector, LeJEPALoss
from models.video_lejepa.encoder import build_encoder
from models.audio_lejepa import AudioLeEncoder
from .encoders import video_token_output, audio_token_output
from .temporal_alignment import align_audio_to_video
from .fusion_base import BRIDGE_REGISTRY, ModalityInputAdapters, ConcatTransformerFusion
from .checkpoint import load_video_encoder_init, load_audio_encoder_init

LOGGER = logging.getLogger(__name__)


class VALeJEPA(nn.Module):
    def __init__(self, video_encoder, audio_encoder, fusion_cfg, sample_rate=16000, audio_checkpointing=True):
        super().__init__()
        self.video_encoder, self.audio_encoder = video_encoder, audio_encoder
        self.sample_rate = int(sample_rate)
        self.audio_checkpointing = audio_checkpointing
        enabled = fusion_cfg.get('enabled', ['feature'])
        self.enabled_modes = [enabled] if isinstance(enabled, str) else list(enabled)
        if not self.enabled_modes or len(set(self.enabled_modes)) != len(self.enabled_modes) or set(self.enabled_modes) - {'early', 'feature', 'late'}:
            raise ValueError('fusion.enabled must contain unique early/feature/late modes')
        if fusion_cfg.get('bridge', 'concat_transformer') not in BRIDGE_REGISTRY:
            raise ValueError('unsupported fusion bridge; v1 implements concat_transformer')
        if not fusion_cfg.get('modality_embedding', True) or not fusion_cfg.get('fuse_token', True):
            raise ValueError('modality embeddings and FUSE token are required')
        if fusion_cfg.get('temporal_embedding', 'sincos') != 'sincos':
            raise ValueError('v1 requires sincos temporal embeddings')
        self.common_dim = int(fusion_cfg.get('common_dim', 768))
        self.input_adapters, self.fusion_adapters, self.loss_projectors = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        self.loss_weights = {}
        for mode in self.enabled_modes:
            cfg = fusion_cfg.get(mode, {})
            self.input_adapters[mode] = ModalityInputAdapters(video_encoder.embed_dim, audio_encoder.embed_dim,
                self.common_dim, int(fusion_cfg.get('input_adapter', {}).get('hidden_dim', 2 * self.common_dim)))
            self.fusion_adapters[mode] = ConcatTransformerFusion(mode, self.common_dim, cfg)
            # LayerNorm supports the hardware calibration batch=1; projector implementation is shared.
            self.loss_projectors[mode] = Projector(self.common_dim, int(fusion_cfg.get('projector_hidden_dim', 2048)),
                int(fusion_cfg.get('projection_dim', 256)), norm_layer=nn.LayerNorm)
            self.loss_weights[mode] = float(cfg.get('loss_weight', 1))
        if any(weight < 0 for weight in self.loss_weights.values()) or not any(self.loss_weights.values()):
            raise ValueError('branch loss weights must be nonnegative with at least one positive')
        self.initialization_report = {}

    def train(self, mode=True):
        super().train(mode)
        for encoder in (self.video_encoder, self.audio_encoder):
            if not any(p.requires_grad for p in encoder.parameters()):
                encoder.eval()
        return self

    def encode_view(self, view, return_alignment=False):
        lengths = view.get('video_lengths')
        if lengths is None:
            return self._encode_valid_view(view, return_alignment)
        if lengths.ndim != 1 or lengths.shape[0] != view['video'].shape[0]:
            raise ValueError('video_lengths must be [B]')
        if (lengths < 1).any() or (lengths > view['video'].shape[2]).any():
            raise ValueError('invalid video_lengths')
        # Padding must not enter the video backbone: CLS pooling and attention
        # would otherwise incorporate zero frames. Group real frame counts first.
        results, alignments, indices = [], [], []
        audio_lengths = view['audio_lengths']
        if (audio_lengths < 1).any() or (audio_lengths > view['audio'].shape[1]).any():
            raise ValueError('invalid audio_lengths')
        groups = torch.stack((lengths, audio_lengths), dim=1).unique(dim=0, sorted=True)
        for length, audio_length in groups.tolist():
            ids = torch.where((lengths == length) & (audio_lengths == audio_length))[0]
            cropped = {key: value.index_select(0, ids) for key, value in view.items()}
            cropped['video'] = cropped['video'][:, :, :length]
            for key in ('video_frame_times', 'video_frame_indices', 'video_padding_mask'):
                if key in cropped:
                    cropped[key] = cropped[key][:, :length]
            # Remove unused audio tail; real per-sample lengths still mask tokens.
            cropped['audio'] = cropped['audio'][:, :audio_length]
            if 'audio_padding_mask' in cropped:
                cropped['audio_padding_mask'] = cropped['audio_padding_mask'][:, :audio_length]
            output = self._encode_valid_view(cropped, return_alignment)
            if return_alignment:
                output, alignment = output
                alignments.append((ids, alignment))
            results.append(output)
            indices.append(ids)
        order = torch.cat(indices).argsort()
        representations = {mode: torch.cat([result[mode] for result in results], 0)[order]
                           for mode in self.enabled_modes}
        if return_alignment:
            # A mixed-length batch has different physical bin grids per group.
            alignment = alignments[0][1] if len(alignments) == 1 else alignments
            return representations, alignment
        return representations

    def _encode_valid_view(self, view, return_alignment=False):
        video = video_token_output(self.video_encoder, view)
        audio = audio_token_output(self.audio_encoder, view, self.sample_rate, self.audio_checkpointing)
        aligned = align_audio_to_video(video, audio, torch.zeros_like(view['start_time'], dtype=torch.float64),
                                       (view['end_time'] - view['start_time']).double())
        representations = {mode: self.fusion_adapters[mode](aligned, self.input_adapters[mode]) for mode in self.enabled_modes}
        return (representations, aligned) if return_alignment else representations

    def forward(self, batch):
        # Execute locals one at a time; all fusion branches reuse each encoder execution.
        representations = [self.encode_view(batch['global'])]
        local = batch['local']
        for k in range(local['video'].shape[1]):
            representations.append(self.encode_view({key: value[:, k] for key, value in local.items()}))
        output = {}
        for mode in self.enabled_modes:
            z = self.loss_projectors[mode](torch.stack([rep[mode] for rep in representations], 1))
            output[mode] = {'global': z[:, :1], 'local': z[:, 1:]}
        return output


class VABranchLoss(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        aux = cfg.get('auxiliary', {})
        if aux.get('video_lejepa_weight', 0) or aux.get('audio_lejepa_weight', 0):
            raise ValueError('v1 auxiliary loss requires separately initialized unimodal projectors and is not implemented')
        self.weights = dict(model.loss_weights)
        self.criteria = nn.ModuleDict()
        for mode in model.enabled_modes:
            sigreg = {**cfg.get('sigreg', {}), **cfg.get('branches', {}).get(mode, {}).get('sigreg', {})}
            self.criteria[mode] = LeJEPALoss(sigreg_weight=sigreg.get('weight', .02),
                knots=sigreg.get('knots', 17), num_proj=sigreg.get('num_proj', 1024),
                normalize_by_n=sigreg.get('normalize_by_n', False))

    def forward(self, output):
        losses = {}
        for mode, value in output.items():
            embeddings = torch.cat((value['global'], value['local']), 1).float()
            # Keep ECF/cos/sin integration in float32 even inside the trainer's AMP region.
            with torch.autocast(embeddings.device.type, enabled=False):
                losses[mode] = self.criteria[mode](embeddings)
        return {'loss': sum(self.weights[mode] * value['loss'] for mode, value in losses.items()), 'branches': losses}


def build_va_lejepa(model_cfg, data_cfg, initialize=True):
    cfg = copy.deepcopy(model_cfg)
    video_cfg, audio_cfg = cfg['video'], cfg['audio']
    if cfg.get('modality_dropout', {}).get('enabled', False):
        raise ValueError('modality dropout is a future extension')
    if video_cfg.get('tubelet_size', 1) != 1 or video_cfg.get('token_drop_rate', 0) != 0 or video_cfg.get('attn_mode', 'block_causal') != 'block_causal':
        raise ValueError('VA video requires tubelet=1, dense tokens, block_causal')
    video = build_encoder(model_name=video_cfg.get('name', 'vit_large'),
        img_size=int(data_cfg.get('video_global_size', 112)), num_frames=int(data_cfg.get('video_frames_global', 48)),
        patch_size=int(video_cfg.get('patch_size', 16)), tubelet_size=1, token_drop_rate=0,
        attn_mode='block_causal', use_rope=video_cfg.get('use_rope', True), use_sdpa=video_cfg.get('use_sdpa', True),
        use_activation_checkpointing=video_cfg.get('use_activation_checkpointing', True))
    assert video.use_cls_token
    seconds = float(data_cfg.get('global_seconds', 6))
    audio_cfg['audio'] = dict(sample_rate=int(data_cfg.get('sample_rate', 16000)), process_seconds=seconds, max_process_seconds=seconds)
    audio = AudioLeEncoder(audio_cfg)
    model = VALeJEPA(video, audio, cfg['fusion'], audio_cfg['audio']['sample_rate'], cfg.get('audio_gradient_checkpointing', True))
    for module, frozen in ((video, video_cfg.get('freeze', False)), (audio, audio_cfg.get('freeze', False))):
        if frozen:
            module.requires_grad_(False)
    if initialize:
        model.initialization_report = {'video': load_video_encoder_init(video, video_cfg), 'audio': load_audio_encoder_init(audio, audio_cfg)}
    LOGGER.info('VA actual encoders: video=%s dim=%d CLS=%s frames=%d size=%d patch=%d tubelet=1 drop=0; audio dim=%d sr=%d max_seconds=%s',
        video_cfg.get('name', 'vit_large'), video.embed_dim, video.use_cls_token, video.num_frames,
        data_cfg.get('video_global_size', 112), video.patch_size, audio.embed_dim, model.sample_rate, seconds)
    return model
