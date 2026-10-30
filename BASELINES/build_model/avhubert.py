"""AV-HuBERT's native encoder and CPU mouth preprocessing on real CSV frames."""
from collections import OrderedDict
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F

from BASELINES.common.official import official
from BASELINES.common.runtime import path, fingerprint


@lru_cache(maxsize=1)
def mouth_resources(predictor,detector,mean_face):
    import dlib
    import numpy as np
    if dlib.DLIB_USE_CUDA:
        raise RuntimeError('Use the CPU dlib build; data workers must not initialize CUDA')
    with official('av_hubert','avhubert/preparation') as source:
        alignment=source('align_mouth')
    return (dlib.get_frontal_face_detector(),dlib.cnn_face_detection_model_v1(detector),
            dlib.shape_predictor(predictor),np.load(mean_face,allow_pickle=False),alignment)


_mouth_cache=OrderedDict()


def crop_lips(frames,assets,cache_key=None):
    """Native 68-point alignment and 96px mouth crop, entirely in RAM."""
    import cv2
    import numpy as np
    filenames=tuple(str(path(assets[k])) for k in ('predictor','detector','mean_face'))
    key=(cache_key,filenames)
    if cache_key is not None and key in _mouth_cache:
        _mouth_cache.move_to_end(key)
        return _mouth_cache[key]
    detector,cnn,predictor,mean_face,alignment=mouth_resources(*filenames)
    landmarks=[]
    # Stored dataset landmarks use original HD coordinates, whereas CSV videos
    # contain 112px faces. Detect in these actual decoded pixels, never reuse
    # untransformed coordinates or pass a whole face as a mouth ROI.
    for frame in frames.numpy():
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        rectangles=detector(gray,1)
        if not rectangles:rectangles=[d.rect for d in cnn(gray)]
        points=None
        if rectangles:
            shape=predictor(gray,rectangles[-1])
            points=np.asarray([(shape.part(i).x,shape.part(i).y) for i in range(68)],dtype=np.float32)
        landmarks.append(points)
    landmarks=alignment.landmarks_interpolate(landmarks)
    if landmarks is None:raise ValueError('No real face landmarks detected for mouth preprocessing: '+str(cache_key))
    read_video,count_frames=alignment.read_video,alignment.get_frame_count
    # Reuse the complete official smoothing/warp/crop algorithm while replacing
    # its file reader with already decoded RGB frames; no temporary videos.
    alignment.read_video=lambda _:iter(frames.numpy())
    alignment.get_frame_count=lambda _:len(frames)
    try:
        mouth=alignment.crop_patch('memory',landmarks,mean_face,[33,36,39,42,45],
                                  (256,256),12,48,68,96,96)
    finally:
        alignment.read_video,alignment.get_frame_count=read_video,count_frames
    if mouth is None or mouth.shape!=(len(frames),96,96,3) or not mouth.std()>0:
        raise ValueError('Invalid native mouth crop: '+str(cache_key))
    result=torch.from_numpy(mouth)
    if cache_key is not None:
        _mouth_cache[key]=result
        while len(_mouth_cache)>64:_mouth_cache.popitem(last=False)
    return result


class AVHubert(nn.Module):
    def gradient_components(self):
        return dict(audio_features=self.encoder.feature_extractor_audio,
                    video_features=self.encoder.feature_extractor_video)

    def __init__(self, cfg, initialize=True):
        super().__init__()
        import fairseq
        # The release selects relative package imports when argv has arguments,
        # but flat debug imports in notebooks. Always select its package path.
        import sys
        with official('av_hubert',keep_packages=('fairseq',)) as source:
            argv=sys.argv
            try:
                sys.argv=[*argv,'--baseline-package-import']
                source('avhubert')
            finally:sys.argv=argv
            from functools import partial
            original_load=torch.load
            # This pinned pre-2.6 loader restores Fairseq dictionaries/configs
            # from the SHA256-checked official release, beyond tensor-only state.
            torch.load=partial(original_load,weights_only=False)
            try:
                models, checkpoint_cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                    [str(path(cfg['weights']))], strict=True)
            finally:torch.load=original_load
            self.encoder = models[0]
        self.encoder.remove_pretraining_modules()
        # The label embedding table is used only for the upstream cluster loss.
        self.encoder.label_embs_concat = None
        self.encoder.mask_emb = None
        self.encoder.feature_grad_mult = 1.0
        self.encoder.modality_dropout=float(cfg.get('modality_dropout',0.0))
        self.output_dim = self.encoder.encoder_embed_dim
        if self.output_dim != 1024:
            raise ValueError('AV-HuBERT requires Large clean PT')
        self.input_spec = dict(modality='audio-video', sample_rate=16000, seconds=4,
                              frames=100, fps=25, size=88, requires_lips=True)
        if cfg.get('mouth_preprocessing'):
            assets=cfg['mouth_preprocessing']
            for key,digest in cfg.get('mouth_preprocessing_sha256',{}).items():
                if fingerprint(assets[key])!=digest:raise ValueError('AV-HuBERT preprocessing asset SHA256 differs: '+key)
            self.input_spec['lip_preprocessing']=assets

    def prepare(self, clip):
        if not clip.get('lip_roi'):raise ValueError('AV-HuBERT requires a real mouth ROI from the data pipeline')
        from python_speech_features import logfbank
        from torchvision.transforms import functional as TF
        import numpy as np
        # Official logfbank uses PCM16 amplitude, then stacks four 100 Hz frames.
        waveform = (clip['audio'].numpy() * 32768).clip(-32768, 32767).astype(np.int16)
        audio = torch.from_numpy(logfbank(waveform, samplerate=16000).astype('float32'))
        audio = F.pad(audio, (0, 0, 0, (-len(audio)) % 4)).reshape(-1, 104)
        audio = F.layer_norm(audio, audio.shape[1:])
        frames = clip['video'].permute(0, 3, 1, 2).float() / 255
        if min(frames.shape[-2:]) < 88:
            raise ValueError('Lip ROI must be at least 88x88; do not pad a missing crop')
        video=TF.rgb_to_grayscale(frames)
        if self.training:
            from torchvision.transforms import RandomCrop
            video=TF.crop(video,*RandomCrop.get_params(video,(88,88)))
            if torch.rand(())<.5:video=TF.hflip(video)
        else:video=TF.center_crop(video,[88,88])
        video=(video-.421)/.165
        # The source clip can be 24/30fps. Match genuine ROI frames to the
        # native stacked-audio 25Hz timeline rather than discard its tail.
        if len(video)!=len(audio):
            video=video[torch.linspace(0,len(video)-1,len(audio)).round().long()]
        length=len(audio)
        return dict(audio=audio.T[None], video=video.permute(1, 0, 2, 3)[None],
                    padding_mask=torch.zeros(1, length, dtype=torch.bool))

    def batch_inputs(self, samples):
        length=max(s['audio'].shape[-1] for s in samples)
        return dict(
            audio=torch.cat([F.pad(s['audio'],(0,length-s['audio'].shape[-1])) for s in samples]),
            video=torch.cat([F.pad(s['video'],(0,0,0,0,0,length-s['video'].shape[2])) for s in samples]),
            padding_mask=torch.cat([F.pad(s['padding_mask'],(0,length-s['padding_mask'].shape[-1]),value=True)
                                    for s in samples]))

    def forward(self, batch):
        x, mask = self.encoder.extract_features(source={k: batch[k] for k in ('audio', 'video')},
                                                padding_mask=batch['padding_mask'], mask=False)
        valid = (~mask)[..., None] if mask is not None else torch.ones_like(x[..., :1])
        return (x*valid).sum(1) / valid.sum(1).clamp_min(1)

