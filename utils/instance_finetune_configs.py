"""Select instance presets for long-media tasks, including regenerated tasks."""
from pathlib import Path


INSTANCE_PRESETS = {
    'finetune_v': ('video', '48frames-8fps-8sentences.yaml', 'vit-l-4layers-instance.yaml'),
    'finetune_a': ('audio', 'audio-4s-8sentences.yaml', 'emotion2vec-plus-large-instance.yaml'),
    'finetune_va': ('audio-video', 'va-6s-8sentences.yaml', 'vitl-emotion2vec-instance.yaml'),
}


def is_long_media_task(data):
    return any(Path(path).stem.endswith(('_LA', '_LV')) for path in data.get('datasets', []))


def select_instance_presets(config, data):
    if not is_long_media_task(data):
        return config
    modality, sampling, model = INSTANCE_PRESETS[config['app']]
    config['yamls']['sampling'] = f'CONFIGS/data/finetune/{modality}/sampling/{sampling}'
    config['yamls']['model'] = f'CONFIGS/models/finetune/{modality}/{model}'
    suffix = '/instance-sentences8'
    folder = config['folder'].rstrip('/')
    config['folder'] = folder if folder.endswith(suffix) else folder + suffix
    if config['app'] == 'finetune_va':
        config['finetune'].update(freeze_video_encoder=False, freeze_audio_encoder=False,
                                   freeze_fusion_adapter=False)
    else:
        config.setdefault('meta', {})['frozen_encoder'] = False
    return config
