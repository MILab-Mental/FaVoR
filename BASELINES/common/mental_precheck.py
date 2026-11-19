"""One shared media precheck for the 38 clinical fine-tuning tasks."""
import argparse
import csv
import fcntl
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from BASELINES.common.runtime import OUTPUT_ROOT, path, write_json
from datasets.common.paths import get_dataset_paths


MANIFEST = 'DATASET/splits-0901/csvs/3_私有临床数据_LV.csv'
SCHEMA_VERSION = 1


def _signature(manifest, root):
    stat = manifest.stat()
    return dict(version=SCHEMA_VERSION, manifest=str(manifest), root=str(root),
                size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _media_status(row, root):
    """Return paired readability and AV eligibility with one probe per file."""
    import soundfile
    from decord import VideoReader, cpu
    from BASELINES.common.data import _ffprobe_duration, _SOUNDFILE_SUFFIXES

    if not all(row[field].strip() for field in ('audio_path', 'video_path')):
        return False, False
    audio, video = (root / row[field] for field in ('audio_path', 'video_path'))
    if '|' in str(audio) or '|' in str(video):
        raise ValueError('Multi-path row needs an explicit segment manifest')
    try:
        if audio.stat().st_size < 1024:
            return False, False
        if audio.suffix.lower() in _SOUNDFILE_SUFFIXES:
            info = soundfile.info(audio)
            if info.frames <= 0:
                return False, False
            audio_seconds = info.frames / info.samplerate
    except Exception:
        return False, False
    if audio.suffix.lower() not in _SOUNDFILE_SUFFIXES:
        try:
            audio_seconds = _ffprobe_duration(audio)
        except Exception:
            audio_seconds = None
    try:
        video_size = video.stat().st_size
        reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
        frames, fps = len(reader), float(reader.get_avg_fps())
    except Exception:
        return False, False
    paired = video_size >= 1024 and frames > 0
    synced = audio_seconds is not None and fps > 0 and abs(frames / fps - audio_seconds) <= max(0.5, 2 / fps)
    return paired, synced


def _task_summary(rows, bad_paired, bad_av, tasks):
    from datasets.common.instances import sentence_identity

    recordings = [sentence_identity(row['video_path'])[0] for row in rows]
    splits = {'0': 'train', '1': 'test', '2': 'validation'}
    summary = {}
    for name, cfg in tasks.items():
        label = cfg['label_column']
        paired, av = Counter(), Counter()
        group_labels = {}
        for index, row in enumerate(rows):
            split = row[label + '_split']
            if split in ('', '-1'):
                continue
            if split not in splits or not 1 <= int(row[label]) <= int(cfg['num_class']):
                raise ValueError(f'{name}: invalid label or split on CSV line {index + 2}')
            key = recordings[index]
            value = split, int(row[label])
            if group_labels.setdefault(key, value) != value:
                raise ValueError(f'{name}: recording labels or splits disagree on CSV line {index + 2}')
            if index not in bad_paired:
                paired[splits[split]] += 1
            if index not in bad_av:
                av[splits[split]] += 1
        if not all(counts[s] for counts in (paired, av) for s in ('train', 'test')):
            raise ValueError(f'{name}: no usable train or test records')
        summary[name] = {'paired': dict(paired), 'av': dict(av)}
    return summary


def load_mental_precheck(root, *, refresh=False):
    """Share a cache across processes and rebuild when the CSV or root changes."""
    from BASELINES.common.experiments import task_catalog

    manifest = path(MANIFEST).resolve()
    root = Path(root).resolve()
    output = OUTPUT_ROOT / 'mental-precheck.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = _signature(manifest, root)
        if not refresh:
            try:
                cached = json.loads(output.read_text())
                if cached['source'] == source:
                    return cached
            except (OSError, ValueError, KeyError, TypeError):
                pass

        tasks = {name: cfg for name, cfg in task_catalog().items() if name.startswith('mental/')}
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = {'audio_path', 'video_path'}
            for task in tasks.values():
                name = task['label_column']
                columns.update((name, name + '_split'))
            if missing := columns - set(reader.fieldnames or ()):
                raise ValueError(f'Missing clinical CSV columns: {sorted(missing)}')
            rows = list(reader)

        print(f'MENTAL PRECHECK scanning {len(rows)} media pairs', flush=True)
        bad_paired, bad_av = set(), set()
        for index, row in enumerate(rows):
            paired, av = _media_status(row, root)
            if not paired:
                bad_paired.add(index)
            if not av:
                bad_av.add(index)
            if (index + 1) % 5000 == 0:
                print(f'MENTAL PRECHECK {index + 1}/{len(rows)}', flush=True)
        summary = _task_summary(rows, bad_paired, bad_av, tasks)
        if source != _signature(manifest, root):
            raise RuntimeError('Clinical CSV changed during precheck')
        report = dict(source=source, created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                      rows=len(rows), invalid_paired_rows=sorted(bad_paired),
                      invalid_av_rows=sorted(bad_av), tasks=summary)
        write_json(output, report)
        print(f'MENTAL PRECHECK wrote {output}', flush=True)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Scan again after replacing media files')
    args = parser.parse_args()
    load_mental_precheck(get_dataset_paths([MANIFEST])[0], refresh=args.refresh)
