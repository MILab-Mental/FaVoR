"""Offline conversion/validation; training never repairs pairs.

python -m datasets.va_lejepa.manifest_tools convert INPUT OUTPUT --reject-report rejects.csv
python -m datasets.va_lejepa.manifest_tools validate OUTPUT --reject-report rejects.csv
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path
from .va_manifest import source_key, validate_identity, read_manifest
from .av_decoder import probe_media, validate_sync


def convert(input_path, output_path, reject_path, source_pattern=None, limit=None):
    rows, rejects, seen = [], [], set()
    with Path(input_path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if limit and line_number > limit:
                break
            if not line.strip():
                continue
            try:
                paths, label = line.strip().rsplit(None, 1)
                float(label)
                boundaries = list(re.finditer(r'\.(?:mp4|avi|mov|mkv|webm)(?=\s+)', paths, re.I))
                if len(boundaries) != 1:
                    raise ValueError('ambiguous video/audio extension boundary')
                boundary = boundaries[0].end()
                video, audio = paths[:boundary], paths[boundary:].strip()
                if Path(audio).suffix.lower() not in {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}:
                    raise ValueError('unsupported audio extension')
                root = Path(input_path).resolve().parent
                video, audio = [str((root / p).resolve()) if not Path(p).is_absolute() else p for p in (video, audio)]
                row = dict(pair_id=hashlib.sha256(f'{video}\0{audio}'.encode()).hexdigest()[:24], source_id=source_key(video, source_pattern),
                           video_path=video, audio_path=audio, label=label)
                validate_identity(row, source_pattern)
                if row['pair_id'] in seen:
                    raise ValueError('duplicate pair')
                seen.add(row['pair_id'])
                rows.append(row)
            except Exception as error:
                rejects.append(dict(file=str(input_path), line=line_number, reason=str(error), raw=line.strip()))
    _write(output_path, rows, ['pair_id', 'source_id', 'video_path', 'audio_path', 'label'])
    _write(reject_path, rejects, ['file', 'line', 'reason', 'raw'])
    return len(rows), len(rejects)


def _write(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t' if str(path).endswith('.tsv') else ',')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['convert', 'validate'])
    parser.add_argument('input')
    parser.add_argument('output', nargs='?')
    parser.add_argument('--reject-report', required=True)
    parser.add_argument('--source-pattern')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--duration-tolerance', type=float, default=.02)
    parser.add_argument('--minimum-seconds', type=float, default=0.1)
    parser.add_argument('--minimum-fps', type=float, default=8.0)
    args = parser.parse_args()
    if args.command == 'convert':
        if not args.output:
            parser.error('convert requires output')
        print(dict(zip(('accepted', 'rejected'), convert(args.input, args.output, args.reject_report, args.source_pattern, args.limit))))
    else:
        rows = read_manifest(args.input, source_pattern=args.source_pattern)
        rejects, accepted = [], []
        for index, row in enumerate(rows[:args.limit] if args.limit else rows, 2):
            try:
                video, audio = probe_media(row['video_path'], 'video'), probe_media(row['audio_path'], 'audio')
                diagnostics = validate_sync(video, audio, row, {'duration_tolerance_seconds': args.duration_tolerance})
                if video['duration'] < args.minimum_seconds or audio['duration'] < args.minimum_seconds:
                    raise ValueError(f'media is shorter than minimum {args.minimum_seconds}s valid interval')
                if video['fps'] < args.minimum_fps:
                    raise ValueError(f'video FPS {video["fps"]} would require repeated frames')
                accepted.append(row)
                print(row['pair_id'], diagnostics)
            except Exception as error:
                rejects.append(dict(file=args.input, line=index, reason=str(error), raw=row['pair_id']))
        _write(args.reject_report, rejects, ['file', 'line', 'reason', 'raw'])
        if args.output:
            _write(args.output, accepted, list(rows[0]))
        if rejects:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
