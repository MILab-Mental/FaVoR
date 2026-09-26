"""Offline conversion/validation; training never repairs pairs.

python -m datasets.va_lejepa.manifest_tools convert INPUT OUTPUT --reject-report rejects.csv
python -m datasets.va_lejepa.manifest_tools validate OUTPUT --reject-report rejects.csv

Progress bars are enabled by default; use --no-progress for quiet batch jobs.
Validation prints a summary; --verbose additionally prints each accepted pair.
"""
import argparse
import csv
import hashlib
import re
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from .va_manifest import source_key, validate_identity, read_manifest
from .av_decoder import probe_media, validate_sync
from .progress import progress_lines


def convert(input_path, output_path, reject_path, source_pattern=None, limit=None, *, progress=True):
    rows, rejects, seen = [], [], set()
    with Path(input_path).open(encoding='utf-8', newline='') as handle, progress_lines(handle, input_path, 'Convert manifest', progress) as lines:
        for line_number, line in enumerate(islice(lines, limit) if limit else lines, 1):
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
    _write(output_path, rows, ['pair_id', 'source_id', 'video_path', 'audio_path', 'label'], progress=progress)
    _write(reject_path, rejects, ['file', 'line', 'reason', 'raw'], progress=progress)
    return len(rows), len(rejects)


def _write(path, rows, fields, *, progress=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t' if str(path).endswith('.tsv') else ',')
        writer.writeheader()
        writer.writerows(tqdm(rows, desc=f'Write {Path(path).name}', unit='row',
                             dynamic_ncols=True, disable=not progress or not rows))


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
    parser.add_argument('--no-progress', action='store_true', help='disable progress bars')
    parser.add_argument('--verbose', action='store_true', help='print diagnostics for every accepted media pair')
    args = parser.parse_args()
    if args.command == 'convert':
        if not args.output:
            parser.error('convert requires output')
        print(dict(zip(('accepted', 'rejected'), convert(args.input, args.output, args.reject_report, args.source_pattern, args.limit, progress=not args.no_progress))))
    else:
        rows = read_manifest(args.input, source_pattern=args.source_pattern, progress=not args.no_progress)
        rejects, accepted = [], []
        pairs = tqdm(rows[:args.limit] if args.limit else rows, desc='Validate media', unit='pair',
                     dynamic_ncols=True, disable=args.no_progress)
        for index, row in enumerate(pairs, 2):
            try:
                video, audio = probe_media(row['video_path'], 'video'), probe_media(row['audio_path'], 'audio')
                diagnostics = validate_sync(video, audio, row, {'duration_tolerance_seconds': args.duration_tolerance})
                if video['duration'] < args.minimum_seconds or audio['duration'] < args.minimum_seconds:
                    raise ValueError(f'media is shorter than minimum {args.minimum_seconds}s valid interval')
                if video['fps'] < args.minimum_fps:
                    raise ValueError(f'video FPS {video["fps"]} would require repeated frames')
                accepted.append(row)
                if args.verbose:
                    tqdm.write(f'{row["pair_id"]} {diagnostics}')
            except Exception as error:
                rejects.append(dict(file=args.input, line=index, reason=str(error), raw=row['pair_id']))
            pairs.set_postfix(accepted=len(accepted), rejected=len(rejects), refresh=False)
        _write(args.reject_report, rejects, ['file', 'line', 'reason', 'raw'], progress=not args.no_progress)
        if args.output:
            _write(args.output, accepted, list(rows[0]), progress=not args.no_progress)
        print(dict(accepted=len(accepted), rejected=len(rejects)))
        if rejects:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
