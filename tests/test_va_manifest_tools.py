import csv
import sys

import pytest

from datasets.va_lejepa import manifest_tools
from datasets.va_lejepa.va_manifest import read_manifest, source_key, validate_identity


def test_ravdess_modality_is_not_recording_identity(tmp_path):
    video = '/data/RAVDESS/videos/02-01-06-01-02-01-12.mp4'
    audio = '/data/RAVDESS/audios/03-01-06-01-02-01-12.wav'
    legacy = tmp_path / 'legacy.txt'
    legacy.write_text(f'{video} {audio} 0\n')
    canonical = tmp_path / 'canonical.csv'
    assert manifest_tools.convert(legacy, canonical, tmp_path / 'rejects.csv', progress=False) == (1, 0)
    row = read_manifest(canonical)[0]
    assert row['source_id'] == 'ravdess:01-06-01-02-01-12'
    # Actor, repetition, statement, intensity, emotion and channel still identify
    # distinct recordings; only the modality field can differ.
    for field in range(1, 7):
        parts = audio.rsplit('/', 1)[1].removesuffix('.wav').split('-')
        parts[field] = '99'
        with pytest.raises(ValueError, match='identity mismatch'):
            validate_identity(dict(row, audio_path='/data/RAVDESS/audios/' + '-'.join(parts) + '.wav'))


def test_modality_normalization_is_scoped_and_explicit_pattern_wins():
    video = '/data/other/02-01-06-01-02-01-12.mp4'
    audio = '/data/other/03-01-06-01-02-01-12.wav'
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_identity(dict(pair_id='pair', source_id=source_key(video), video_path=video, audio_path=audio))
    assert source_key('/data/RAVDESS/02-01-06-01-02-01-12.mp4', r'(.*)') == '02-01-06-01-02-01-12'


def test_conversion_progress_limit_and_multiline_csv_read(tmp_path, capsys):
    legacy = tmp_path / 'legacy.txt'
    legacy.write_text('/a space/x.mp4 /b space/x.wav 0\n/a/y.mp4 /b/y.wav 1\n')
    canonical = tmp_path / 'canonical.csv'
    assert manifest_tools.convert(legacy, canonical, tmp_path / 'rejects.csv', limit=1) == (1, 0)
    rows = read_manifest(canonical, progress=True)
    assert len(rows) == 1 and rows[0]['source_id'] == 'x'
    stderr = capsys.readouterr().err
    assert 'Convert manifest' in stderr and 'Read manifest' in stderr
    # Tracking physical lines must preserve DictReader's quoted multiline fields.
    rows[0]['label'] = 'first\nsecond'
    manifest_tools._write(canonical, rows, list(rows[0]))
    assert read_manifest(canonical, progress=True)[0]['label'] == 'first\nsecond'


@pytest.mark.parametrize('no_progress,verbose', [(False, False), (True, True)])
def test_validate_cli_progress_summary_and_reject_exit(tmp_path, monkeypatch, capsys, no_progress, verbose):
    canonical, valid, rejects = [tmp_path / name for name in ('canonical.csv', 'valid.csv', 'rejects.csv')]
    rows = [dict(pair_id=name, source_id=name, video_path=f'/data/{name}.mp4',
                 audio_path=f'/data/{name}.wav') for name in ('good', 'bad')]
    manifest_tools._write(canonical, rows, list(rows[0]))

    def probe(path, kind):
        return dict(duration=1.0 if 'good' in path or kind == 'video' else 2.0,
                    start=0.0, fps=25.0, sample_rate=16000)

    monkeypatch.setattr(manifest_tools, 'probe_media', probe)
    argv = ['manifest_tools', 'validate', str(canonical), str(valid), '--reject-report', str(rejects)]
    if no_progress:
        argv.append('--no-progress')
    if verbose:
        argv.append('--verbose')
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as error:
        manifest_tools.main()
    assert error.value.code == 1
    assert [row['pair_id'] for row in read_manifest(valid)] == ['good']
    with rejects.open() as handle:
        assert 'sync mismatch' in next(csv.DictReader(handle))['reason']
    output = capsys.readouterr()
    assert "'accepted': 1, 'rejected': 1" in output.out
    assert ('video_duration' in output.out) == verbose
    assert ('Validate media' in output.err) != no_progress
