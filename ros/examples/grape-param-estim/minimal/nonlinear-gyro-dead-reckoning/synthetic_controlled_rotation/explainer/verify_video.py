#!/usr/bin/env python3
"""Inspect encoded dimensions, audio, actual scene frames and publication size."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(publication, work):
    publication, work = Path(publication), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    record = json.loads((publication/'verification.json').read_text())
    timing = {row['id']:row for row in json.loads((publication/'scene_timing.json').read_text())}
    voices = json.loads((publication/'narration_manifest.json').read_text())
    chapters, previews = [], []
    for chapter in record['chapters']:
        video = publication/chapter['file']
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-show_chapters', '-of', 'json', str(video)], text=True))
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video),
            '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
        v = next(row for row in info['streams'] if row['codec_type'] == 'video')
        a = next(row for row in info['streams'] if row['codec_type'] == 'audio')
        pcm = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video),
            '-map', '0:a:0', '-ac', '1', '-ar', '24000', '-f', 's16le', 'pipe:1'])
        values = np.frombuffer(pcm, dtype='<i2').astype(float)
        statistics = dict(peak_pcm16=float(np.max(np.abs(values))),
            rms_pcm16=float(np.sqrt(np.mean(values**2))),
            full_scale_sample_count=int(np.count_nonzero(np.abs(values) >= 32767)),
            decoded_audio_seconds=len(values)/24000)
        chapter_frames = []
        start = 0.
        for sid in chapter['scene_ids']:
            duration = timing[sid]['duration_seconds']
            stamp = start+.55*duration
            image_path = work/(sid+'.png')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{stamp:.6f}',
                '-i', str(video), '-frames:v', '1', str(image_path)], check=True)
            image = Image.open(image_path).convert('RGB')
            pixels = np.asarray(image, float)
            frame = dict(id=sid, title=timing[sid]['title'], chapter=chapter['file'],
                time_seconds=stamp, size=list(image.size), pixel_std=float(np.std(pixels)),
                sha256=sha(image_path))
            chapter_frames.append(frame)
            previews.append((sid, timing[sid]['title'], image_path))
            start += duration
        chapters.append(dict(file=chapter['file'], bytes=video.stat().st_size,
            dimensions=[v['width'], v['height']], fps=v['avg_frame_rate'],
            video_codec=v['codec_name'], audio_codec=a['codec_name'],
            duration_seconds=float(info['format']['duration']), audio=statistics,
            complete_video_audio_decode_succeeded=True,
            embedded_chapters=info.get('chapters', []),
            scene_frames=chapter_frames))
    ratios = [row['timing']['measured_normal_to_final_ratio'] for row in voices['scenes']]
    removed_peaks = [row['timing']['removed_tail_peak_pcm16'] for row in voices['scenes']]
    checks = dict(schema='grape/synthetic-encoded-video-check/v1', chapters=chapters,
        all_dimensions_960_540=all(row['dimensions'] == [960, 540] for row in chapters),
        all_files_under_50MB=all(row['bytes'] < 50_000_000 for row in chapters),
        japanese_voice=voices['voice'], declared_speed_multiplier=voices['speed_multiplier'],
        measured_speed_ratio_min=min(ratios), measured_speed_ratio_max=max(ratios),
        maximum_trimmed_audio_tail_peak_pcm16=max(removed_peaks),
        all_chapters_have_nonzero_audio=all(row['audio']['rms_pcm16'] > 0. for row in chapters),
        narration_scene_count=len(voices['scenes']), inspected_encoded_scene_count=len(previews),
        note='Audio signal/encoding and frame checks complement manual Japanese-script and visual review; no ASR verification is claimed.')
    (publication/'encoded_checks.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False)+'\n')
    font = ImageFont.truetype(FONT, 15)
    # Contact sheets stay in the work directory; individual full-resolution
    # encoded frames remain available for detailed visual inspection.
    sheets = []
    for first in range(0, len(previews), 12):
        selected = previews[first:first+12]
        canvas = Image.new('RGB', (3*480, 4*310), 'white')
        drawing = ImageDraw.Draw(canvas)
        for index, (sid, title, path) in enumerate(selected):
            x, y = (index%3)*480, (index//3)*310
            image = Image.open(path).convert('RGB').resize((480, 270), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))
            drawing.text((x+4, y+273), sid+' '+title[:27], font=font, fill='black')
        destination = work/f'contact_{first//12+1:02d}.png'
        canvas.save(destination)
        sheets.append(str(destination))
    print(json.dumps(dict(checks=str(publication/'encoded_checks.json'), contact_sheets=sheets,
        dimensions_ok=checks['all_dimensions_960_540'], size_ok=checks['all_files_under_50MB'],
        speed_range=[min(ratios), max(ratios)], scene_count=len(previews)), ensure_ascii=False))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication', type=Path)
    parser.add_argument('--work-dir', type=Path, default=Path('/tmp/synthetic-explainer-20261113/encoded_review'))
    args = parser.parse_args()
    verify(args.publication, args.work_dir)
