#!/usr/bin/env python3
"""Build narrated, data-backed Japanese chapters without changing inference.

Normal-speed TTS and raw intermediate clips stay in a separate work folder.
Only final chapter MP4s, subtitles, source/timing records and small numerical
explanation assets are exported. All subprocess arguments are structured.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import wave

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'publication/three_frequency_0p5hz_sigma5'
WIDTH, HEIGHT, FPS = 960, 540, 15
BYTE_LIMIT = 50_000_000


def run_command(arguments, **kwargs):
    return subprocess.run([str(a) for a in arguments], check=True, **kwargs)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                     '-show_format', '-of', 'json', str(path)], text=True))


def story_scenes(story):
    scenes = []
    for part in story['parts']:
        for original in part['scenes']:
            scene = dict(original)
            scene['part'] = part['id']
            scene['part_title'] = part['title']
            scenes.append(scene)
    identifiers = [str(s['id']) for s in scenes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Scene IDs must be unique')
    if not all(s['narration'].strip() for s in scenes):
        raise ValueError('Every explanatory scene must have narration')
    return scenes


def render_job(job):
    scene, duration, destination, data_dir, fps = job
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/synthetic-explainer-mpl')
    from render_scenes import render_scene
    render_scene(scene, duration, destination, data_dir=data_dir, fps=fps)
    return dict(id=str(scene['id']), path=str(destination), duration=duration)


def cache_key(scene, duration, data_dir):
    source_paths = sorted(p for p in Path(data_dir).iterdir()
                          if p.suffix in ('.npz', '.json'))
    derived = scene.get('data_view', {}).get('derived_data')
    value = dict(scene=scene, duration=duration, renderer=sha(HERE/'render_scenes.py'),
                 source={p.name:sha(p) for p in source_paths},
                 derived_sha256=sha(derived) if derived else None)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def chapter_pcm(rows, destination):
    """Join PCM first, so AAC priming occurs only once per final chapter."""
    parameters = None
    with wave.open(str(destination), 'wb') as output:
        for _, audio, duration in rows:
            with wave.open(str(audio['audio_path']), 'rb') as source:
                current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                if parameters is None:
                    parameters = current
                    output.setnchannels(current[0]); output.setsampwidth(current[1]); output.setframerate(current[2])
                if current != parameters:
                    raise ValueError('Narration PCM parameters differ between scenes')
                frames = source.getnframes()
                target_frames = round(duration*current[2])
                if frames > target_frames:
                    raise ValueError('A visual scene is shorter than its uncut narration')
                output.writeframes(source.readframes(frames))
                output.writeframes(b'\0'*(target_frames-frames)*current[0]*current[1])


def srt_time(seconds):
    milliseconds = int(round(max(0., seconds)*1000))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    sec, ms = divmod(rest, 1000)
    return f'{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}'


def subtitles(rows, destination):
    entries = []
    offset = 0.
    for scene, audio, duration in rows:
        boundaries = [b for b in audio.get('boundaries', []) if b.get('text')]
        if boundaries:
            for item in boundaries:
                begin = float(item.get('offset_seconds', 0.))
                length = float(item.get('duration_seconds', 0.))
                entries.append((offset+begin, offset+min(duration, begin+length), item['text']))
        else:
            entries.append((offset, offset+float(audio['duration_seconds']), scene['narration']))
        offset += duration
    lines = []
    for index, (begin, end, text) in enumerate(entries, 1):
        lines.extend((str(index), f'{srt_time(begin)} --> {srt_time(end)}', text, ''))
    destination.write_text('\n'.join(lines))


def concat_chapter(paths, rows, destination, srt):
    # Paths are generated under our own work directory. ffmpeg concat syntax
    # still needs single-quote escaping, independently of shell escaping.
    listing = destination.with_suffix('.concat.txt')
    listing.write_text('\n'.join("file '"+str(p.resolve()).replace("'", "'\\''")+"'" for p in paths)+'\n')
    pcm = destination.with_suffix('.wav')
    chapter_pcm(rows, pcm)
    metadata_path = destination.with_suffix('.ffmetadata')
    metadata = [';FFMETADATA1']
    groups = []
    for scene, _, duration in rows:
        if not groups or groups[-1]['part'] != scene['part']:
            groups.append(dict(part=scene['part'], title=scene['part_title'], duration=0.))
        groups[-1]['duration'] += duration
    offset = 0.
    for group in groups:
        title = group['title'].replace('\\', '\\\\').replace('=', '\\=').replace(';', '\\;').replace('#', '\\#').replace('\n', ' ')
        metadata.extend(('[CHAPTER]', 'TIMEBASE=1/1000', f'START={round(offset*1000)}',
                         f'END={round((offset+group["duration"])*1000)}', 'title='+title))
        offset += group['duration']
    metadata_path.write_text('\n'.join(metadata)+'\n')
    run_command(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', listing,
        '-i', pcm, '-i', srt, '-f', 'ffmetadata', '-i', metadata_path,
        '-map', '0:v:0', '-map', '1:a:0', '-map', '2:s:0', '-map_chapters', '3', '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '96k',
        '-c:s', 'mov_text', '-metadata:s:a:0', 'language=jpn', '-metadata:s:s:0', 'language=jpn',
        '-disposition:s:0', '0', '-movflags', '+faststart', destination])


def export_chapter(paths, rows, base, output, work, result):
    srt = work/(base+'.srt')
    subtitles(rows, srt)
    candidate = work/(base+'.mp4')
    concat_chapter(paths, rows, candidate, srt)
    if candidate.stat().st_size >= BYTE_LIMIT:
        if len(paths) > 1:
            middle = len(paths)//2
            export_chapter(paths[:middle], rows[:middle], base+'_a', output, work, result)
            export_chapter(paths[middle:], rows[middle:], base+'_b', output, work, result)
            return
        # A single scene is never discarded. Re-encode it at a bitrate chosen
        # from the user's explicit file-size limit, preserving dimensions.
        duration = float(probe(candidate)['format']['duration'])
        video_bitrate = max(100_000, int(8*45_000_000/duration)-96_000)
        reduced = work/(base+'_bounded.mp4')
        run_command(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', candidate,
            '-map', '0', '-c:v', 'libx264', '-preset', 'medium', '-b:v', str(video_bitrate),
            '-maxrate', str(video_bitrate), '-bufsize', str(2*video_bitrate),
            '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-c:s', 'copy', '-movflags', '+faststart', reduced])
        candidate = reduced
    if candidate.stat().st_size >= BYTE_LIMIT:
        raise RuntimeError('The final encoded scene still exceeds the requested 50 MB limit')
    target = output/(base+'.mp4')
    shutil.copy2(candidate, target)
    shutil.copy2(srt, output/(base+'.srt'))
    info = probe(target)
    video = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
    audio = next(stream for stream in info['streams'] if stream['codec_type'] == 'audio')
    if (video['width'], video['height']) != (WIDTH, HEIGHT):
        raise RuntimeError('Encoded dimensions differ from the requested 960x540')
    result.append(dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target),
        title=rows[0][0].get('part_title', base),
        duration_seconds=float(info['format']['duration']), video=video, audio=audio,
        scene_ids=[str(s['id']) for s, _, _ in rows],
        subtitle_file=base+'.srt',
        source='concatenated exact per-scene render and 2x narration; optional soft Japanese subtitles'))


def build(args):
    story = json.loads(args.story.read_text())
    scenes = story_scenes(story)
    work, output = args.work_dir.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name:sha(p) for p in args.data_dir.glob('*.npz')}
    from prepare_assets import prepare
    derived = work/'derived_data.npz'
    asset_record = prepare(args.data_dir, derived)
    write_json(work/'derived_data.json', asset_record)
    if args.narration_manifest:
        voice_manifest_path = args.narration_manifest
    else:
        voice_manifest_path = work/'narration_manifest.json'
        run_command([args.tts_python, HERE/'synthesize_voice.py', '--story', args.story,
                     '--work-dir', work])
    voices = json.loads(voice_manifest_path.read_text())
    voice_by_id = {str(row['id']):row for row in voices['scenes']}
    (work/'silent').mkdir(exist_ok=True)
    jobs, active, duration_by_id = [], {}, {}
    for scene in scenes:
        sid = str(scene['id'])
        row = voice_by_id[sid]
        # Speech itself remains exactly 2x. A short separate visual tail lets
        # the listener see the final animation frame before the next scene.
        duration = math.ceil(max(float(row['duration_seconds'])+.5, 7.)*args.fps)/args.fps
        duration_by_id[sid] = duration
        scene.setdefault('data_view', {})['derived_data'] = str(derived)
        destination = work/'silent'/(sid+'.mp4')
        cache_path = work/'silent'/(sid+'.cache.json')
        key = cache_key(scene, duration, args.data_dir)
        if (destination.exists() and cache_path.exists()
                and json.loads(cache_path.read_text()).get('key') == key):
            active[sid] = destination
        else:
            jobs.append((scene, duration, destination, args.data_dir.resolve(), args.fps))
    print(f'Render {len(jobs)} scenes; cached {len(active)}; workers {args.workers}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(render_job, job):job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            result = future.result()
            sid = result['id']
            destination = Path(result['path'])
            write_json(destination.with_suffix('.cache.json'), dict(key=cache_key(job[0], job[1], args.data_dir)))
            active[sid] = destination
            print('Rendered', sid, f'{result["duration"]:.2f}s', flush=True)
    chapters, timing, all_paths, all_rows, part_timing = [], [], [], [], []
    timeline_offset = 0.
    for index, part in enumerate(story['parts'], 1):
        part_scenes = [s for s in scenes if s['part'] == part['id']]
        paths, rows = [], []
        for scene in part_scenes:
            sid = str(scene['id'])
            audio = voice_by_id[sid]
            duration = duration_by_id[sid]
            paths.append(active[sid])
            rows.append((scene, audio, duration))
            timing.append(dict(id=sid, part=part['id'], duration_seconds=duration,
                               speech_seconds=audio['duration_seconds'], title=scene['title']))
        part_duration = sum(row[2] for row in rows)
        part_timing.append(dict(part=part['id'], title=part['title'], start_seconds=timeline_offset,
                                duration_seconds=part_duration))
        timeline_offset += part_duration
        all_paths.extend(paths); all_rows.extend(rows)
    # Prefer one complete movie. Split at scene boundaries only if its
    # actual encoded size exceeds the user's 50 MB per-file limit.
    export_chapter(all_paths, all_rows, 'synthetic_complete_ja_2x', output, work, chapters)
    for chapter in chapters:
        chapter['title'] = '合成回転の完全解説' if len(chapters) == 1 else '合成回転の完全解説（分割）'
    after = {p.name:sha(p) for p in args.data_dir.glob('*.npz')}
    write_json(output/'narration_manifest.json', voices)
    write_json(output/'scene_timing.json', timing)
    write_json(output/'chapter_timing.json', part_timing)
    write_json(output/'derived_data.json', asset_record)
    shutil.copy2(derived, output/'derived_data.npz')
    shutil.copy2(args.story, output/'story.json')
    narration_md = HERE/'narration.md'
    if narration_md.exists(): shutil.copy2(narration_md, output/'narration.md')
    for scene in scenes:
        sidecar = active[str(scene['id'])].with_suffix('.json')
        if sidecar.exists():
            (output/'scene_sources').mkdir(exist_ok=True)
            shutil.copy2(sidecar, output/'scene_sources'/sidecar.name)
    verification = dict(schema='grape/synthetic-narrated-explainer/v1', width=WIDTH, height=HEIGHT,
        fps=args.fps, voice_speed=2., audio_language='ja-JP', byte_limit_per_file=BYTE_LIMIT,
        chapters=chapters, logical_parts=part_timing,
        all_files_below_limit=all(c['bytes'] < BYTE_LIMIT for c in chapters),
        original_numeric_sources_sha256=source_hashes,
        original_numeric_sources_unchanged=source_hashes == after,
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        story_sha256=sha(args.story),
        intermediate_frames_audio_and_clips_published=False,
        subtitle_timing='TTS sentence boundary offsets divided by two, shifted by actual encoded scene durations; fallback only when boundaries unavailable')
    write_json(output/'verification.json', verification)
    lines = ['# 合成回転：日本語・2倍速の詳細解説', '',
             '960×540 px。元の3周波数・0.5 Hz・仮定sigma=5 deg/s。音声は日本語、通常速度から音程を保って2倍速に処理。', '',
             '| 章 | 動画 | 長さ | 容量 | 字幕 |', '|---|---|---:|---:|---|']
    for chapter in chapters:
        lines.append(f'| {chapter["title"]} | [{chapter["file"]}]({chapter["file"]}) | {chapter["duration_seconds"]:.1f} s | '
                     f'{chapter["bytes"]/1_000_000:.2f} MB | [SRT]({chapter["subtitle_file"]}) |')
    lines.extend(['', '動画内の章（MP4のチャプター情報にも保存）：', '', '| 開始 | 内容 |', '|---|---|'])
    for part in part_timing:
        seconds = int(part['start_seconds'])
        lines.append(f'| {seconds//60:02d}:{seconds%60:02d} | {part["title"]} |')
    lines.extend(['', '[全文台本](narration.md) / [場面と原理](story.json) / [検証](verification.json)', '',
        '図は保存済みの同じ候補・観測・真値・入力から作成。RK4の最初の1ステップだけを同じ式で再計算した説明用の数値トレースも保存し、元の保存軌道との一致を検証している。', ''])
    (output/'INDEX.md').write_text('\n'.join(lines))
    manifest = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    write_json(output/'manifest.json', dict(files=manifest, bytes_excluding_manifest=sum(p['bytes'] for p in manifest)))
    print('Completed', output, 'videos', len(chapters), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--story', type=Path, default=HERE/'story.json')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--work-dir', type=Path, default=Path('/tmp/synthetic-explainer-20261113'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/synthetic-explainer-20261113/final'))
    parser.add_argument('--tts-python', type=Path, default=Path('/tmp/evian-narration-venv/bin/python'))
    parser.add_argument('--narration-manifest', type=Path)
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--workers', type=int, default=4)
    build(parser.parse_args())
