#!/usr/bin/env python3
"""Japanese neural narration at exactly twice the normal PCM timeline.

The service synthesizes ordinary-rate speech. FFmpeg's atempo=2.0 speeds
that recording without raising its pitch. A declared silence guard protects
the final phoneme; the resulting PCM is padded/trimmed only at the tail to
exactly half the normal guarded PCM frame count. Raw/cached audio belongs
in the caller's temporary work directory, not the scientific publication.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import wave


VOICE = 'ja-JP-NanamiNeural'
NORMAL_RATE = '+0%'
SPEED = 2.0
SAMPLE_RATE = 24000
TAIL_GUARD_SECONDS = .2


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _story_scenes(story):
    if 'parts' in story:
        scenes = []
        for part in story['parts']:
            for scene in part['scenes']:
                scenes.append(dict(scene, part=part['id'], part_title=part['title']))
        return scenes
    return list(story['scenes'])


def spoken_text(text):
    """Make literal units readable; keep the original narration separately."""
    # A middle dot is punctuation to a synthesizer, not an unambiguous
    # decimal separator. Retain the complete integer token (including 十,
    # 百, ...), make 点 explicit, and use 零 for every fractional zero.
    # Non-numeric middle dots, such as ルンゲ・クッタ, stay untouched.
    text = re.sub(r'([〇零一二三四五六七八九十百千万]+)・([〇零一二三四五六七八九]+)',
                  lambda match: match[1].replace('〇', '零')+'点'+match[2].replace('〇', '零'), text)
    replacements = (
        ('deg/s', '度毎秒'), ('rad/s', 'ラジアン毎秒'),
        ('RK4', 'アールケー・フォー'), ('RPY', 'ロール、ピッチ、ヨー'),
        ('NPZ', 'エヌピーゼット'), ('CSV', 'シーエスブイ'),
        ('Haar', 'ハール'), ('Bingham', 'ビンガム'), ('Euler', 'オイラー'),
        ('Gaussian', 'ガウス'), ('Monte Carlo', 'モンテカルロ'),
    )
    for before, after in replacements:
        text = text.replace(before, after)
    text = re.sub(r'(?<=\d)\s*Hz\b', 'ヘルツ', text)
    return text


def _wav_info(path):
    with wave.open(str(path), 'rb') as stream:
        return dict(frames=stream.getnframes(), rate=stream.getframerate(),
                    channels=stream.getnchannels(), sample_width=stream.getsampwidth())


def _guard_and_double(raw_mp3, baseline, guarded, natural, doubled):
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(raw_mp3), '-ar', str(SAMPLE_RATE), '-ac', '1',
                    '-c:a', 'pcm_s16le', str(baseline)], check=True)
    with wave.open(str(baseline), 'rb') as source:
        params = source.getparams()
        pcm = source.readframes(source.getnframes())
    voice_frames = len(pcm)//(params.nchannels*params.sampwidth)
    guard_frames = round(TAIL_GUARD_SECONDS*params.framerate)
    with wave.open(str(guarded), 'wb') as target:
        target.setparams(params)
        target.writeframes(pcm+b'\0'*(guard_frames*params.nchannels*params.sampwidth))
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(guarded), '-af', 'atempo=2.0', '-ar', str(SAMPLE_RATE),
                    '-ac', '1', '-c:a', 'pcm_s16le', str(natural)], check=True)
    with wave.open(str(natural), 'rb') as source:
        fast_params = source.getparams()
        accelerated = source.readframes(source.getnframes())
    frame_bytes = fast_params.nchannels*fast_params.sampwidth
    natural_frames = len(accelerated)//frame_bytes
    normal_frames = voice_frames+guard_frames
    target_frames = (normal_frames+1)//2
    removed = accelerated[target_frames*frame_bytes:]
    # The guard is long enough to absorb atempo's finite-window tail offset.
    # Record the actual removed tail rather than asserting speech quality
    # from the duration alone.
    import array
    removed_samples = array.array('h', removed)
    removed_peak = max((abs(x) for x in removed_samples), default=0)
    adjusted = accelerated[:target_frames*frame_bytes]
    adjusted += b'\0'*max(0, target_frames*frame_bytes-len(adjusted))
    with wave.open(str(doubled), 'wb') as target:
        target.setparams(fast_params)
        target.writeframes(adjusted)
    all_samples = array.array('h', adjusted)
    peak = max((abs(x) for x in all_samples), default=0)
    rms = (sum(x*x for x in all_samples)/len(all_samples))**.5 if all_samples else 0.
    return dict(normal_voice_frames=voice_frames, normal_guard_frames=guard_frames,
                normal_frames=normal_frames, natural_atempo_frames=natural_frames,
                final_frames=target_frames, sample_rate_hz=SAMPLE_RATE,
                normal_voice_duration_seconds=voice_frames/SAMPLE_RATE,
                normal_duration_seconds=normal_frames/SAMPLE_RATE,
                duration_seconds=target_frames/SAMPLE_RATE,
                measured_normal_to_final_ratio=normal_frames/target_frames,
                tail_frames_trimmed=max(0, natural_frames-target_frames),
                tail_frames_padded=max(0, target_frames-natural_frames),
                removed_tail_peak_pcm16=removed_peak,
                final_peak_pcm16=peak, final_rms_pcm16=rms,
                normal_tail_guard_seconds=guard_frames/SAMPLE_RATE)


def _converted_boundaries(boundaries):
    converted = []
    for event in boundaries:
        item = dict(event)
        if 'offset' in event:
            item['normal_offset_seconds'] = float(event['offset'])*1e-7
            item['offset_seconds'] = float(event['offset'])*1e-7/SPEED
        if 'duration' in event:
            item['normal_duration_seconds'] = float(event['duration'])*1e-7
            item['duration_seconds'] = float(event['duration'])*1e-7/SPEED
        converted.append(item)
    return converted


async def _synthesize_scene(scene, semaphore, work_dir, voice, retries):
    import edge_tts
    scene_id = str(scene['id'])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', scene_id):
        raise ValueError('Scene ID must be a plain filename component: '+scene_id)
    original = scene['narration']
    spoken = spoken_text(original)
    specification = dict(narration=original, spoken_text=spoken, voice=voice,
                         normal_rate=NORMAL_RATE, speed_multiplier=SPEED,
                         sample_rate_hz=SAMPLE_RATE, tail_guard_seconds=TAIL_GUARD_SECONDS,
                         engine='edge_tts', engine_version=edge_tts.__version__)
    fingerprint = hashlib.sha256(json.dumps(specification, ensure_ascii=False,
                                sort_keys=True).encode()).hexdigest()
    raw_directory = work_dir/'voice_raw'
    fast_directory = work_dir/'voice_2x'
    raw_mp3 = raw_directory/(scene_id+'.mp3')
    baseline = raw_directory/(scene_id+'_normal.wav')
    guarded = raw_directory/(scene_id+'_guarded.wav')
    natural = raw_directory/(scene_id+'_atempo.wav')
    doubled = fast_directory/(scene_id+'.wav')
    record_path = fast_directory/(scene_id+'.json')
    if record_path.exists() and doubled.exists():
        cached = json.loads(record_path.read_text())
        if cached.get('fingerprint') == fingerprint and cached.get('audio_sha256') == _sha(doubled):
            # Scene titles can change during visual review without changing
            # spoken content. Keep the provenance current without resynthesis.
            cached.update(title=scene['title'], part=scene.get('part'),
                          part_title=scene.get('part_title'))
            _write_json(record_path, cached)
            return cached
    async with semaphore:
        boundaries = []
        started = time.monotonic()
        attempts = 0
        for attempt in range(retries):
            attempts = attempt+1
            try:
                audio_chunks, boundaries = [], []
                communicate = edge_tts.Communicate(spoken, voice=voice, rate=NORMAL_RATE,
                                                  boundary='SentenceBoundary')
                async for event in communicate.stream():
                    if event['type'] == 'audio': audio_chunks.append(event['data'])
                    else: boundaries.append(event)
                if not audio_chunks:
                    raise RuntimeError('TTS stream returned no audio')
                temporary = raw_mp3.with_suffix('.mp3.part')
                temporary.write_bytes(b''.join(audio_chunks))
                temporary.replace(raw_mp3)
                break
            except Exception:
                if attempt+1 == retries: raise
                await asyncio.sleep(min(15., 2.**attempt))
        timing = _guard_and_double(raw_mp3, baseline, guarded, natural, doubled)
        record = dict(id=scene_id, part=scene.get('part'), part_title=scene.get('part_title'),
                      title=scene['title'], fingerprint=fingerprint, **specification,
                      text=original, speedup=SPEED,
                      audio_path=str(doubled.resolve()), raw_mp3_path=str(raw_mp3.resolve()),
                      audio_sha256=_sha(doubled), raw_mp3_sha256=_sha(raw_mp3),
                      boundaries=_converted_boundaries(boundaries), timing=timing,
                      duration=timing['duration_seconds'], duration_seconds=timing['duration_seconds'],
                      normal_duration_seconds=timing['normal_duration_seconds'],
                      baseline_pcm_seconds=timing['normal_duration_seconds'],
                      baseline_voice_pcm_seconds=timing['normal_voice_duration_seconds'],
                      attempts=attempts, wall_seconds=time.monotonic()-started,
                      speed_method='normal Japanese neural speech, pitch-preserving ffmpeg atempo=2.0; protected silence tail adjusted to exact half guarded PCM frame count')
        _write_json(record_path, record)
        print(scene_id, f"{record['duration_seconds']:.3f}s at 2x", flush=True)
        return record


async def _synthesize_story(story_path, work_dir, *, voice=VOICE, concurrency=3, retries=4):
    import edge_tts
    story_path, work_dir = Path(story_path).resolve(), Path(work_dir).resolve()
    for directory in (work_dir, work_dir/'voice_raw', work_dir/'voice_2x'):
        directory.mkdir(parents=True, exist_ok=True)
    story = json.loads(story_path.read_text())
    scenes = _story_scenes(story)
    ids = [str(scene['id']) for scene in scenes]
    if len(ids) != len(set(ids)):
        raise ValueError('Narration scene IDs must be unique')
    semaphore = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*[_synthesize_scene(scene, semaphore, work_dir, voice, retries)
                                    for scene in scenes], return_exceptions=True)
    succeeded = [result for result in results if not isinstance(result, BaseException)]
    failures = [dict(id=ids[i], error=type(result).__name__+': '+str(result))
                for i, result in enumerate(results) if isinstance(result, BaseException)]
    manifest = dict(schema='grape/synthetic-explainer-japanese-narration/v1',
        story_path=str(story_path), story_sha256=_sha(story_path), voice=voice,
        engine='edge_tts', engine_version=edge_tts.__version__,
        normal_rate=NORMAL_RATE, speed_multiplier=SPEED, sample_rate_hz=SAMPLE_RATE,
        scenes=succeeded, failures=failures,
        scene_count=len(scenes), completed_scene_count=len(succeeded),
        total_normal_duration_seconds=sum(r['normal_duration_seconds'] for r in succeeded),
        total_duration_seconds=sum(r['duration_seconds'] for r in succeeded),
        raw_audio_is_temporary_cache=True)
    manifest_path = work_dir/'narration_manifest.json'
    _write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(f'{len(failures)} narration scenes failed; details: {manifest_path}')
    return manifest


def synthesize_story(story_path, work_dir, *, voice=VOICE, concurrency=3, retries=4):
    """Synthesize nested part/scenes, cache raw audio, and return the manifest."""
    return asyncio.run(_synthesize_story(story_path, work_dir, voice=voice,
                                       concurrency=concurrency, retries=retries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--story', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--voice', default=VOICE)
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--retries', type=int, default=4)
    args = parser.parse_args()
    manifest = synthesize_story(args.story, args.work_dir, voice=args.voice,
                                concurrency=args.concurrency, retries=args.retries)
    print(json.dumps(dict(scenes=manifest['completed_scene_count'],
                          normal_seconds=manifest['total_normal_duration_seconds'],
                          twice_speed_seconds=manifest['total_duration_seconds']), ensure_ascii=False))
