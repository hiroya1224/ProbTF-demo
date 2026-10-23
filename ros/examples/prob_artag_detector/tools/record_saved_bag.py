#!/usr/bin/env python3
"""Record saved real-camera AR bag payloads in an isolated, unchanged RViz view."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import shutil
import signal
import socket
import subprocess
import time
import xmlrpc.client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--display', default=':99')
    parser.add_argument('--master-port', type=int, default=11993)
    parser.add_argument('--xvfb', required=True)
    parser.add_argument('--monitor-script', type=Path,
                        default=Path(__file__).with_name('monitor_replay_payloads.py'))
    args = parser.parse_args()
    args.run_folder = args.run_folder.resolve()
    args.scratch = args.scratch.resolve()
    args.output = args.output.resolve()
    args.scratch.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    display_number = args.display.lstrip(':').split('.')[0]
    if Path('/tmp/.X11-unix/X' + display_number).exists():
        raise RuntimeError('Display already in use: ' + args.display)
    with socket.socket() as master_probe:
        master_probe.bind(('127.0.0.1', args.master_port))
    bag = args.run_folder / 'all_topics_30s.bag'
    source_config = args.run_folder / 'demo.rviz'
    config = args.scratch / 'demo.rviz'
    if not bag.is_file() or not source_config.is_file():
        raise RuntimeError('Missing saved bag or its saved demo.rviz')
    shutil.copyfile(source_config, config)
    config_hash = hashlib.sha256(source_config.read_bytes()).hexdigest()
    assert hashlib.sha256(config.read_bytes()).hexdigest() == config_hash
    env = os.environ.copy()
    env.update({
        'DISPLAY': args.display,
        'ROS_MASTER_URI': 'http://127.0.0.1:%d' % args.master_port,
        'ROS_IP': '127.0.0.1',
        'ROS_HOSTNAME': '127.0.0.1',
        'ROS_HOME': str(args.scratch / 'ros_home'),
        'ROS_LOG_DIR': str(args.scratch / 'ros_logs'),
        'LIBGL_ALWAYS_SOFTWARE': '1',
        'QT_QPA_PLATFORM': 'xcb',
        'PYTHONDONTWRITEBYTECODE': '1',
    })
    Path(env['ROS_HOME']).mkdir(exist_ok=True)
    Path(env['ROS_LOG_DIR']).mkdir(exist_ok=True)
    topics = ['/tf_static', '/tf', '/prob_artag_demo/probtf',
              '/prob_artag_detector/debug_image']
    processes = []
    logs = []
    events = []
    master_fd = slave_fd = None
    ffmpeg = monitor = None
    manifest = {
        'method': 'Original ROS bag payload replay into original saved RViz configuration',
        'bag': str(bag), 'bag_size_bytes': bag.stat().st_size,
        'rviz_config_source': str(source_config),
        'rviz_config_copy': str(config), 'rviz_config_sha256': config_hash,
        'output': str(args.output), 'playback_rate': 1.0,
        'recording_fps': 30, 'monitored_topics': topics,
        'played_topics': 'all original bag topics, preserving the complete bag time range',
        'display': args.display, 'ros_master_uri': env['ROS_MASTER_URI'],
        'original_payload_reestimated': False,
        'screenshots_taken': False, 'events': events, 'failure': None,
    }

    def event(name, **extra):
        row = dict(event=name, wall_time=time.time(), monotonic=time.monotonic(), **extra)
        events.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def launch(name, command, stdin=None):
        log = (args.scratch / (name + '.log')).open('wb')
        logs.append(log)
        proc = subprocess.Popen(command, stdin=stdin, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        processes.append((name, proc))
        event(name + '_started', pid=proc.pid, command=command)
        return proc

    def wait_until(description, predicate, timeout=30, proc=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError('%s exited early (%s)' % (description, proc.returncode))
            if predicate():
                return
            time.sleep(0.05)
        raise RuntimeError('Timed out waiting for ' + description)

    def stop(proc, sig=signal.SIGTERM, timeout=8):
        if proc is None or proc.poll() is not None:
            return
        os.killpg(proc.pid, sig)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)

    try:
        xvfb = launch('xvfb', [args.xvfb, args.display, '-screen', '0', '2400x1600x24',
                              '-nolisten', 'tcp', '-ac', '-noreset', '+extension', 'GLX'])

        def x_ready():
            return subprocess.run(['xwininfo', '-root'], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL).returncode == 0
        wait_until('virtual X display', x_ready, proc=xvfb)
        roscore = launch('roscore', ['roscore', '-p', str(args.master_port)])

        def ros_ready():
            try:
                return xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/ar_capture')[0] == 1
            except (OSError, xmlrpc.client.Error):
                return False
        wait_until('dedicated ROS master', ros_ready, proc=roscore)
        subprocess.run(['rosparam', 'set', '/use_sim_time', 'true'], env=env, check=True)
        monitor = launch('payload_monitor', [
            '/usr/bin/python3', str(args.monitor_script),
            '--output', str(args.scratch / 'payload_monitor.json'),
            '--ready', str(args.scratch / 'monitor_ready.json')])
        wait_until('payload monitor', lambda: (args.scratch / 'monitor_ready.json').exists(),
                   proc=monitor)
        rviz = launch('rviz', ['rviz', '-d', str(config), '__name:=ar_saved_bag_rviz'])
        window = {}

        def rviz_ready():
            tree = subprocess.check_output(['xwininfo', '-root', '-tree'], env=env, text=True)
            for line in tree.splitlines():
                if 'rviz' not in line.lower():
                    continue
                match = re.match(r'\s*(0x[0-9a-f]+).*?\s(\d+)x(\d+)[+-]\d+[+-]\d+', line)
                if not match or int(match[2]) < 1000 or int(match[3]) < 800:
                    continue
                details = subprocess.check_output(['xwininfo', '-id', match[1]], env=env, text=True)
                values = {}
                for key in ['Absolute upper-left X', 'Absolute upper-left Y', 'Width', 'Height']:
                    result = re.search(r'^\s*' + re.escape(key) + r':\s*(-?\d+)', details, re.MULTILINE)
                    if not result:
                        return False
                    values[key] = int(result[1])
                if 'Map State: IsViewable' not in details:
                    return False
                window.update(id=match[1], x=values['Absolute upper-left X'],
                              y=values['Absolute upper-left Y'],
                              width=values['Width'], height=values['Height'])
                (args.scratch / 'rviz_window_geometry.txt').write_text(details)
                return True
            return False
        wait_until('mapped RViz window', rviz_ready, proc=rviz)
        event('rviz_window_ready', **window)
        manifest['window_geometry'] = dict(window)
        if window['width'] != 2200 or window['height'] != 1400:
            raise RuntimeError('RViz geometry differs from saved 2200x1400 configuration: %r' % window)
        # Give the RViz displays time to register before the paused player advertises.
        time.sleep(1.0)
        master_fd, slave_fd = pty.openpty()
        # Play the complete bag so its original begin/end clock is preserved.
        # --wait-for-subscribers would incorrectly wait on unused auxiliary topics.
        player_cmd = ['rosbag', 'play', '--clock', '--pause',
                      '--rate', '1.0', str(bag)]
        player = launch('rosbag_play', player_cmd, stdin=slave_fd)
        os.close(slave_fd)
        slave_fd = None

        def player_paused():
            return 'PAUSED' in (args.scratch / 'rosbag_play.log').read_text(errors='replace')
        wait_until('paused rosbag player and subscribers', player_paused, timeout=40, proc=player)
        monitor_name = json.loads((args.scratch / 'monitor_ready.json').read_text())['node']

        def required_connections_ready():
            master = xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
            for node in [monitor_name, '/ar_saved_bag_rviz']:
                lookup = master.lookupNode('/ar_capture', node)
                if lookup[0] != 1:
                    return False
                result = xmlrpc.client.ServerProxy(lookup[2]).getBusInfo('/ar_capture')
                if result[0] != 1:
                    return False
                incoming = {row[4] for row in result[2]
                            if len(row) >= 6 and row[2] == 'i' and row[5]}
                if not set(topics).issubset(incoming):
                    return False
            return True
        wait_until('four recorded-topic connections to RViz and payload monitor',
                   required_connections_ready, timeout=30, proc=player)
        event('rosbag_player_paused_ready')
        progress = args.scratch / 'video_progress.txt'
        capture_input = '%s+%d,%d' % (args.display, window['x'], window['y'])
        ffmpeg = launch('ffmpeg', [
            'ffmpeg', '-hide_banner', '-loglevel', 'info', '-y',
            '-thread_queue_size', '1024', '-f', 'x11grab', '-draw_mouse', '0',
            '-framerate', '30', '-video_size', '%dx%d' % (window['width'], window['height']),
            '-i', capture_input, '-an', '-c:v', 'libx264', '-threads', '4',
            '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-progress', str(progress), str(args.output)],
            stdin=subprocess.PIPE)

        def frame_ready():
            if not progress.exists():
                return False
            return any(int(number) >= 1 for number in re.findall(r'^frame=(\d+)\s*$',
                       progress.read_text(errors='replace'), re.MULTILINE))
        wait_until('first encoded video frame', frame_ready, timeout=30, proc=ffmpeg)
        event('video_first_frame_confirmed')
        os.write(master_fd, b' ')
        event('rosbag_pause_released')
        player.wait(timeout=65)
        event('rosbag_play_finished', returncode=player.returncode)
        if player.returncode != 0:
            raise RuntimeError('rosbag playback failed: %d' % player.returncode)
        time.sleep(0.5)
        ffmpeg.stdin.write(b'q\n')
        ffmpeg.stdin.flush()
        ffmpeg.wait(timeout=30)
        event('video_recording_finished', returncode=ffmpeg.returncode)
        if ffmpeg.returncode != 0:
            raise RuntimeError('ffmpeg failed: %d' % ffmpeg.returncode)
        stop(monitor, signal.SIGTERM)
        event('payload_monitor_finished', returncode=monitor.returncode)
        if not (args.scratch / 'payload_monitor.json').exists():
            raise RuntimeError('Payload monitor did not save results')
        probe = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration,size:stream=codec_name,width,height,r_frame_rate,nb_frames',
            '-of', 'json', str(args.output)], text=True)
        manifest['video_probe'] = json.loads(probe)
        manifest['output_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        assert source_config.read_bytes() == config.read_bytes()
        manifest['rviz_config_unchanged'] = True
        manifest['process_returncodes'] = {name: proc.poll() for name, proc in processes}
    except BaseException as error:
        manifest['failure'] = repr(error)
        event('recording_failed', error=repr(error))
        raise
    finally:
        # Only the process groups launched by this invocation are signalled.
        for name, proc in reversed(processes):
            stop(proc, signal.SIGTERM)
        for fd in [master_fd, slave_fd]:
            if fd is not None:
                os.close(fd)
        for log in logs:
            log.close()
        manifest['final_process_returncodes'] = {name: proc.poll() for name, proc in processes}
        (args.scratch / 'capture_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
