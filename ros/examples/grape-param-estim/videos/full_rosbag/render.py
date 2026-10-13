#!/usr/bin/env python3
"""Extract complete ROS1 bags, then render timed 3D flight videos offline.

Extraction and rendering are separate commands so ROS Noetic's Python and a
modern headless VTK/PyAV environment can be used without changing either one.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone, timedelta
from fractions import Fraction


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
NS = 1_000_000_000
FPS = 120
WIDTH, HEIGHT = 960, 540
CAMERA_POSITION = (2.3, -3.8, 2.6)
CAMERA_FOCAL_POINT = (0, -.25, .60)
CAMERA_SCALE = .85
CASES = {
    "failure1": "single_rosbag_1.json",
    "failure2": "single_rosbag_2.json",
    "success": "single_rosbag_succeeded.json",
}
MOCAP = "/gimbalrotor/mocap/pose"
ODOM = "/gimbalrotor/uav/baselink/odom"
STATE = "/gimbalrotor/flight_state"
STATE_NAMES = {
    0: "ARM OFF", 1: "START", 2: "ARM ON", 3: "TAKEOFF",
    4: "LAND", 5: "HOVER", 6: "STOP", 17: "FORCE LAND",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_schedule(duration_ns, fps=FPS):
    """Regular PTS rounded to ns, with a shortened last sample, no time stretch."""
    if duration_ns <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    count = (duration_ns * fps + NS - 1) // NS
    pts = [(i * NS + fps // 2) // fps for i in range(count)]
    ends = pts[1:] + [duration_ns]
    durations = [end - start for start, end in zip(pts, ends)]
    if min(durations) <= 0:
        raise ValueError("frame schedule exceeds timestamp precision")
    return pts, durations


def timestamp(ns):
    seconds, fraction = divmod(int(ns), NS)
    date = datetime.fromtimestamp(seconds, timezone(timedelta(hours=9)))
    return date.strftime("%Y-%m-%d %H:%M:%S") + ".{:09d} JST".format(fraction)


def extract(case, cache, bag_root=None):
    import numpy as np
    import rosbag

    config_path = PROJECT / "minimal" / "bag_jsons" / CASES[case]
    config = json.loads(config_path.read_text())
    bag_path = Path(config["bag_path"])
    if bag_root is not None:
        matches = list(bag_root.rglob(bag_path.name))
        if len(matches) != 1:
            raise ValueError("--bag-root must contain exactly one " + bag_path.name)
        bag_path = matches[0]
    streams = {}
    static = {}
    states = []
    counts = {}
    topic_extents = {}

    def pose_row(stamp, record, pose):
        p, q = pose.position, pose.orientation
        return (stamp, record, p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    with rosbag.Bag(str(bag_path)) as bag:
        # Keep integer ROS time; the public float methods lose sub-us precision.
        if not bag._chunks:
            raise ValueError("an indexed, nonempty ROS1 bag is required")
        start = min(chunk.start_time.to_nsec() for chunk in bag._chunks)
        end = max(chunk.end_time.to_nsec() for chunk in bag._chunks)
        for topic, message, record in bag.read_messages(
                topics=[MOCAP, ODOM, STATE, "/tf", "/tf_static"]):
            record_ns = record.to_nsec() - start
            counts[topic] = counts.get(topic, 0) + 1
            topic_extents.setdefault(topic, [record_ns, record_ns])[1] = record_ns
            if topic in (MOCAP, ODOM):
                if message.header.frame_id.strip("/") != "world":
                    raise ValueError("unexpected pose reference frame")
                pose = message.pose if topic == MOCAP else message.pose.pose
                name = "mocap" if topic == MOCAP else "odom"
                streams.setdefault(name, []).append(pose_row(
                    message.header.stamp.to_nsec() - start, record_ns, pose))
            elif topic == STATE:
                if not states or states[-1][1] != message.data:
                    states.append((record_ns, message.data))
            else:
                for transform in message.transforms:
                    parent = transform.header.frame_id.strip("/")
                    child = transform.child_frame_id.strip("/")
                    key = parent + "->" + child
                    p, q = transform.transform.translation, transform.transform.rotation
                    value = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                    if topic == "/tf_static":
                        if key in static and not np.allclose(static[key], value):
                            raise ValueError("static transform changed: " + key)
                        static[key] = value
                    else:
                        streams.setdefault(key, []).append([
                            transform.header.stamp.to_nsec() - start,
                            record_ns] + value)

    arrays = {"state_record_ns": np.array(states, dtype=np.int64)}
    stream_info = {}
    for name, rows in streams.items():
        # Separate int64 time columns from floating-point pose columns.
        times = np.asarray([row[0] for row in rows], dtype=np.int64)
        records = np.asarray([row[1] for row in rows], dtype=np.int64)
        poses = np.asarray([row[2:] for row in rows], dtype=float)
        arrays[name + ":header_ns"] = times
        arrays[name + ":record_ns"] = records
        arrays[name + ":pose"] = poses
        stream_info[name] = {
            "samples": len(rows), "header_start_ns": int(times.min()),
            "header_end_ns": int(times.max()),
            "record_start_ns": int(records.min()), "record_end_ns": int(records.max()),
        }
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache / (case + ".npz")), **arrays)
    metadata = {
        "case": case, "bag_name": bag_path.name,
        "bag_source": str(bag_path), "bag_sha256": sha256(bag_path),
        "bag_size_bytes": bag_path.stat().st_size,
        "bag_start_ns": start, "bag_end_ns": end, "duration_ns": end - start,
        "start_jst": timestamp(start), "end_jst": timestamp(end),
        "estimation_window_seconds": [config["start_seconds"], config["end_seconds"]],
        "topic_message_counts": counts, "topic_record_extents_ns": topic_extents,
        "streams": stream_info, "static_transforms": static,
        "flight_events": [{"elapsed_ns": int(t), "state": int(s),
                           "name": STATE_NAMES.get(int(s), str(s))} for t, s in states],
    }
    (cache / (case + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print("{}: complete bag, {:.9f} s, {} mocap samples".format(
        case, (end - start) / NS, len(streams["mocap"])), flush=True)


def remux_exact(source, destination, pts, durations):
    """Copy FFmpeg's H.264 packets and assign the exact MP4 sample timing."""
    import av

    options = {"movflags": "+faststart", "video_track_timescale": str(NS),
               "movie_timescale": str(NS), "use_editlist": "0"}
    count = 0
    with av.open(str(source)) as inp, av.open(str(destination), "w", options=options) as out:
        stream = out.add_stream_from_template(inp.streams.video[0])
        stream.time_base = Fraction(1, NS)
        for packet in inp.demux(video=0):
            if packet.dts is None:
                continue
            if count >= len(pts) or packet.pts != packet.dts:
                raise ValueError("expected one packet per frame with no B frames")
            packet.stream = stream
            packet.time_base = Fraction(1, NS)
            packet.pts = packet.dts = pts[count]
            packet.duration = durations[count]
            out.mux(packet)
            count += 1
    if count != len(pts):
        raise ValueError("encoded frame count differs from the complete time grid")


def verify_video(path, pts, durations):
    """Check every packet, both endpoints, frame count, and a full decode."""
    import av

    count = 0
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if stream.time_base != Fraction(1, NS) or stream.duration != sum(durations):
            raise ValueError("MP4 duration/time base differs from the bag")
        for packet in video.demux(stream):
            if packet.dts is None:
                continue
            if (count >= len(pts) or packet.pts != pts[count]
                    or packet.dts != pts[count] or packet.duration != durations[count]):
                raise ValueError("unexpected video timestamp at frame " + str(count))
            count += 1
    if count != len(pts):
        raise ValueError("missing or extra video frames")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-threads", "2",
                    "-i", str(path), "-f", "null", "-"], check=True)
    return {"packet_count": count, "first_pts_ns": pts[0],
            "last_pts_ns": pts[-1], "last_frame_duration_ns": durations[-1],
            "end_ns": pts[-1] + durations[-1], "decode_errors": 0}


def render(case, cache, output, preview=False):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation, Slerp

    # EGL renders without a desktop window. Mesa also works on this host's CPU.
    os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    mesa = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    if Path(mesa).exists():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", mesa)
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    meta = json.loads((cache / (case + ".json")).read_text())
    data = np.load(str(cache / (case + ".npz")))
    duration = meta["duration_ns"]
    pts, durations = frame_schedule(duration)
    # The endpoint snapshot is shown in the short last frame (at most 8.334 ms early).
    sample_ns = np.array(pts, dtype=np.int64)
    sample_ns[-1] = duration
    if preview:
        seconds = {"failure1": [0, 7.4, 20, 25.4, duration / NS],
                   "failure2": [0, 11.8, 27, 31.5, duration / NS],
                   "success": [0, 14, 75, 150, 202, duration / NS]}[case]
        maximum_z = np.argmax(data["mocap:pose"][:, 2])
        seconds.append(int(data["mocap:header_ns"][maximum_z]) / NS)
        seconds = sorted(set(seconds))
        sample_ns = np.array([min(duration, round(s * NS)) for s in seconds], dtype=np.int64)

    def sampled(name):
        t = data[name + ":header_ns"]
        pose = data[name + ":pose"]
        order = np.argsort(t, kind="stable")
        t, pose = t[order], pose[order]
        # Keep the last recorded value at a duplicate measurement timestamp.
        keep = np.r_[t[1:] != t[:-1], True]
        t, pose = t[keep], pose[keep]
        query = np.clip(sample_ns, t[0], t[-1])
        p = np.column_stack([np.interp(query, t, pose[:, j]) for j in range(3)])
        if len(t) == 1:
            r = np.repeat(Rotation.from_quat(pose[:, 3:]).as_matrix(), len(query), axis=0)
        else:
            r = Slerp(t.astype(float), Rotation.from_quat(pose[:, 3:]))(query).as_matrix()
        # Do not invent a flight across a measurement outage longer than 100 ms.
        left = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 1)
        right = np.minimum(left + 1, len(t) - 1)
        gap = (t[right] - t[left] > NS // 10) & (query < t[right])
        p[gap] = pose[left[gap], :3]
        r[gap] = Rotation.from_quat(pose[left[gap], 3:]).as_matrix()
        matrices = np.tile(np.eye(4), (len(query), 1, 1))
        matrices[:, :3, :3], matrices[:, :3, 3] = r, p
        return matrices, t, pose, gap

    def constant(key):
        value = meta["static_transforms"][key]
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
        matrix[:3, 3] = value[:3]
        return matrix

    fc, mocap_t, mocap_pose, gaps = sampled("mocap")
    odom, _, _, _ = sampled("odom")
    pending = sample_ns < mocap_t[0]
    fc[pending] = odom[pending]
    root_to_main = constant("gimbalrotor/root->gimbalrotor/main_body")
    if not np.allclose(root_to_main, np.eye(4)):
        raise ValueError("this renderer expects coincident root and main_body frames")
    body = fc @ np.linalg.inv(constant("gimbalrotor/main_body->gimbalrotor/fc"))
    cog_local, _, _, _ = sampled("gimbalrotor/root->gimbalrotor/cog")
    cog = body @ cog_local
    rotors = []
    for number in range(1, 5):
        local, _, _, _ = sampled("gimbalrotor/main_body->gimbalrotor/gimbal{}_link".format(number))
        rotors.append(body @ local @ constant(
            "gimbalrotor/gimbal{}_link->gimbalrotor/thrust{}".format(number, number)))
    states = data["state_record_ns"]
    state_indexes = np.searchsorted(states[:, 0], sample_ns, side="right") - 1
    state_values = [int(states[j, 1]) if j >= 0 else None for j in state_indexes]

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(WIDTH, HEIGHT)
    window.SetMultiSamples(0)
    renderer = vtk.vtkRenderer()
    renderer.SetViewport(0, 0, .77, 1)
    renderer.SetBackground(.043, .059, .082)
    renderer.SetUseFXAA(True)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(*CAMERA_POSITION)
    camera.SetFocalPoint(*CAMERA_FOCAL_POINT)
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(CAMERA_SCALE)
    camera.SetClippingRange(.1, 20)

    def line(points, color, width=2, opacity=1):
        vertices = vtk.vtkPoints()
        vertices.SetData(numpy_to_vtk(np.asarray(points, dtype=float), deep=True))
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(len(points))
        for j in range(len(points)):
            cells.InsertCellPoint(j)
        poly = vtk.vtkPolyData()
        poly.SetPoints(vertices)
        poly.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().LightingOff()
        renderer.AddActor(actor)
        return vertices, poly, cells, actor

    def change_line(item, points):
        vertices, poly, cells, actor = item
        vertices.SetData(numpy_to_vtk(np.asarray(points, dtype=float), deep=True))
        if cells.GetNumberOfConnectivityIds() != len(points):
            cells.Reset()
            cells.InsertNextCell(len(points))
            for j in range(len(points)):
                cells.InsertCellPoint(j)
        vertices.Modified()
        poly.Modified()
        actor.SetVisibility(len(points) >= 2)

    def actor_from(source, color):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        renderer.AddActor(actor)
        return actor

    def sphere(radius, color):
        source = vtk.vtkSphereSource()
        source.SetRadius(radius)
        source.SetThetaResolution(20)
        source.SetPhiResolution(16)
        return actor_from(source, color)

    def transform_actor(actor, matrix):
        transform = vtk.vtkMatrix4x4()
        transform.DeepCopy(matrix.ravel())
        actor.SetUserMatrix(transform)

    axis_colors = [(1, .27, .28), (.30, .87, .40), (.32, .60, 1)]
    fixed_labels = []
    for x in np.arange(-1, 1.01, .25):
        line([[x, -1, 0], [x, .75, 0]], (.16, .22, .29), 1)
    for y in np.arange(-1, .76, .25):
        line([[-1, y, 0], [1, y, 0]], (.16, .22, .29), 1)
    for j, color in enumerate(axis_colors):
        endpoint = np.eye(3)[j] * .30
        line([[0, 0, 0], endpoint], color, 3)
        fixed_labels.append((endpoint, "XYZ"[j], tuple(round(c * 255) for c in color)))
    line([[-.70, -.50, 0], [-.70, -.50, 1.25]], (.42, .48, .56), 1)
    for z in np.arange(0, 1.26, .25):
        line([[-.73, -.50, z], [-.67, -.50, z]], (.42, .48, .56), 1)
        fixed_labels.append(([-.73, -.50, z], "{:.2f} m".format(z), (134, 150, 168)))

    body_source = vtk.vtkCubeSource()
    body_source.SetXLength(.13)
    body_source.SetYLength(.10)
    body_source.SetZLength(.038)
    body_actor = actor_from(body_source, (.65, .72, .80))
    fc_actor = sphere(.017, (1, .76, .25))
    cog_actor = sphere(.015, (.20, .92, .96))
    arms = [line([[0, 0, 0], [0, 0, 0]], (.68, .75, .84), 5) for _ in range(4)]
    rings = [line(np.zeros((49, 3)), (.82, .87, .92), 2) for _ in range(4)]
    hubs = [sphere(.019, (.78, .83, .89)) for _ in range(4)]
    triads = [[line([[0, 0, 0], [0, 0, 0]], c, 2) for c in axis_colors] for _ in range(7)]
    full_trail = line([[0, 0, 0], [0, 0, 0]], (.13, .42, .51), 1.5)
    recent_trail = line([[0, 0, 0], [0, 0, 0]], (.18, .88, .94), 2.5)
    height_line = line([[0, 0, 0], [0, 0, 0]], (.32, .47, .57), 1)
    angles = np.linspace(0, 2 * np.pi, 49)
    ring_local = np.column_stack([.105 * np.cos(angles), .105 * np.sin(angles), np.zeros(49)])

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.ShouldRerenderOff()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    mono_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    fonts = {size: ImageFont.truetype(font_path, size) for size in (11, 12, 13, 15, 18, 24)}
    clock_font = ImageFont.truetype(mono_path, 29)
    text_color, muted = (226, 234, 242), (148, 164, 181)

    def project(point):
        renderer.SetWorldPoint(*point, 1)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        return int(x), HEIGHT - int(y)

    def draw_frame(index):
        t = int(sample_ns[index])
        transform_actor(body_actor, body[index])
        fc_actor.SetPosition(*fc[index, :3, 3])
        cog_actor.SetPosition(*cog[index, :3, 3])
        for j, rotor in enumerate(rotors):
            matrix = rotor[index]
            change_line(arms[j], [body[index, :3, 3], matrix[:3, 3]])
            change_line(rings[j], ring_local @ matrix[:3, :3].T + matrix[:3, 3])
            hubs[j].SetPosition(*matrix[:3, 3])
        for group, matrix, size in zip(triads, [body[index], fc[index], cog[index]]
                                       + [r[index] for r in rotors], [.21, .13, .085] + [.075] * 4):
            for j in range(3):
                change_line(group[j], [matrix[:3, 3], matrix[:3, 3] + size * matrix[:3, j]])
        last = np.searchsorted(mocap_t, t, side="right")
        first = np.searchsorted(mocap_t, t - 2 * NS)
        change_line(full_trail, mocap_pose[:last:3, :3])
        change_line(recent_trail, mocap_pose[first:last, :3])
        position = fc[index, :3, 3]
        change_line(height_line, [[position[0], position[1], 0], position])
        window.Render()
        capture.Modified()
        capture.Update()
        pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
        image = Image.fromarray(pixels.reshape(HEIGHT, WIDTH, 3)[::-1].copy())
        draw = ImageDraw.Draw(image)
        draw.rectangle((740, 0, WIDTH, HEIGHT), fill=(15, 22, 31))
        for point, label, color in fixed_labels:
            x, y = project(point)
            draw.text((x - 35 if label.endswith("m") else x + 4, y), label,
                      fill=color, font=fonts[11], stroke_width=1, stroke_fill=(11, 15, 21))
        for matrix, label, color, offset in [
            (body[index], "main_body", text_color, (-75, -25)),
            (fc[index], "fc", (255, 202, 78), (7, -23)),
            (cog[index], "cog", (61, 228, 239), (7, 8))]:
            x, y = project(matrix[:3, 3])
            draw.text((x + offset[0], y + offset[1]), label, fill=color, font=fonts[12],
                      stroke_width=1, stroke_fill=(11, 15, 21))
        for j, matrix in enumerate(rotors):
            x, y = project(matrix[index, :3, 3])
            draw.text((x - 8, y + 9), "T" + str(j + 1), fill=muted, font=fonts[11],
                      stroke_width=1, stroke_fill=(11, 15, 21))
        draw.rectangle((0, 0, 739, 82), fill=(11, 15, 21))
        draw.text((20, 11), case + "  |  FULL ROSBAG", fill=text_color, font=fonts[18])
        minutes, seconds = divmod(t // 1_000_000, 60_000)
        elapsed = "{:02d}:{:02d}.{:03d}".format(minutes, seconds // 1000, seconds % 1000)
        draw.text((19, 38), elapsed, fill=(246, 249, 252), font=clock_font)
        draw.text((258, 51), "1x real time  /  120 fps", fill=muted, font=fonts[13])
        state = state_values[index]
        draw.text((762, 18), "FLIGHT STATE", fill=muted, font=fonts[12])
        draw.text((762, 39), STATE_NAMES.get(state, "WAITING"),
                  fill=(255, 202, 78), font=fonts[18])
        draw.text((762, 91), "WORLD POSITION  [m]", fill=muted, font=fonts[12])
        for j, axis in enumerate("XYZ"):
            draw.text((762, 116 + 25 * j), "{}   {:+.3f}".format(axis, position[j]),
                      fill=text_color, font=fonts[15])
        rpy = Rotation.from_matrix(fc[index, :3, :3]).as_euler("xyz", degrees=True)
        draw.text((762, 207), "ATTITUDE  [deg]", fill=muted, font=fonts[12])
        for j, label in enumerate(("Roll", "Pitch", "Yaw")):
            draw.text((762, 232 + 25 * j), "{:<5}  {:+6.1f}".format(label, rpy[j]),
                      fill=text_color, font=fonts[15])
        draw.text((762, 330), "COORDINATE FRAMES", fill=muted, font=fonts[12])
        draw.text((762, 352), "world / main_body", fill=text_color, font=fonts[13])
        draw.text((762, 374), "fc / cog / thrust 1-4", fill=text_color, font=fonts[13])
        for j, axis in enumerate("XYZ"):
            draw.text((762 + j * 57, 403), axis,
                      fill=tuple(round(c * 255) for c in axis_colors[j]), font=fonts[18])
        draw.text((762, 438), "Floor grid: 0.25 m", fill=muted, font=fonts[12])
        if pending[index]:
            source_label = "Mocap pending: odometry"
        elif gaps[index]:
            source_label = "Mocap gap: holding pose"
        else:
            source_label = "Recorded mocap + TF"
        draw.text((762, 463), source_label, fill=muted, font=fonts[11])
        draw.rectangle((0, 490, WIDTH, HEIGHT), fill=(11, 15, 21))
        draw.text((20, 503), timestamp(meta["bag_start_ns"] + t), fill=muted, font=fonts[12])
        draw.text((20, 521), meta["bag_name"], fill=muted, font=fonts[11])
        draw.rectangle((420, 506, 714, 510), fill=(40, 52, 64))
        draw.rectangle((420, 506, 420 + int(294 * t / duration), 510), fill=(52, 207, 222))
        for event in meta["flight_events"]:
            if event["state"] in (3, 4):
                x = 420 + int(294 * event["elapsed_ns"] / duration)
                draw.line((x, 502, x, 514), fill=(239, 188, 91), width=1)
        if t == duration:
            draw.text((553, 46), "END OF BAG", fill=(255, 202, 78), font=fonts[15])
        return image

    output.mkdir(parents=True, exist_ok=True)
    if preview:
        for j, t in enumerate(sample_ns):
            image = draw_frame(j)
            image.save(str(output / (case + "_{:.3f}.png".format(t / NS))))
        window.Finalize()
        return

    wall_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="grape-full-bag-") as temporary:
        intermediate = Path(temporary) / "encoded.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", "{}x{}".format(WIDTH, HEIGHT),
                   "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-bf", "0",
                   "-g", str(FPS), "-threads", "4", "-video_track_timescale", str(NS),
                   "-use_editlist", "0", str(intermediate)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for j in range(len(sample_ns)):
                process.stdin.write(draw_frame(j).tobytes())
                if j % (FPS * 5) == 0:
                    print("{}: {:6.2f}/{:.2f} s, {:.1f} rendered fps".format(
                        case, sample_ns[j] / NS, duration / NS,
                        (j + 1) / max(.001, time.monotonic() - wall_start)), flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg encoding failed")
        except BaseException:
            process.stdin.close()
            process.terminate()
            process.wait()
            raise
        finally:
            window.Finalize()
        destination = output / (case + ".mp4")
        remux_exact(intermediate, destination, pts, durations)
    checks = verify_video(destination, pts, durations)
    with gzip.open(str(output / (case + "_frame_times.csv.gz")), "wt", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["frame", "video_pts_ns", "duration_ns", "bag_time_at_pts_ns",
                         "display_sample_bag_ns", "pose_source"])
        for j, (p, d, t) in enumerate(zip(pts, durations, sample_ns)):
            writer.writerow([j, p, d, meta["bag_start_ns"] + p,
                             meta["bag_start_ns"] + int(t),
                             "odom" if pending[j] else "mocap_hold" if gaps[j] else "mocap"])
    with (output / (case + "_flight_events.csv")).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["bag_elapsed_seconds", "bag_timestamp_ns", "flight_state", "name"])
        for event in meta["flight_events"]:
            writer.writerow(["{:.9f}".format(event["elapsed_ns"] / NS),
                             meta["bag_start_ns"] + event["elapsed_ns"],
                             event["state"], event["name"]])
    meta.update({
        "video_file": destination.name, "video_sha256": sha256(destination),
        "video_size_bytes": destination.stat().st_size,
        "frame_count": len(pts), "nominal_fps": FPS, "width": WIDTH, "height": HEIGHT,
        "codec": "H.264", "pixel_format": "yuv420p", "crf": 18,
        "time_base": "1/1000000000", "playback_speed": 1,
        "pose_time_source": "header.stamp relative to bag start; no shift to first pose",
        "pose_interpolation": "linear position, quaternion SLERP; hold across gaps >100 ms",
        "initial_pose": "recorded FC odometry until first mocap measurement",
        "endpoint_snapshot_advance_ns": duration - pts[-1],
        "camera_position": CAMERA_POSITION, "camera_focal_point": CAMERA_FOCAL_POINT,
        "camera_parallel_scale": CAMERA_SCALE, "verification": checks,
    })
    (output / (case + "_timing.json")).write_text(json.dumps(meta, indent=2) + "\n")
    print("{}: verified {} frames, {:.9f} s, {:.2f} MiB".format(
        case, len(pts), duration / NS, destination.stat().st_size / 2**20), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "render"])
    parser.add_argument("--case", choices=list(CASES) + ["all"], default="all")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--bag-root", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    for case in CASES if args.case == "all" else [args.case]:
        if args.command == "extract":
            extract(case, args.cache_dir, args.bag_root)
        else:
            render(case, args.cache_dir, args.output, args.preview)


if __name__ == "__main__":
    main()
