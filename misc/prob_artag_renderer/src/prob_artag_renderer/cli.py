"""Command-line entry points for single frames, scenes, and sequences."""

import argparse
from typing import Any, Dict, Optional, Sequence

from .camera_model import CameraModel
from .config import load_config
from .dataset_writer import DatasetWriter
from .renderer import AprilTagRenderer
from .scene_sampler import SCENARIOS, SceneSampler


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML override file")
    parser.add_argument("--output", required=True, help="dataset output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=SCENARIOS, default="frontal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=("egl", "osmesa", "pyglet"))


def _objects(args: argparse.Namespace):
    overrides: Optional[Dict[str, Any]] = None
    if args.backend:
        overrides = {"render": {"backend": args.backend}}
    config = load_config(args.config, overrides)
    camera = CameraModel.from_dict(config["camera"])
    return config, camera, SceneSampler(camera, config, args.seed)


def _render_samples(args: argparse.Namespace, samples: Sequence[Any], config: Dict[str, Any],
                    camera: CameraModel) -> int:
    writer = DatasetWriter(args.output, overwrite=args.overwrite)
    renderer = AprilTagRenderer(camera, config)
    try:
        for sample in samples:
            path = writer.write(renderer.render(sample))
            print(path)
    finally:
        renderer.close()
    return 0


def render_single_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render one deterministic AprilTag frame")
    _common(parser)
    args = parser.parse_args(argv)
    config, camera, sampler = _objects(args)
    return _render_samples(args, [sampler.sample(args.scenario, count=1)], config, camera)


def render_scene_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render one deterministic multi-tag scene")
    _common(parser)
    parser.set_defaults(scenario="multi_tag")
    parser.add_argument("--count", type=int)
    args = parser.parse_args(argv)
    config, camera, sampler = _objects(args)
    return _render_samples(args, [sampler.sample(args.scenario, count=args.count)], config, camera)


def generate_dataset_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render a continuous-camera dataset")
    _common(parser)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--count", type=int)
    args = parser.parse_args(argv)
    config, camera, sampler = _objects(args)
    samples = sampler.sample_sequence(args.scenario, args.frames, count=args.count)
    return _render_samples(args, samples, config, camera)
