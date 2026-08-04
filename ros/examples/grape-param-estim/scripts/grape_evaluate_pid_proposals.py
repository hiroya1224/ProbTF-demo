#!/usr/bin/env python3
"""ROS-installed entry point for posterior PID candidate evaluation."""

from grape_param_estim.pid.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
