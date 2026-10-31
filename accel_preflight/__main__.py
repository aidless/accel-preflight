# -*- coding: utf-8 -*-
"""python -m accel_preflight 入口。"""
import sys

from accel_preflight import cli

if __name__ == "__main__":
    sys.exit(cli.main(sys.argv[1:]))
