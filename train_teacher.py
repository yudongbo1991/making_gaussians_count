"""Train the bundled upstream2DGS teacher and save the 15k teacher."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source", type=Path, required=True)
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--iterations", type=int, choices=[15000], default=15000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.model.exists():
        raise FileExistsError("Use a fresh teacher model directory.")
    args.model.mkdir(parents=True)
    saves = [15000]
    command = [
        sys.executable,
        "-u",
        str(root / "teacher/train.py"),
        "-s",
        str(args.source.resolve()),
        "-m",
        str(args.model.resolve()),
        "--eval",
        "--images",
        args.images,
        "--resolution",
        str(args.resolution),
        "--data_device",
        "cpu",
        "--iterations",
        str(args.iterations),
        "--save_iterations",
        *map(str, saves),
        "--test_iterations",
        str(args.iterations),
        "--quiet",
    ]
    started = time.time()
    with (args.model / "console.log").open("x") as stream:
        child = subprocess.run(
            command,
            cwd=root / "teacher",
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=str(root / "teacher")),
        )
    record = dict(
        exit_code=child.returncode,
        wall_seconds=time.time() - started,
        command=command,
        teacher_iterations=args.iterations,
        saved_iterations=saves,
    )
    (args.model / "exit_status.json").write_text(json.dumps(record, indent=2))
    if child.returncode:
        raise SystemExit(child.returncode)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
