"""Build the portable processing core for local Python/driver/HIL adapters."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build(output_directory: Path, compiler: str | None = None) -> Path:
    if compiler:
        command = [compiler]
        if Path(compiler).stem.lower() == "zig":
            command.append("cc")
    elif importlib.util.find_spec("ziglang") is not None:
        command = [sys.executable, "-m", "ziglang", "cc"]
    else:
        executable = next(
            (path for name in ("cc", "gcc", "clang") if (path := shutil.which(name))),
            None,
        )
        if executable is None:
            raise RuntimeError(
                "No C99 compiler found. Supply --cc or install ziglang==0.16.0 "
                "in this Python environment."
            )
        command = [executable]
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        "openadcs_processing.dll" if sys.platform == "win32"
        else "libopenadcs_processing.dylib" if sys.platform == "darwin"
        else "libopenadcs_processing.so"
    )
    output = output_directory / filename
    command += [
        "-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-shared", "-DOPENADCS_SHARED", "-I", str(ROOT / "onboard" / "include"),
    ]
    if sys.platform != "win32":
        command.append("-fPIC")
    command += [
        str(ROOT / "onboard" / "src" / name)
        for name in ("sun_sensor.c", "gyro.c", "adcs.c", "gps.c")
    ]
    command += ["-lm", "-o", str(output)]
    subprocess.run(command, check=True, cwd=ROOT)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", help="C compiler executable, or zig executable")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build" / "onboard",
    )
    args = parser.parse_args()
    print(build(args.output, args.cc))


if __name__ == "__main__":
    main()
