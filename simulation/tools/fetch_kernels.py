"""Explicitly download public NAIF kernels; never download during simulation."""

import argparse
from pathlib import Path
from urllib.request import urlopen

BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
KERNELS = {
    "naif0012.tls": "lsk/naif0012.tls",
    "de440s.bsp": "spk/planets/de440s.bsp",
    "earth_latest_high_prec.bpc": "pck/earth_latest_high_prec.bpc",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("simulation/data/kernels"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, relative_url in KERNELS.items():
        destination = args.destination / name
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(BASE + relative_url, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(destination)


if __name__ == "__main__":
    main()
