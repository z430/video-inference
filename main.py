import time
from pathlib import Path

from vip.handler import create_input_stream


def main(args):
    stream = create_input_stream(Path(args.input))

    while True:
        start = time.perf_counter()
        frame = stream.read()
        if frame is None:
            break

        print(f"FPS: {1 / (time.perf_counter() - start)}")

    stream.stop()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="VIP")
    parser.add_argument("--input", help="Input video file")
    parser.add_argument("--output", help="Output Result video file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
