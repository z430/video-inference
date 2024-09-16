import time
from pathlib import Path

from alive_progress import alive_bar

from vip.handler import create_input_stream
from vip.models import YOLOv5
from vip.utils import nvtx_range

INPUT_STREAM = Path("data/shinjuku-live.mp4")

def main():
    stream = create_input_stream(INPUT_STREAM)
    engine = YOLOv5()

    with alive_bar(max_cols=200) as bar:
        while True:
            with nvtx_range("pipeline"):
                start = time.perf_counter()
                with nvtx_range("read"):
                    frame = stream.read()
                results = engine.predict(frame)
                if frame is None:
                    break

                text = f"FPS: {1 / (time.perf_counter() - start)}"
                bar.title = "Processing"
                bar.text = text
                bar()

    stream.stop()


if __name__ == "__main__":
    main()
