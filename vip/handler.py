from pathlib import Path

from vidgear.gears import VideoGear


def create_input_stream(input_path: Path) -> VideoGear:
    assert input_path.exists(), f"File {input_path} does not exist"
    return VideoGear(source=str(input_path)).start()
