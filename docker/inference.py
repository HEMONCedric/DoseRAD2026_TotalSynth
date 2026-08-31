"""Container entry point: read `/input` and write `/output`."""

from pathlib import Path

from doserad2026.runtime import ProtonRuntime


def run(runtime: ProtonRuntime) -> int:
    runtime.predict(Path("/input"), Path("/output"))
    return 0


if __name__ == "__main__":
    standalone = ProtonRuntime()
    standalone.load()
    raise SystemExit(run(standalone))
