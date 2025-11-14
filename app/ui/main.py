"""Application entrypoint that opens the Tk dashboard immediately."""

from __future__ import annotations

import threading
import traceback

from ..pipeline import run_pipeline
from ..utils import log_error, log_info
from .viewer import launch_viewer


def _run_pipeline_async(viewer) -> None:
    """Execute the heavy pipeline work in a background thread."""

    log_info("Pipeline started in background thread. Logs werden im UI angezeigt.")
    try:
        artifacts = run_pipeline()
    except Exception as exc:  # pragma: no cover - defensive logging
        log_error(f"Pipeline execution failed: {exc}")
        log_error(traceback.format_exc())
        viewer.after(0, lambda: viewer.set_status("Pipeline fehlgeschlagen – siehe Log."))
    else:
        records = artifacts.get("records", [])
        viewer.after(0, lambda: viewer.update_records(records))


def main() -> None:
    """Launch Tkinter UI and kick off the pipeline."""

    viewer = launch_viewer()
    thread = threading.Thread(target=_run_pipeline_async, args=(viewer,), daemon=True)
    thread.start()
    log_info("Tkinter-Dashboard gestartet. Pipeline laeuft im Hintergrund.")
    viewer.mainloop()


if __name__ == "__main__":
    main()

