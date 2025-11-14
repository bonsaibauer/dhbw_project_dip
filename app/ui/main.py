"""Application entrypoint that opens the Tk dashboard immediately."""

from __future__ import annotations

import threading
import traceback

from .. import config
from ..pipeline import PipelineOptions, run_pipeline
from ..utils import log_error, log_info
from .viewer import launch_viewer

_CURRENT_RUN: dict[str, threading.Event | threading.Thread | None] = {
    "thread": None,
    "cancel_event": None,
}


def _run_pipeline_async(viewer, options: PipelineOptions, cancel_event: threading.Event | None) -> None:
    """Execute the heavy pipeline work in a background thread."""

    log_info(
        "Pipeline started in background thread. Logs werden im UI angezeigt.",
    )
    try:
        artifacts = run_pipeline(options, cancel_event=cancel_event)
    except RuntimeError as exc:
        message = str(exc)
        if "Pipeline cancelled" in message:
            viewer.after(0, lambda: viewer.set_status("Pipeline abgebrochen."))
        else:  # pragma: no cover
            log_error(f"Pipeline runtime error: {exc}")
            viewer.after(0, lambda: viewer.set_status("Pipeline-Fehler - siehe Log."))
    except Exception as exc:  # pragma: no cover - defensive logging
        log_error(f"Pipeline execution failed: {exc}")
        log_error(traceback.format_exc())
        viewer.after(0, lambda: viewer.set_status("Pipeline fehlgeschlagen - siehe Log."))
    else:
        records = artifacts.get("records", [])
        viewer.after(0, lambda: viewer.update_records(records))
    finally:
        viewer.after(0, lambda: viewer.set_running(False))
        _CURRENT_RUN["thread"] = None
        _CURRENT_RUN["cancel_event"] = None


def _start_pipeline(viewer, options: PipelineOptions | None = None) -> None:
    """Helper that spawns a daemon thread for the heavy processing."""

    if config.missing_paths():
        viewer.after(0, viewer.notify_setup_required)
        return
    if _CURRENT_RUN["thread"] and _CURRENT_RUN["thread"].is_alive():  # type: ignore[union-attr]
        return
    effective_options = options or PipelineOptions()
    cancel_event = threading.Event()
    _CURRENT_RUN["cancel_event"] = cancel_event
    viewer.set_running(True)
    thread = threading.Thread(
        target=_run_pipeline_async,
        args=(viewer, effective_options, cancel_event),
        daemon=True,
    )
    _CURRENT_RUN["thread"] = thread
    thread.start()


def _stop_pipeline(viewer) -> None:
    """Signals the current run to stop."""

    event = _CURRENT_RUN.get("cancel_event")
    thread = _CURRENT_RUN.get("thread")
    if event and isinstance(event, threading.Event) and thread and thread.is_alive():  # type: ignore[union-attr]
        event.set()
        viewer.set_status("Stop angefordert...")
    else:
        viewer.set_status("Kein laufender Auftrag aktiv.")


def main() -> None:
    """Launch Tkinter UI and kick off the pipeline."""

    viewer = launch_viewer(
        start_callback=lambda opts: _start_pipeline(viewer, opts),
        stop_callback=lambda: _stop_pipeline(viewer),
    )
    _start_pipeline(viewer, PipelineOptions())
    log_info("Tkinter-Dashboard gestartet. Pipeline laeuft im Hintergrund.")
    viewer.mainloop()


if __name__ == "__main__":
    main()
