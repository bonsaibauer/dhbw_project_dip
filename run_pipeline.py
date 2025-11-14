"""CLI entrypoint for the fryum inspection pipeline."""

from app import config
from app.pipeline import run_pipeline
from app.ui.main import _start_pipeline, _stop_pipeline  # reuse threading helper
from app.ui.viewer import launch_viewer


if __name__ == "__main__":
    if config.missing_paths():
        print("Pfad-Setup erforderlich. Bitte starte das Tk-Dashboard (setup_and_run.bat) und konfiguriere die Datenpfade.")
    else:
        artifacts = run_pipeline()
        print(f"Accuracy: {artifacts['evaluation'].accuracy:.4f}")
        viewer = launch_viewer(
            records=artifacts.get("records"),
            start_callback=lambda opts: _start_pipeline(viewer, opts),
            stop_callback=lambda: _stop_pipeline(viewer),
        )
        viewer.mainloop()
