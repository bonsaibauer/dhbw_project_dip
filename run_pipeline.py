"""CLI entrypoint for the fryum inspection pipeline."""

from app.pipeline import run_pipeline
from app.ui.viewer import launch_viewer


if __name__ == "__main__":
    artifacts = run_pipeline()
    print(f"Accuracy: {artifacts['evaluation'].accuracy:.4f}")
    launch_viewer(artifacts.get("records"))
