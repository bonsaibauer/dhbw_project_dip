"""CLI entrypoint for the fryum inspection pipeline."""

from app.pipeline import run_pipeline


if __name__ == "__main__":
    artifacts = run_pipeline()
    print(f"Accuracy: {artifacts['evaluation'].accuracy:.4f}")
