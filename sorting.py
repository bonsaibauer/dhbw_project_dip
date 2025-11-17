import csv
import csv
import os
import shutil
import stat

from settings import get_paths, get_sort_log
from validation import render_table

PATHS = get_paths()
PIPELINE_CSV = PATHS["PIPELINE_CSV"]
SORT_DIR = PATHS["SORT_DIR"]
SORT_LOG = get_sort_log()


def progress_bar(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


def load_rows(csv_path):
    if not os.path.exists(csv_path):
        print(f"Fehler: CSV '{csv_path}' nicht gefunden.")
        return [], []
    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader]
        headers = reader.fieldnames or []
    return rows, headers


def ensure_columns(headers, required):
    updated = list(headers)
    for name in required:
        if name not in updated:
            updated.append(name)
    return updated


def store_rows(csv_path, headers, rows):
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def clean_directory(folder):
    if not os.path.exists(folder):
        return

    def _on_rm_error(func, path, exc_info):
        if isinstance(exc_info[1], PermissionError):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        else:
            raise

    shutil.rmtree(folder, onerror=_on_rm_error)


def sort_images_from_csv(csv_path, sorted_dir, log_progress=True):
    rows, headers = load_rows(csv_path)
    if not rows:
        return

    clean_directory(sorted_dir)

    classes = ["Normal", "Bruch", "Farbfehler", "Rest"]
    class_counts = {cls: 0 for cls in classes}
    for cls in classes:
        os.makedirs(os.path.join(sorted_dir, cls), exist_ok=True)

    total_files = len(rows)

    for idx, row in enumerate(rows, 1):
        target_class = row.get("target_class") or "Rest"
        if target_class not in classes:
            classes.append(target_class)
            os.makedirs(os.path.join(sorted_dir, target_class), exist_ok=True)
        if target_class not in class_counts:
            class_counts[target_class] = 0
        class_counts[target_class] += 1

        filename = (
            row.get("destination_filename")
            or row.get("filename")
            or os.path.basename(row.get("source_path", ""))
        )
        src_path = row.get("source_path", "")
        if not src_path or not os.path.exists(src_path):
            continue

        dest_dir = os.path.join(sorted_dir, target_class)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy(src_path, dest_path)
        row["sorted_path"] = dest_path

        if log_progress and total_files > 0:
            progress_bar("  Sortierung", idx, total_files)

    if total_files > 0:
        if log_progress:
            print()
        print("\nErgebnisübersicht:\n")
        summary_headers = ["Klasse", "Anzahl", "Anteil %"]
        summary_rows = []
        for cls in classes:
            amount = class_counts.get(cls, 0)
            share = (amount / total_files * 100) if total_files else 0.0
            summary_rows.append([cls, str(amount), f"{share:.1f}"])
        render_table(summary_headers, summary_rows)

    headers = ensure_columns(headers, ["sorted_path"])
    store_rows(csv_path, headers, rows)


def main():
    sort_images_from_csv(PIPELINE_CSV, SORT_DIR, SORT_LOG)


if __name__ == "__main__":
    main()
