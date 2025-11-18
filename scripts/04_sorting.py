import csv
import json
import os
import shutil
import stat
from functools import lru_cache

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_dir = os.path.join(project_root, "config")
path_path = os.path.join(config_dir, "path.json")


def load_json(path, error_msg):
    if not os.path.exists(path):
        raise FileNotFoundError(error_msg)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_path_config():
    return load_json(
        path_path,
        f"Pfaddatei '{path_path}' nicht gefunden.",
    )


def norm_path(path_value):
    return os.path.normpath(path_value) if path_value else ""


def load_paths():
    return {
        key: norm_path(value)
        for key, value in load_path_config().get("paths", {}).items()
    }


SORT_LOG_ENABLED = True


def render_table(headers, rows, indent="  "):
    """Gibt eine Tabelle mit fester Spaltenbreite aus."""
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header_line = indent + " | ".join(
        headers[i].ljust(widths[i]) for i in range(len(headers))
    )
    divider_line = indent + "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(divider_line)
    for row in rows:
        print(indent + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    print()

path_map = load_paths()
pipe_csv = path_map["pipeline_csv_path"]
sorted_dir = path_map["sorted_output_directory"]
sort_flag = SORT_LOG_ENABLED


def show_progress(prefix, current, total, bar_len=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    label = prefix.ljust(20)
    print(f"\r{label}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


def read_rows(csv_path):
    if not os.path.exists(csv_path):
        print(f"Fehler: CSV '{csv_path}' nicht gefunden.")
        return [], []
    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [row for row in reader]
        headers = reader.fieldnames or []
    return rows, headers


def ensure_cols(headers, required):
    updated = list(headers)
    for name in required:
        if name not in updated:
            updated.append(name)
    return updated


def write_rows(csv_path, headers, rows):
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def try_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def base_filename(row):
    return row.get("filename") or os.path.basename(row.get("source_path", ""))


def prefixed_name(row, base_name, target_class):
    if target_class.lower() != "normal":
        return base_name

    score = try_float(row.get("symmetry_score"))
    if score is None:
        return base_name

    return f"{score:06.2f}_{base_name}"


def resolve_destination_name(row):
    if row.get("destination_filename"):
        return row["destination_filename"]

    target_class = row.get("target_class") or ""
    filename = base_filename(row)

    return prefixed_name(row, filename, target_class)


def clear_folder(folder):
    if not os.path.exists(folder):
        return

    def _on_rm_error(func, path, exc_info):
        if isinstance(exc_info[1], PermissionError):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        else:
            raise

    shutil.rmtree(folder, onerror=_on_rm_error)


def sort_images(csv_path, sorted_dir, log_progress=True):
    rows, headers = read_rows(csv_path)
    if not rows:
        return

    clear_folder(sorted_dir)

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

        filename = resolve_destination_name(row)
        row["destination_filename"] = filename
        src_path = row.get("source_path", "")
        if not src_path or not os.path.exists(src_path):
            continue

        dest_dir = os.path.join(sorted_dir, target_class)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy(src_path, dest_path)
        row["sorted_path"] = dest_path

        if log_progress and total_files > 0:
            show_progress("  Sortierung", idx, total_files)

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

    headers = ensure_cols(headers, ["sorted_path", "destination_filename"])
    write_rows(csv_path, headers, rows)


def sort_cli():
    sort_images(
        pipe_csv,
        sorted_dir,
        sort_flag,
    )


if __name__ == "__main__":
    sort_cli()
