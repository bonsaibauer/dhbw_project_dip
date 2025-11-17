import csv
import os
import shutil

from main import (
    get_label_class_map,
    get_label_priorities,
    get_paths,
)

PATHS = get_paths()
PIPELINE_CSV = PATHS["PIPELINE_CSV"]
ANNO_FILE = PATHS["ANNO_FILE"]
FAIL_DIR = PATHS["FAIL_DIR"]
LABEL_PRIORITIES = get_label_priorities()
LABEL_CLASS_MAP = get_label_class_map()
DEFAULT_CLASSES = ["Normal", "Bruch", "Farbfehler", "Rest"]


def normalize_path(path):
    """Bringt Pfadangaben in das Format der Annotationen."""
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    marker = "Data/Images/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def select_label(raw_label, label_priorities):
    """Wählt das Label mit der höchsten Priorität aus."""
    if not raw_label:
        return None
    candidates = [lbl.strip().lower() for lbl in raw_label.split(",") if lbl.strip()]
    if not candidates:
        return None
    candidates.sort(key=lambda lbl: label_priorities.get(lbl, 100))
    return candidates[0]


def load_annotations(annotation_file, label_priorities, label_class_map):
    """Lädt CSV-Annotationen und mapped sie auf die Zielklassen."""
    annotations = {}
    if not os.path.exists(annotation_file):
        print(f"\nHinweis: '{annotation_file}' nicht gefunden, Validierung übersprungen.")
        return annotations

    with open(annotation_file, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rel_path = normalize_path(row.get("image", ""))
            if not rel_path:
                continue
            base_label = select_label(row.get("label", ""), label_priorities)
            if not base_label:
                continue
            annotations[rel_path] = label_class_map.get(base_label, "Rest")

    return annotations


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


def priority_chain(label_priorities, label_class_map):
    """Erstellt einen String mit der Priorisierungskette."""
    class_priority = {}
    for label, prio in label_priorities.items():
        cls = label_class_map.get(label, label.title())
        if cls not in class_priority or prio < class_priority[cls]:
            class_priority[cls] = prio
    if not class_priority:
        return ""
    ordered = [
        name for name, _ in sorted(class_priority.items(), key=lambda item: item[1])
    ]
    return " > ".join(ordered)


def copy_mismatch(pred_entry, expected_label, falsch_dir):
    """Kopiert ein fehlklassifiziertes Bild in den Kontrollordner."""
    os.makedirs(falsch_dir, exist_ok=True)
    rel_name = normalize_path(pred_entry.get("relative_path", ""))
    if not rel_name:
        rel_name = os.path.basename(pred_entry["source_path"])
    rel_name = rel_name.replace("/", "_")
    base, ext = os.path.splitext(rel_name)
    safe_expected = expected_label.replace(" ", "_")
    safe_pred = pred_entry["predicted"].replace(" ", "_")
    new_name = f"{base}_gt-{safe_expected}_pred-{safe_pred}{ext}"
    dest_path = os.path.join(falsch_dir, new_name)
    shutil.copy(pred_entry["source_path"], dest_path)


def validate_predictions(predictions, annotations, falsch_dir, label_priorities, label_class_map):
    """Vergleicht Vorhersagen mit Annotationen und erstellt eine Auswertung."""
    if not annotations:
        print("\nKeine Annotationen geladen -> Validierung übersprungen.")
        return

    if os.path.exists(falsch_dir):
        shutil.rmtree(falsch_dir)

    total = 0
    correct = 0
    mismatches = []
    per_class = {}

    for pred in predictions:
        rel_path = normalize_path(pred.get("relative_path", ""))
        expected = annotations.get(rel_path)
        if expected is None:
            continue
        total += 1
        cls_stats = per_class.setdefault(expected, {"total": 0, "correct": 0})
        cls_stats["total"] += 1
        if expected == pred["predicted"]:
            correct += 1
            cls_stats["correct"] += 1
        else:
            mismatches.append((pred, expected))

    if total == 0:
        print("\nKeine passenden Annotationen gefunden -> Validierung übersprungen.")
        return

    accuracy = (correct / total) * 100
    skipped = len(predictions) - total
    print("\nValidierung (image_anno.csv):")
    headers = ["Klasse", "Erwartet", "Treffer", "Genauigkeit %", "Falsch"]
    rows = [
        ["Gesamt", str(total), str(correct), f"{accuracy:.2f}", str(len(mismatches))]
    ]
    if skipped:
        rows.append(["Ohne passende Annotation", str(skipped), "-", "-", "-"])
    class_order = DEFAULT_CLASSES[:]
    class_order.extend(
        name for name in sorted(per_class.keys()) if name not in DEFAULT_CLASSES
    )
    for cls_name in class_order:
        stats = per_class.get(cls_name, {"total": 0, "correct": 0})
        acc_cls = (
            (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
        )
        rows.append(
            [
                cls_name,
                str(stats["total"]),
                str(stats["correct"]),
                f"{acc_cls:.2f}",
                "-",
            ]
        )
    render_table(headers, rows)
    chain = priority_chain(label_priorities, label_class_map)
    if chain:
        print(f"Priorisierung (höchste Priorität links): {chain}\n")
    if mismatches:
        for pred, expected in mismatches:
            copy_mismatch(pred, expected, falsch_dir)


def load_predictions_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Fehler: CSV '{csv_path}' nicht gefunden.")
        return []

    predictions = []
    with open(csv_path, encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row.get("target_class"):
                continue
            predictions.append(
                {
                    "relative_path": row.get("relative_path", ""),
                    "predicted": row.get("target_class", "Rest"),
                    "source_path": row.get("source_path", ""),
                    "reason": row.get("reason", ""),
                }
            )
    return predictions


def main():
    predictions = load_predictions_from_csv(PIPELINE_CSV)
    annotations = load_annotations(ANNO_FILE, LABEL_PRIORITIES, LABEL_CLASS_MAP)
    validate_predictions(
        predictions,
        annotations,
        FAIL_DIR,
        LABEL_PRIORITIES,
        LABEL_CLASS_MAP,
    )


if __name__ == "__main__":
    main()
