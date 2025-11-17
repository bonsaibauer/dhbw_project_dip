import csv
import os


def path_rel(path):
    """Bringt Pfadangaben in das Format der Annotationen."""
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    marker = "Data/Images/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def label_map(raw_label, label_priorities):
    """Wählt das Label mit der höchsten Priorität aus."""
    if not raw_label:
        return None
    candidates = [lbl.strip().lower() for lbl in raw_label.split(",") if lbl.strip()]
    if not candidates:
        return None
    candidates.sort(key=lambda lbl: label_priorities.get(lbl, 100))
    return candidates[0]


def anno_load(annotation_file, label_priorities, label_class_map):
    """Lädt CSV-Annotationen und mapped sie auf die vier Zielklassen."""
    annotations = {}
    if not os.path.exists(annotation_file):
        print(f"\nHinweis: '{annotation_file}' nicht gefunden, Validierung übersprungen.")
        return annotations

    with open(annotation_file, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rel_path = path_rel(row.get("image", ""))
            if not rel_path:
                continue
            base_label = label_map(row.get("label", ""), label_priorities)
            if not base_label:
                continue
            annotations[rel_path] = label_class_map.get(base_label, "Rest")

    return annotations
