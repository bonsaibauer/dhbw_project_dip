import os
import shutil

from data_management import path_rel


def tbl_show(headers, rows, indent="  "):
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


def prio_map(label_priorities, label_class_map):
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


def miss_copy(pred_entry, expected_label, falsch_dir):
    """Kopiert ein fehlklassifiziertes Bild in den Kontrollordner."""
    os.makedirs(falsch_dir, exist_ok=True)
    rel_name = path_rel(pred_entry.get("relative_path", ""))
    if not rel_name:
        rel_name = os.path.basename(pred_entry["source_path"])
    rel_name = rel_name.replace("/", "_")
    base, ext = os.path.splitext(rel_name)
    safe_expected = expected_label.replace(" ", "_")
    safe_pred = pred_entry["predicted"].replace(" ", "_")
    new_name = f"{base}_gt-{safe_expected}_pred-{safe_pred}{ext}"
    dest_path = os.path.join(falsch_dir, new_name)
    shutil.copy(pred_entry["source_path"], dest_path)


def pred_chk(predictions, annotations, falsch_dir, label_priorities, label_class_map):
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
        rel_path = path_rel(pred.get("relative_path", ""))
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
    print("- Gesamtstatistik:\n")
    summary_headers = ["Statistik", "Wert"]
    summary_rows = [
        ["Bewertet", str(total)],
        ["Treffer", str(correct)],
        ["Genauigkeit %", f"{accuracy:.2f}"],
        ["Falsch zugeordnet", str(len(mismatches))],
    ]
    if skipped:
        summary_rows.append(["Ohne passende Annotation", str(skipped)])
    tbl_show(summary_headers, summary_rows)
    chain = prio_map(label_priorities, label_class_map)
    if chain:
        print(f"Priorisierung (höchste Priorität links): {chain}\n")

    if per_class:
        print("- Klassenübersicht:")
        headers = ["Klasse", "Erwartet", "Treffer", "Genauigkeit %"]
        rows = []
        for cls_name in sorted(per_class.keys()):
            stats = per_class[cls_name]
            acc_cls = (
                (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
            )
            rows.append(
                [
                    cls_name,
                    str(stats["total"]),
                    str(stats["correct"]),
                    f"{acc_cls:.2f}",
                ]
            )
        tbl_show(headers, rows)

    if mismatches:
        for pred, expected in mismatches:
            miss_copy(pred, expected, falsch_dir)
