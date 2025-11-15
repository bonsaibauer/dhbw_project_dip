import shutil
import cv2
import numpy as np
import os
import csv

# ==========================================
# ANPASSBARE PARAMETER & VERZEICHNISSE
# ==========================================
RAW_DATA_DIR = os.path.join("data", "Images")
OUTPUT_DIR = "output"
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed")
SORTED_DATA_DIR = os.path.join(OUTPUT_DIR, "sorted")
FALSCH_DIR = os.path.join(OUTPUT_DIR, "Falsch")
ANNOTATION_FILE = os.path.join("data", "image_anno.csv")
VERBOSE_SORT_OUTPUT = True

LOWER_GREEN = np.array([35, 40, 30])
UPPER_GREEN = np.array([85, 255, 255])
CONTOUR_AREA_MIN = 30000
WARP_SIZE = (600, 400)
TARGET_WIDTH = 400
TARGET_HEIGHT = 400

EPSILON_FACTOR = 0.04
MIN_HOLE_AREA = 100
MIN_WINDOW_AREA = 500
MAX_CENTER_HOLE_AREA = 3000

EROSION_KERNEL_SIZE = (5, 5)
EROSION_ITERATIONS = 5
BLACKHAT_KERNEL_SIZE = (15, 15)
BLACKHAT_CONTRAST_THRESHOLD = 30
NOISE_KERNEL_SIZE = (2, 2)
DEFECT_SPOT_THRESHOLD = 15
SYMMETRY_SENSITIVITY = 3.0

LABEL_PRIORITIES = {
    "middle breakage": 0,
    "corner or edge breakage": 0,
    "fryum stuck together": 1,
    "different colour spot": 2,
    "similar colour spot": 2,
    "burnt": 2,
    "small scratches": 3,
    "other": 4,
    "normal": 5,
}

LABEL_CLASS_MAP = {
    "normal": "Normal",
    "middle breakage": "Bruch",
    "corner or edge breakage": "Bruch",
    "fryum stuck together": "Rest",
    "different colour spot": "Farbfehler",
    "similar colour spot": "Farbfehler",
    "burnt": "Farbfehler",
    "small scratches": "Rest",
    "other": "Rest",
}

CLASS_DESCRIPTIONS = {
    "Normal": "7 Löcher, symmetrisch und ohne sichtbare Flecken.",
    "Bruch": "Lochanzahl passt nicht -> Bruch oder Fragment.",
    "Farbfehler": "Flecken/Schatten erkannt, obwohl Geometrie ok.",
    "Rest": "Kein Objekt oder unklare Form/Fall außerhalb der Regeln.",
}

# ==========================================
# TEIL 1: BILDVORVERARBEITUNG (SEGMENTIERUNG)
# ==========================================

def run_preprocessing(image, result):
    image_copy = image.copy()
    image_work = image.copy() # Arbeitskopie für Maskierung

    # --- HSV Hintergrundentfernung ---
    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    
    # Grün-Definition für Hintergrund (Anpassbar je nach Licht)
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_object = cv2.bitwise_not(mask_green) # Objekt ist Nicht-Grün
    
    image_work = cv2.bitwise_and(image_work, image_work, mask=mask_object)

    # Konturen finden
    _, thresh = cv2.threshold(cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    processed = False
    for ele in contours:
        # Nur große Objekte beachten (Filterung von Rauschen)
        if cv2.contourArea(ele) > CONTOUR_AREA_MIN:
            rect = cv2.minAreaRect(ele)
            
            # Querformat erzwingen (Winkel und Dimensionen tauschen)
            # rect = ((center_x, center_y), (width, height), angle)
            if rect[1][1] > rect[1][0]:
                rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
            
            boxf = cv2.boxPoints(rect)
            boxf = np.int64(boxf) # Konvertierung zu Integer für drawContours

            # Maskierung innerhalb der Box
            mask = np.zeros((image_copy.shape[0], image_copy.shape[1])).astype(np.uint8)
            cv2.drawContours(mask, [ele], -1, (255), cv2.FILLED) # Weiß füllen
            
            # Alles außerhalb der Kontur schwarz machen
            image_work[mask == 0] = (0, 0, 0)

            # Perspektivische Transformation (Warp)
            size = WARP_SIZE
            # Ziel-Koordinaten für den Warp
            dst_pts = np.array([[0, size[1]-1], [0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1]], dtype="float32")
            
            # Sortieren der boxf Punkte, damit sie zu dst_pts passen
            # (Dies ist eine Vereinfachung, idealerweise nutzt man eine order_points Funktion)
            # Hier verlassen wir uns auf cv2.minAreaRect Reihenfolge, was oft okay ist, aber riskant.
            
            M = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)
            
            warped = cv2.warpPerspective(image_work, M, size, cv2.INTER_CUBIC)
            warped = cv2.resize(warped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

            result.append({"name": "Result", "data": warped})
            processed = True
            
    return processed

def prepare_dataset(source_dir, target_dir):
    """Liest Bilder ein, segmentiert sie und speichert sie in target_dir."""
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    image_files = collect_image_files(source_dir)
    total_files = len(image_files)
    if total_files == 0:
        print(f"Keine Bilder in '{source_dir}' gefunden.")
        return

    print(f"Segmentierung ({total_files} Bilder):")
    class_dirs = {}
    
    for idx, (root, name) in enumerate(image_files, 1):
        full_path = os.path.join(root, name)
        image = cv2.imread(full_path)
        
        if image is not None:
            res = []
            has_result = run_preprocessing(image, res)
            
            if has_result:
                class_name = os.path.basename(root)
                if class_name not in class_dirs:
                    save_path = os.path.join(target_dir, class_name)
                    os.makedirs(save_path, exist_ok=True)
                    class_dirs[class_name] = save_path
                save_path = class_dirs[class_name]

                for item in res:
                    if item["name"] == "Result":
                        cv2.imwrite(os.path.join(save_path, name), item["data"])

        print_progress("  Segmentierung", idx, total_files)

    print("\nSegmentierung abgeschlossen.")

# ==========================================
# TEIL 2: QUALITÄTSKONTROLLE & SORTIERUNG
# ==========================================

def get_contours_hierarchy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def analyze_geometry_features(contours, hierarchy):
    stats = {
        "has_object": False, "area": 0, "solidity": 0,
        "num_windows": 0, "has_center_hole": False, "main_contour": None,
        "window_areas": []  # <--- NEU: Liste für die Flächengrößen
    }
    
    if not contours: return stats

    # Hauptobjekt finden
    main_cnt_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    main_cnt = contours[main_cnt_idx]
    area = cv2.contourArea(main_cnt)
    
    stats["has_object"] = True
    stats["area"] = area
    stats["main_contour"] = main_cnt

    # Löcher analysieren
    epsilon_factor = EPSILON_FACTOR
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            
            # Nur direkte Kinder des Hauptobjekts (Löcher)
            if parent_idx == main_cnt_idx:
                hole_area = cv2.contourArea(cnt)
                if hole_area < MIN_HOLE_AREA: continue 

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
                corners = len(approx)

                # Unterscheidung Fenster vs. Mittelloch
                if 3 <= corners <= 5 and hole_area > MIN_WINDOW_AREA:
                    stats["num_windows"] += 1
                    stats["window_areas"].append(hole_area) # <--- NEU: Fläche speichern
                elif corners > 5 and hole_area < MAX_CENTER_HOLE_AREA:
                    stats["has_center_hole"] = True
                    
    return stats

def detect_defects(image, spot_threshold=DEFECT_SPOT_THRESHOLD, debug=False):
    """
    Nutzt Morphological Black-Hat, um dunkle Flecken unabhängig von
    Schattenverläufen zu finden.
    """
    # 1. Konvertierung in Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Maske erstellen (Objekt vom schwarzen Hintergrund trennen)
    # Alles was nicht fast schwarz ist, ist das Objekt
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 3. Sicherheits-Erosion
    # WICHTIG: Wir müssen den Rand stark verkleinern.
    # Der Übergang vom hellen Snack zum schwarzen Hintergrund ist eine "riesige Kante",
    # die wir ignorieren müssen.
    kernel_erode = np.ones(EROSION_KERNEL_SIZE, np.uint8)
    mask_analysis = cv2.erode(mask_obj, kernel_erode, iterations=EROSION_ITERATIONS) 
    # Falls zu viel vom Objekt verschwindet: iterations verringern (z.B. 3)

    # 4. Black-Hat Transformation
    # Wir definieren eine Größe für die Flecken, die wir suchen.
    # Kernel-Größe (15, 15) bedeutet: "Suche Dinge, die kleiner sind als ca. 15 Pixel"
    # Alles was größer ist (wie Schattenverläufe), wird ignoriert.
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BLACKHAT_KERNEL_SIZE)
    blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_morph)

    # Das blackhat_img ist jetzt fast schwarz, nur die Flecken leuchten weiß hervor.
    # Je heller der Pixel im blackhat_img, desto stärker ist der lokale Kontrast (Fleck).

    # 5. Schwellenwert auf den Kontrast
    # Hier definieren wir: "Der Fleck muss mindestens 30 Helligkeitsstufen dunkler 
    # sein als seine direkte Umgebung".
    contrast_threshold = BLACKHAT_CONTRAST_THRESHOLD 
    _, mask_defects = cv2.threshold(blackhat_img, contrast_threshold, 255, cv2.THRESH_BINARY)

    # 6. Nur Defekte IM Objekt betrachten
    valid_defects = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    
    # (Optional) Rauschen entfernen (Punkte kleiner als 2px weg)
    valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, np.ones(NOISE_KERNEL_SIZE, np.uint8))

    # 7. Ergebnis
    spot_area = cv2.countNonZero(valid_defects)
    is_defective = spot_area > spot_threshold

    # DEBUG: Um zu sehen, was der Black-Hat sieht
    if debug:
        # Wir geben das maskierte Blackhat-Bild zurück, damit du die Flecken leuchten siehst
        debug_view = cv2.bitwise_and(blackhat_img, blackhat_img, mask=mask_analysis)
        return {"is_defective": is_defective, "spot_area": spot_area, "debug_image": debug_view}

    return {
        "is_defective": is_defective,
        "spot_area": spot_area
    }

def print_table(headers, rows, indent="  "):
    """Gibt eine einfache Tabelle mit fester Spaltenbreite aus."""
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header_line = indent + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    divider_line = indent + "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(divider_line)
    for row in rows:
        print(indent + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    print()

def print_progress(prefix, current, total, bar_length=30):
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = ratio * 100
    print(f"\r{prefix} [{bar}] {percent:5.1f}% ({current}/{total})", end="", flush=True)

def collect_image_files(source_dir):
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append((root, name))
    return image_files

def describe_priority_chain():
    """Ermittelt die Priorisierungsreihenfolge der Zielklassen."""
    class_priority = {}
    for label, prio in LABEL_PRIORITIES.items():
        cls = LABEL_CLASS_MAP.get(label, label.title())
        if cls not in class_priority or prio < class_priority[cls]:
            class_priority[cls] = prio
    if not class_priority:
        return ""
    ordered = [name for name, _ in sorted(class_priority.items(), key=lambda item: item[1])]
    return " > ".join(ordered)

def normalize_relative_path(path):
    """Bringt Pfadangaben in ein einheitliches Format."""
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    marker = "Data/Images/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")

def resolve_priority_label(raw_label):
    """Wählt bei Mehrfachlabels den wichtigsten Eintrag."""
    if not raw_label:
        return None
    candidates = [lbl.strip().lower() for lbl in raw_label.split(",") if lbl.strip()]
    if not candidates:
        return None
    candidates.sort(key=lambda lbl: LABEL_PRIORITIES.get(lbl, 100))
    return candidates[0]

def load_annotations(annotation_file):
    annotations = {}
    if not os.path.exists(annotation_file):
        print(f"\nHinweis: '{annotation_file}' nicht gefunden, Validierung übersprungen.")
        return annotations

    with open(annotation_file, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rel_path = normalize_relative_path(row.get("image", ""))
            if not rel_path:
                continue
            base_label = resolve_priority_label(row.get("label", ""))
            if not base_label:
                continue
            annotations[rel_path] = LABEL_CLASS_MAP.get(base_label, "Rest")

    return annotations

def copy_misclassified(pred_entry, expected_label, falsch_dir):
    """Speichert fehlerhafte Bilder mit zusätzlicher Labelinformation."""
    os.makedirs(falsch_dir, exist_ok=True)
    rel_name = normalize_relative_path(pred_entry.get("relative_path", ""))
    if not rel_name:
        rel_name = os.path.basename(pred_entry["source_path"])
    rel_name = rel_name.replace("/", "_")
    base, ext = os.path.splitext(rel_name)
    safe_expected = expected_label.replace(" ", "_")
    safe_pred = pred_entry["predicted"].replace(" ", "_")
    new_name = f"{base}_gt-{safe_expected}_pred-{safe_pred}{ext}"
    dest_path = os.path.join(falsch_dir, new_name)
    shutil.copy(pred_entry["source_path"], dest_path)

def validate_predictions(predictions, annotations, falsch_dir):
    """Vergleicht Sortierergebnis mit den Annotationen und erstellt einen Report."""
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
        rel_path = normalize_relative_path(pred.get("relative_path", ""))
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
    print("- Gesamtstatistik:")
    print()
    summary_headers = ["Statistik", "Wert"]
    summary_rows = [
        ["Bewertet", str(total)],
        ["Treffer", str(correct)],
        ["Genauigkeit %", f"{accuracy:.2f}"],
        ["Falsch zugeordnet", str(len(mismatches))],
    ]
    if skipped:
        summary_rows.append(["Ohne passende Annotation", str(skipped)])
    print_table(summary_headers, summary_rows)
    chain = describe_priority_chain()
    if chain:
        print(f"Priorisierung (höchste Priorität links): {chain}")
        print()

    if per_class:
        print("- Klassenübersicht:")
        headers = ["Klasse", "Erwartet", "Treffer", "Genauigkeit %"]
        rows = []
        for cls_name in sorted(per_class.keys()):
            stats = per_class[cls_name]
            acc_cls = (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0.0
            rows.append([cls_name, str(stats["total"]), str(stats["correct"]), f"{acc_cls:.2f}"])
        print_table(headers, rows)

    if mismatches:
        for pred, expected in mismatches:
            copy_misclassified(pred, expected, falsch_dir)

def sort_dataset_manual_rules(source_data_dir, sorted_data_dir):
    print("\nStarte Sortierung und Symmetrie-Berechnung...")
    CLASSES = ["Normal", "Bruch", "Farbfehler", "Rest"]
    
    if os.path.exists(sorted_data_dir):
        shutil.rmtree(sorted_data_dir)
    for cls in CLASSES:
        os.makedirs(os.path.join(sorted_data_dir, cls), exist_ok=True)

    stats_counter = {k: 0 for k in CLASSES}
    reason_counter = {k: {} for k in CLASSES}
    predictions = []

    for root, dirs, files in os.walk(source_data_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            if image is None: continue

            contours, hierarchy = get_contours_hierarchy(image)
            geo = analyze_geometry_features(contours, hierarchy)
            
            target_class = "Normal"
            reason = "OK"
            # Prefix für den Dateinamen (Standard leer)
            file_prefix = "" 

            total_holes = geo["num_windows"] + (1 if geo["has_center_hole"] else 0)

            if not geo["has_object"]:
                target_class = "Rest"
                reason = "Kein Objekt"
            elif total_holes < 7:
                target_class = "Bruch"
                reason = f"Zu wenig Löcher: {total_holes}"
            elif total_holes > 7:
                target_class = "Rest"
                reason = f"Zu viele Fragmente: {total_holes}"
            else:
                # --- Geometrie ist OK (7 Löcher), jetzt Farbcheck ---
                col_res = detect_defects(image, spot_threshold=DEFECT_SPOT_THRESHOLD)
                
                if col_res["is_defective"]:
                    target_class = "Farbfehler"
                    reason = f"Fleck: {col_res['spot_area']}px"
                else:
                    # --- ALLES OK -> SYMMETRIE BERECHNEN ---
                    # Wir berechnen, wie stark die Fensterflächen voneinander abweichen.
                    areas = geo["window_areas"]
                    
                    if len(areas) > 0:
                        mean_a = np.mean(areas)
                        std_a = np.std(areas) # Standardabweichung
                        
                        # Variationskoeffizient (Wie viel % weicht ein Fenster vom Durschnitt ab?)
                        # Kleiner ist besser.
                        cv = std_a / mean_a if mean_a > 0 else 0
                        
                        # Score invertieren: 100 = 0% Abweichung, 0 = 30%+ Abweichung
                        # Faktor 3.0 ist ein Skalierungsfaktor für Empfindlichkeit
                        symmetry_score = max(0, min(100, int(100 * (1 - (cv * SYMMETRY_SENSITIVITY)))))
                    else:
                        symmetry_score = 0

                    # Prefix erstellen: z.B. "098_" für sehr symmetrisch
                    file_prefix = f"{symmetry_score:03d}_"
                    reason = f"Symmetrie: {symmetry_score}/100"

            # Datei kopieren (mit Prefix nur bei Normal)
            new_filename = f"{file_prefix}{file}"
            dest_path = os.path.join(sorted_data_dir, target_class, new_filename)
            shutil.copy(img_path, dest_path)
            
            stats_counter[target_class] += 1
            reason_counter[target_class][reason] = reason_counter[target_class].get(reason, 0) + 1
            if VERBOSE_SORT_OUTPUT:
                rel_dest = os.path.relpath(dest_path, sorted_data_dir).replace("\\", "/")
                print(f"[{target_class}] {new_filename} | Grund: {reason} | Ziel: {rel_dest}")
            
            rel_path = normalize_relative_path(os.path.relpath(img_path, source_data_dir))
            predictions.append({
                "relative_path": rel_path,
                "predicted": target_class,
                "source_path": img_path,
                "destination_path": dest_path,
                "reason": reason,
                "original_name": file,
            })

    total_sorted = sum(stats_counter.values())
    print("\nSortierung abgeschlossen:\n")
    headers = ["Klasse", "Anzahl", "Anteil %", "Beschreibung", "Häufigster Grund"]
    rows = []
    for cls in CLASSES:
        amount = stats_counter[cls]
        share = (amount / total_sorted * 100) if total_sorted else 0
        desc = CLASS_DESCRIPTIONS.get(cls, "")
        reason_text = "-"
        if reason_counter[cls]:
            top_reason, top_count = max(reason_counter[cls].items(), key=lambda kv: kv[1])
            reason_text = f"{top_reason} ({top_count}x)"
        rows.append([cls, str(amount), f"{share:.1f}", desc, reason_text])
    print_table(headers, rows)

    return predictions

# ==========================================
# MAIN PROGRAMM
# ==========================================

# KORREKTUR: __name__ und __main__ (doppelte Unterstriche!)
if __name__ == '__main__':
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. VORVERARBEITUNG
    # Prüfen ob Rohdaten da sind
    if os.path.exists(RAW_DATA_DIR):
        # Vorverarbeitung starten
        prepare_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        
        # 2. SORTIERUNG
        if os.path.exists(PROCESSED_DATA_DIR):
            # KORREKTUR: Hier den richtigen Funktionsnamen aufrufen!
            predictions = sort_dataset_manual_rules(PROCESSED_DATA_DIR, SORTED_DATA_DIR)
            annotations = load_annotations(ANNOTATION_FILE)
            validate_predictions(predictions, annotations, FALSCH_DIR)

    else:
        print(f"Fehler: Quellordner '{RAW_DATA_DIR}' nicht gefunden! Bitte Ordner erstellen und Bilder hineinlegen.")
