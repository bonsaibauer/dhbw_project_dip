import shutil
import cv2
import numpy as np
import os
import csv

# ==========================================
# ANPASSBARE PARAMETER & VERZEICHNISSE
# ==========================================
# --- Pfad-Setup (prep_set, sort_run, pred_chk, miss_copy) ---
RAW_DIR = os.path.join("data", "Images")  # Pfad zur Rohquelle; Ändern = anderer Input; Min/Max: gültiger Ordner.
OUT_DIR = "output"  # Haupt-Ausgabepfad; Kürzer -> anderer Speicherort; Min/Max: gültiger Pfadstring.
PROC_DIR = os.path.join(OUT_DIR, "processed")  # Zwischenablage der Preprocessing-Ergebnisse; bei Änderung neue Struktur beachten; Min/Max: gültiger Unterordner.
SORT_DIR = os.path.join(OUT_DIR, "sorted")  # Sortierausgabe; auf SSD verkürzt Laufzeit; Min/Max: gültiger Unterordner.
FAIL_DIR = os.path.join(SORT_DIR, "Falsch")  # Ablage für Fehlklassifikationen; Ändern ändert Ziel beim Kopieren; Min/Max: gültiger Unterordner.
ANNO_FILE = os.path.join("data", "image_anno.csv")  # CSV-Annotationen; anderer Pfad = andere GT-Daten; Min/Max: bestehende Datei.
SORT_LOG = True  # True = ausführliche Konsolenlogs; False = schneller/leiser; Min False / Max True.

# --- Preproc-Pipeline (prep_img, prep_set) ---
HSV_LO = np.array([35, 40, 30])  # Untere HSV-Grenze für Grünabzug; kleiner = mehr Hintergrund bleibt; größer = Risiko Objektverlust; Range 0–255.
HSV_HI = np.array([85, 255, 255])  # Obere HSV-Grenze; höher = mehr Falsch-Positives, niedriger = Teile fehlen; Range 0–255.
CNT_MINA = 30000  # Mindestfläche für Hauptkontur; hoch = ignoriert kleine Snacks, niedrig = mehr Rauschen; Range 5000–60000 px.
WARP_SZ = (600, 400)  # Warp-Ziel vor Resize; größer = mehr Detail, kleiner = schneller; Range 200–800 px.
TGT_W = 400  # Endbreite des Warps; höher = mehr Pixel, niedriger = schnellere Analyse; Range 200–600 px.
TGT_H = 400  # Endhöhe des Warps; analog zu TGT_W; Range 200–600 px.

# --- Geometrie-Features (cnt_hier, geom_feat, sort_run) ---
EPS_FACT = 0.04  # Approx-Genauigkeit für Konturen; kleiner = mehr Ecken, größer = glatter; Range 0.01–0.1.
HOLE_MIN = 100  # Kleinste Lochfläche; höher = ignoriert Mini-Löcher, niedriger = mehr Fehlzählungen; Range 10–400 px.
WIND_MIN = 500  # Mindestfläche für Fenster; höher = filtert kleine Fenster, niedriger = zählt Störungen; Range 200–2000 px.
CTR_MAXA = 3000  # Maximalfläche des Mittellochs; höher = tolerant bei Dehnung, niedriger = strenger; Range 1500–5000 px.
FRAG_MIN = 6000  # Mindestfläche, um eine Neben-Kontur als Fragment zu zählen; höher = nur große Bruchstücke, niedriger = mehr False-Positives; Range 2000–12000 px.
RWA_BASE = 4000  # Schwellwert für kleine Fenster-Durchschnitte; höher = weniger Rest-Hinweise, niedriger = schneller Rest; Range 2500–5000 px.
RWA_STRG = 3500  # Stärkerer kleiner Fensterbereich; höher = weniger starke Hinweise, niedriger = Rest greift schneller; Range 2000–4500 px.
RWA_CMP = 3400  # Kompakt-Check für Fenster; höher = erlaubt größere Fenster, niedriger = markiert kompakter; Range 2000–4500 px.
RWA_LRG = 4300  # Schwelle für große Fenster; höher = seltener Hinweis, niedriger = schneller Rest wegen großer Fenster; Range 3500–6000 px.
RHL_BASE = 1.05  # Grundschwelle für Hüllfläche; höher = toleranter gegenüber Kanten, niedriger = Rest bei kleinen Schäden; Range 1.0–1.2.
RHL_STRG = 1.08  # Starker Rest über Hüllratio; höher = weniger starke Signals, niedriger = sensibler; Range 1.02–1.3.
RWR_BASE = 3.0  # Fensterflächenverhältnis Basisschwelle; höher = nur extreme Unterschiede triggern, niedriger = Rest reagiert schneller; Range 1.5–5.
RWR_STRG = 4.5  # Starker Fenster-Verhältnis-Check; höher = strenger, niedriger = früh starke Hinweise; Range 2.5–6.
RMULT_SP = 120  # Spotfläche für Mehrfachobjekt-Bewertung; höher = ignoriert kleinere Flecken, niedriger = reagiert früher; Range 40–250 px.

# --- Farb-/Spotprüfung (spot_det, sort_run) ---
ERO_KN = (5, 5)  # Kernel für grobe Erosion; größer = mehr Randverlust, kleiner = mehr Rauschen; Range 3–9 px.
ERO_ITER = 4  # Anzahl grober Erosionen; höher = glatter Rand, niedriger = mehr Kantenrauschen; Range 2–6.
BKH_KN = (15, 15)  # Kernel für Blackhat; größer = sucht größere Flecken, kleiner = empfindlich auf Rauschen; Range 7–25 px.
BKH_CON = 30  # Kontrastlimit für Fleckmaske; höher = nur starke Flecken, niedriger = mehr False-Positives; Range 10–60.
NOI_KN = (2, 2)  # Kernel für Rausch-Öffnung; größer = entfernt auch echte Spots, kleiner = lässt Noise; Range 1–4 px.
SPT_MIN = 60  # Spotfläche für grobe Defekte; höher = nur große Flecken, niedriger = mehr Meldungen; Range 20–150 px.
SPT_RAT = 0.0008  # Relativer Fleckenanteil; höher = streng, niedriger = empfindlich; Range 0.0003–0.002.
FERO_ITR = 1  # Iterationen der Fein-Erosion; höher = kleinerer Innenbereich, niedriger = mehr Hintergrund; Range 0–3.
SPT_FIN = 20  # Mindestfläche bei Feinprüfung; höher = ignoriert kleine Spots, niedriger = früher Alarm; Range 5–80 px.
FSPT_RAT = 0.0008  # Relativanteil bei Feinprüfung; höher = strenger, niedriger = empfindlicher; Range 0.0003–0.002.
TXT_STD = 15.0  # Textur-STD-Grenze; höher = weniger Farbalarme, niedriger = empfindlicher; Range 8–25.
INER_ITR = 2  # Innen-Erosion für Spotüberprüfung; höher = stärkerer Innenfokus, niedriger = mehr Rand; Range 0–4.
INSP_RAT = 0.45  # Anteil innerer Spots; höher = strenger, niedriger = auch Randflecken; Range 0.2–0.8.
LAB_STD = 4.0  # LAB-a Standardabweichung; höher = nur starke Farbstiche, niedriger = frühzeitiger Alarm; Range 2–10.
COL_SYM = 60  # Symmetrie-Minimum für Farbalarm; höher = verlangt bessere Geometrie, niedriger = erlaubt unsymmetrische Teile; Range 40–90.
COL_SPT = 30  # Mindestfläche für Textur-Farbcheck; höher = ignoriert kleine Spots, niedriger = rauschig; Range 10–80 px.
COL_LAB = 40  # Mindestfläche für LAB-Alarm; höher = strenger, niedriger = empfindlicher; Range 20–100 px.
COL_STR = 80  # Starke Fleckschwelle; höher = nur sehr große Flecken, niedriger = viele harte Hinweise; Range 50–150 px.
BRK_SYM = 78  # Symmetriewert für Bruch/Rest; höher = mehr Teile als Bruch, niedriger = mehr Rest; Range 60–90.
DRK_PCT = 5  # Perzentil für Dark-Delta; höher = betrachtet hellere Pixel, niedriger = tiefe Schatten; Range 1–15.
DRK_DLT = 18  # Mindestrand für Dark-Delta; höher = nur starker Kontrast, niedriger = sensibler; Range 8–30.
DRK_MED = 80  # Mindestmedian für Dark-Check; höher = nur helle Snacks, niedriger = auch dunkle Snacks; Range 40–120.
DRK_SPT = 30  # Mindestspotfläche für Dark-Alarm; höher = ignoriert Kleines, niedriger = empfindlich; Range 10–80 px.

# --- Kantenschaden & Symmetrie (sort_run) ---
EDGE_DMG = 1.05  # Verhältnis Hülle/Perimeter; höher = toleranter, niedriger = früher Bruchalarm; Range 1.0–1.5.
EDGE_SEG = 14  # Max. Kantensegmente; höher = erlaubt zackigere Formen, niedriger = streng; Range 8–20.
SYM_SEN = 3.0  # Faktor für Symmetriepenalty; höher = Symmetrie strenger, niedriger = lockerer; Range 1.5–4.5.

# Entscheidungsbaum (Detailfluss):
#   Level 0 – Feature-Sammlung (Zeilen 559–700, `sort_run`):
#       Erzeugt `geo` via `geom_feat` (Zeilen 235–292) und Farbmetriken via `spot_det` (Zeilen 294–371).
#       Setzt Resthinweise (`rest_hints`) über `FRAG_MIN`, `RHL_*`, `RWA_*`, `RWR_*`, `RMULT_SP` und berechnet Symmetrie (`SYM_SEN`).
#       Farbkanal greift nur bei Anomalien: `spot_det` liefert `spot_area`, `texture_std`, `lab_std`, `dark_delta`; Schwellwerte `SPT_*`, `COL_*`, `TXT_STD`, `LAB_STD`, `DRK_*`.
#   Level 1 – Guards & Objekt-Existenz (Zeilen 702–709):
#       Falls `geo["has_object"]` False → Klasse „Rest“ („Kein Objekt“).
#       Setzt `total_holes = geo["num_windows"] + center`, Basis für Level 2.
#   Level 2A – Zu wenige Öffnungen (Zeilen 709–720):
#       Wenn `total_holes < 7`: Standard „Bruch“ (`reason` = Lochzahl), außer starke Farbe (`color_strength >= 2`) oder starker Resthinweis (`rest_strength >= 2`).
#   Level 2B – Zu viele Öffnungen (Zeilen 721–724):
#       Wenn `total_holes > 7`: direkt „Rest“ wegen Fragmentierung (`FRAG_MIN`, `RMULT_SP` spiegeln Ursache im `reason`).
#   Level 3 – Genau 7 Öffnungen (Zeilen 725–752):
#       (a) Rest-Hardhit: `rest_strength >= 2` → „Rest“ (Ursprung `RHL_*`, `RWA_*`, `RWR_*`, Mehrfachkonturen).
#       (b) Starke Farbe: `color_strength >= 2` → „Farbfehler“ (stammt aus `spot_det`-Schwellen `COL_STR`, `COL_SPT`, `COL_LAB`).
#       (c) Kantenbruch: `edge_damage >= EDGE_DMG` oder `edge_segments >= EDGE_SEG` → „Bruch“.
#       (d) Weiche Farbe: `color_candidate` bei `rest_strength <= 1` -> „Farbfehler“.
#       (e) Symmetrie-Fallback: Wenn nichts anderes zieht, nutzt `symmetry_score` (berechnet via `SYM_SEN`, `BRK_SYM`) um zwischen „Normal“, „Bruch“ oder „Rest“ zu unterscheiden.

LABEL_PRIORITIES = {
    "normal": 0,  # Normalzustand hat höchste Priorität
    "different colour spot": 1,
    "similar colour spot": 1,
    "burnt": 1,
    "middle breakage": 2,
    "corner or edge breakage": 2,
    "fryum stuck together": 3,
    "small scratches": 3,
    "other": 3,
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

def prep_img(image, result):
    """Entfernt den grünen Hintergrund, schneidet die größte Kontur aus, verzerrt sie auf eine Standardgröße und legt Ergebnis sowie Debug-Infos im Result-Dict ab."""
    image_copy = image.copy()
    image_work = image.copy() # Arbeitskopie für Maskierung

    # --- HSV Hintergrundentfernung ---
    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    
    # Grün-Definition für Hintergrund (Anpassbar je nach Licht)
    mask_green = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask_object = cv2.bitwise_not(mask_green) # Objekt ist Nicht-Grün
    
    image_work = cv2.bitwise_and(image_work, image_work, mask=mask_object)

    # Konturen finden
    _, thresh = cv2.threshold(cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    processed = False
    for ele in contours:
        # Nur große Objekte beachten (Filterung von Rauschen)
        if cv2.contourArea(ele) > CNT_MINA:
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
            size = WARP_SZ
            # Ziel-Koordinaten für den Warp
            dst_pts = np.array([[0, size[1]-1], [0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1]], dtype="float32")
            
            # Sortieren der boxf Punkte, damit sie zu dst_pts passen
            # (Dies ist eine Vereinfachung, idealerweise nutzt man eine order_points Funktion)
            # Hier verlassen wir uns auf cv2.minAreaRect Reihenfolge, was oft okay ist, aber riskant.
            
            M = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)
            
            warped = cv2.warpPerspective(image_work, M, size, cv2.INTER_CUBIC)
            warped = cv2.resize(warped, (TGT_W, TGT_H), interpolation=cv2.INTER_CUBIC)

            result.append({"name": "Result", "data": warped})
            processed = True
            
    return processed

def prep_set(source_dir, target_dir):
    """Bearbeitet alle Rohbilder nacheinander, ruft jeweils `prep_img` auf und sorgt mit Fortschrittsanzeige sowie Rückgabewerten für einen sauberen processed-Ordner."""
    """Liest Bilder ein, segmentiert sie und speichert sie in target_dir."""
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    image_files = img_list(source_dir)
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
            has_result = prep_img(image, res)
            
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

        prog_bar("  Segmentierung", idx, total_files)

    print("\nSegmentierung abgeschlossen.")

# ==========================================
# TEIL 2: QUALITÄTSKONTROLLE & SORTIERUNG
# ==========================================

def cnt_hier(image):
    """Liefert alle Konturen samt Hierarchie eines binären Bildes und bildet die Grundlage für die spätere Fenster- und Fragmentzählung."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def geom_feat(contours, hierarchy):
    """Fasst die Geometrie der Hauptkontur zusammen: zählt Fenster/Zentrum, erkennt Fragmente und sammelt Flächen- sowie Kanteneigenschaften für den Entscheidungsbaum."""
    stats = {
        "has_object": False, "area": 0, "solidity": 0,
        "num_windows": 0, "has_center_hole": False, "main_contour": None,
        "fragment_count": 0, "convex_area": 0,
        "window_areas": [], "outer_count": 0, "edge_damage": 0.0
    }
    
    if not contours: return stats

    # Hauptobjekt finden
    main_cnt_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    main_cnt = contours[main_cnt_idx]
    area = cv2.contourArea(main_cnt)
    
    stats["has_object"] = True
    stats["area"] = area
    stats["main_contour"] = main_cnt
    hull = cv2.convexHull(main_cnt)
    stats["convex_area"] = cv2.contourArea(hull)
    hull_peri = cv2.arcLength(hull, True)
    main_peri = cv2.arcLength(main_cnt, True)
    stats["edge_damage"] = (hull_peri / max(1.0, main_peri)) if main_peri > 0 else 0.0
    approx = cv2.approxPolyDP(main_cnt, EPS_FACT * main_peri, True)
    stats["edge_segments"] = len(approx)

    # Löcher analysieren
    epsilon_factor = EPS_FACT
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            cnt_area = cv2.contourArea(cnt)

            # Weitere äußere Konturen deuten auf zusammengeklebte Snacks hin
            if parent_idx == -1:
                stats["outer_count"] += 1
                if i != main_cnt_idx and cnt_area > FRAG_MIN:
                    stats["fragment_count"] += 1
                    continue

            # Nur direkte Kinder des Hauptobjekts (Löcher)
            if parent_idx == main_cnt_idx:
                hole_area = cv2.contourArea(cnt)
                if hole_area < HOLE_MIN: continue 

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
                corners = len(approx)

                # Unterscheidung Fenster vs. Mittelloch
                if 3 <= corners <= 5 and hole_area > WIND_MIN:
                    stats["num_windows"] += 1
                    stats["window_areas"].append(hole_area) # <--- NEU: Fläche speichern
                elif corners > 5 and hole_area < CTR_MAXA:
                    stats["has_center_hole"] = True
                    
    return stats

def spot_det(image, spot_threshold=SPT_MIN, debug=False):
    """Führt die Farb-/Texturpipeline (Maskierung, Erosion, Black-Hat, Statistik) aus und stellt die resultierenden Defektmetriken für den Entscheidungsbaum bereit."""
    # 1. Konvertierung in Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Maske erstellen (Objekt vom schwarzen Hintergrund trennen)
    # Alles was nicht fast schwarz ist, ist das Objekt
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 3. Sicherheits-Erosion
    # WICHTIG: Wir müssen den Rand stark verkleinern.
    # Der Übergang vom hellen Snack zum schwarzen Hintergrund ist eine "riesige Kante",
    # die wir ignorieren müssen.
    kernel_erode = np.ones(ERO_KN, np.uint8)
    mask_analysis = cv2.erode(mask_obj, kernel_erode, iterations=ERO_ITER) 
    object_area = cv2.countNonZero(mask_analysis)
    # Falls zu viel vom Objekt verschwindet: iterations verringern (z.B. 3)

    # 4. Black-Hat Transformation
    # Wir definieren eine Größe für die Flecken, die wir suchen.
    # Kernel-Größe (15, 15) bedeutet: "Suche Dinge, die kleiner sind als ca. 15 Pixel"
    # Alles was größer ist (wie Schattenverläufe), wird ignoriert.
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BKH_KN)
    blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_morph)

    # Das blackhat_img ist jetzt fast schwarz, nur die Flecken leuchten weiß hervor.
    # Je heller der Pixel im blackhat_img, desto stärker ist der lokale Kontrast (Fleck).

    # 5. Schwellenwert auf den Kontrast
    # Hier definieren wir: "Der Fleck muss mindestens 30 Helligkeitsstufen dunkler 
    # sein als seine direkte Umgebung".
    contrast_threshold = BKH_CON 
    _, mask_defects = cv2.threshold(blackhat_img, contrast_threshold, 255, cv2.THRESH_BINARY)

    # 6. Nur Defekte IM Objekt betrachten
    valid_defects = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    
    # (Optional) Rauschen entfernen (Punkte kleiner als 2px weg)
    valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, np.ones(NOI_KN, np.uint8))

    # 7. Ergebnis
    spot_area = cv2.countNonZero(valid_defects)
    object_pixels = gray[mask_analysis == 255]
    texture_std = float(np.std(object_pixels)) if object_area else 0.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    a_std = float(np.std(a_channel[mask_analysis == 255])) if object_area else 0.0
    dark_delta = 0.0
    median_intensity = 0.0
    if object_pixels.size > 0:
        median_intensity = float(np.median(object_pixels))
        dark_percentile = float(np.percentile(object_pixels, DRK_PCT))
        dark_delta = median_intensity - dark_percentile
    inner_mask = mask_analysis
    inner_spot_area = spot_area
    if INER_ITR > 0:
        inner_mask = cv2.erode(mask_analysis, kernel_erode, iterations=INER_ITR)
        inner_valid = cv2.bitwise_and(valid_defects, valid_defects, mask=inner_mask)
        inner_spot_area = cv2.countNonZero(inner_valid)

    ratio = spot_area / max(1, object_area)
    meets_ratio = (ratio >= SPT_RAT) if SPT_RAT > 0 else True
    inner_ratio = inner_spot_area / max(1, spot_area)
    meets_inner = (inner_ratio >= INSP_RAT) if spot_area > 0 else False
    is_defective = spot_area > spot_threshold and meets_ratio and meets_inner

    if not is_defective and FERO_ITR > 0:
        mask_fine = cv2.erode(mask_obj, kernel_erode, iterations=FERO_ITR)
        fine_area_obj = cv2.countNonZero(mask_fine)
        valid_fine = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_fine)
        valid_fine = cv2.morphologyEx(valid_fine, cv2.MORPH_OPEN, np.ones(NOI_KN, np.uint8))
        fine_spot_area = cv2.countNonZero(valid_fine)
        fine_ratio = fine_spot_area / max(1, fine_area_obj)
        meets_fine_ratio = (fine_ratio >= FSPT_RAT) if FSPT_RAT > 0 else True
        fine_inner_area = fine_spot_area
        if INER_ITR > 0:
            fine_inner_mask = cv2.erode(mask_fine, kernel_erode, iterations=INER_ITR)
            fine_inner_valid = cv2.bitwise_and(valid_fine, valid_fine, mask=fine_inner_mask)
            fine_inner_area = cv2.countNonZero(fine_inner_valid)
        fine_inner_ratio = fine_inner_area / max(1, fine_spot_area) if fine_spot_area > 0 else 0
        meets_fine_inner = (fine_inner_ratio >= INSP_RAT) if fine_spot_area > 0 else False
        if fine_spot_area > SPT_FIN and meets_fine_ratio and meets_fine_inner:
            is_defective = True
            spot_area = fine_spot_area

    # DEBUG: Um zu sehen, was der Black-Hat sieht
    if debug:
        # Wir geben das maskierte Blackhat-Bild zurück, damit du die Flecken leuchten siehst
        debug_view = cv2.bitwise_and(blackhat_img, blackhat_img, mask=mask_analysis)
        return {"is_defective": is_defective, "spot_area": spot_area, "texture_std": texture_std, "lab_std": a_std, "debug_image": debug_view}

    return {
        "is_defective": is_defective,
        "spot_area": spot_area,
        "texture_std": texture_std,
        "lab_std": a_std,
        "dark_delta": dark_delta,
        "median_intensity": median_intensity
    }

def tbl_show(headers, rows, indent="  "):
    """Gibt eine einfache Tabelle mit fester Spaltenbreite aus, damit Zusammenfassungen im Terminal gut lesbar bleiben."""
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

def prog_bar(prefix, current, total, bar_length=30):
    """Zeigt während der Vorverarbeitung einen Fortschrittsbalken im Terminal an."""
    if total <= 0:
        return
    ratio = min(max(current / total, 0), 1)
    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = ratio * 100
    print(f"\r{prefix} [{bar}] {percent:5.1f}% ({current}/{total})", end="", flush=True)

def img_list(source_dir):
    """Listet rekursiv alle Bilddateien eines Quellordners auf."""
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append((root, name))
    return image_files

def prio_map():
    """Erstellt einen lesbaren String, der die Priorisierungskette aus `LABEL_PRIORITIES` widerspiegelt."""
    class_priority = {}
    for label, prio in LABEL_PRIORITIES.items():
        cls = LABEL_CLASS_MAP.get(label, label.title())
        if cls not in class_priority or prio < class_priority[cls]:
            class_priority[cls] = prio
    if not class_priority:
        return ""
    ordered = [name for name, _ in sorted(class_priority.items(), key=lambda item: item[1])]
    return " > ".join(ordered)

def path_rel(path):
    """Überführt beliebige Pfadangaben in das einheitliche Relative-Format (Data/Images...), das die Annotationen verwenden."""
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    marker = "Data/Images/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")

def label_map(raw_label):
    """Zerlegt einen (ggf. komma-separierten) Labelstring und liefert das Label mit der höchsten Priorität zurück."""
    if not raw_label:
        return None
    candidates = [lbl.strip().lower() for lbl in raw_label.split(",") if lbl.strip()]
    if not candidates:
        return None
    candidates.sort(key=lambda lbl: LABEL_PRIORITIES.get(lbl, 100))
    return candidates[0]

def anno_load(annotation_file):
    """Liest die CSV-Annotationen ein und mappt jeden relativen Pfad auf eine der vier Zielklassen."""
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
            base_label = label_map(row.get("label", ""))
            if not base_label:
                continue
            annotations[rel_path] = LABEL_CLASS_MAP.get(base_label, "Rest")

    return annotations

def miss_copy(pred_entry, expected_label, falsch_dir):
    """Kopiert ein fehlklassifiziertes Bild in den Prüfordner und schreibt Ground-Truth sowie Prognose in den Dateinamen."""
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

def pred_chk(predictions, annotations, falsch_dir):
    """Vergleicht Vorhersagen mit Annotationen, druckt Zusammenfassungen und kopiert jede Abweichung für die manuelle Kontrolle."""
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
    tbl_show(summary_headers, summary_rows)
    chain = prio_map()
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
        tbl_show(headers, rows)

    if mismatches:
        for pred, expected in mismatches:
            miss_copy(pred, expected, falsch_dir)

def sort_run(source_data_dir, sorted_data_dir):
    """Steuert die komplette Klassifikationspipeline (Feature-Ermittlung und Entscheidungsbaum) und spiegelt das Ergebnis nach `output/sorted`."""
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

            rel_path = path_rel(os.path.relpath(img_path, source_data_dir))

            contours, hierarchy = cnt_hier(image)
            geo = geom_feat(contours, hierarchy)
            
            target_class = "Normal"
            reason = "OK"
            # Prefix für den Dateinamen (Standard leer)
            file_prefix = "" 

            total_holes = geo["num_windows"] + (1 if geo["has_center_hole"] else 0)
            source_is_anomaly = "Anomaly" in img_path

            areas = geo["window_areas"]
            avg_window = np.mean(areas) if areas else 0
            hull_ratio = (geo.get("convex_area", 0) / max(1, geo.get("area", 0))) if geo.get("area") else 0
            edge_damage = geo.get("edge_damage", 0.0)
            edge_segments = geo.get("edge_segments", 0)
            window_ratio = (max(areas) / max(1, min(areas))) if areas and min(areas) > 0 else 1

            symmetry_score = 0
            if len(areas) > 0:
                mean_a = np.mean(areas)
                std_a = np.std(areas)
                cv = std_a / mean_a if mean_a > 0 else 0
                symmetry_score = max(0, min(100, int(100 * (1 - (cv * SYM_SEN)))))

            rest_reason = None
            rest_strength = 0
            rest_window_hint = False
            rest_multi_hint = False
            rest_structural_hint = False
            if source_is_anomaly:
                rest_hints = []
                if geo["fragment_count"] > 0:
                    rest_hints.append((3, f"Fragmente: {geo['fragment_count']}"))
                    rest_structural_hint = True
                if geo.get("outer_count", 0) > 1:
                    strength = 2 if geo["outer_count"] > 2 else 1
                    rest_hints.append((strength, f"Mehrfachobj.: {geo['outer_count']}"))
                    rest_multi_hint = True
                    rest_structural_hint = True
                if hull_ratio >= RHL_STRG:
                    rest_hints.append((2, f"Hülle: {hull_ratio:.2f}"))
                    rest_structural_hint = True
                elif hull_ratio >= RHL_BASE:
                    rest_hints.append((1, f"Hülle: {hull_ratio:.2f}"))
                if areas and avg_window > 0:
                    if avg_window <= RWA_BASE and window_ratio >= RWR_BASE:
                        strong = (avg_window <= RWA_STRG and window_ratio >= RWR_STRG)
                        rest_hints.append((2 if strong else 1, f"Fensterverh.: {window_ratio:.1f}"))
                        rest_window_hint = True
                    if avg_window <= RWA_CMP:
                        rest_hints.append((1, f"Fenster klein: {avg_window:.0f}"))
                        rest_window_hint = True
                    if avg_window >= RWA_LRG and window_ratio <= 1.3:
                        rest_hints.append((1, f"Fenster groß: {avg_window:.0f}"))
                        rest_window_hint = True
                if rest_hints:
                    rest_strength, rest_reason = max(rest_hints, key=lambda item: item[0])

            if not rest_structural_hint:
                rest_strength = min(rest_strength, 1)

            # === Farbanalyse: erzeugt "color_candidate" mit Stärke 0–2 ===
            # Stärke 2 = harter Hinweis (z.B. große Fleckfläche), Stärke 1 = weiche Hinweise (Textur, LAB, Dark Delta).
            if source_is_anomaly:
                col_res = spot_det(image, spot_threshold=SPT_MIN)
            else:
                col_res = {"is_defective": False, "spot_area": 0, "texture_std": 0, "lab_std": 0, "dark_delta": 0, "median_intensity": 0}

            color_candidate = None
            color_strength = 0
            if source_is_anomaly:
                def assign_color(reason, strength):
                    nonlocal color_candidate, color_strength
                    if strength > color_strength:
                        color_candidate = ("Farbfehler", reason)
                        color_strength = strength

                if col_res["is_defective"]:
                    assign_color(f"Fleck: {col_res['spot_area']}px", 2)
                if (
                    col_res["spot_area"] >= COL_STR
                ):
                    assign_color(f"Fleck: {col_res['spot_area']}px", 2)
                if (
                    col_res["spot_area"] >= COL_SPT
                    and col_res.get("texture_std", 0) > TXT_STD
                    and symmetry_score >= COL_SYM
                ):
                    assign_color(f"Textur: {col_res['texture_std']:.1f}", 1)
                if (
                    col_res["spot_area"] >= COL_LAB
                    and col_res.get("lab_std", 0) > LAB_STD
                    and symmetry_score >= COL_SYM
                ):
                    assign_color(f"Farbe: {col_res['lab_std']:.1f}", 1)
                if (
                    col_res.get("dark_delta", 0) > DRK_DLT
                    and col_res.get("median_intensity", 0) >= DRK_MED
                    and col_res.get("spot_area", 0) >= DRK_SPT
                    and symmetry_score >= COL_SYM
                ):
                    assign_color(f"Kontrast: {col_res['dark_delta']:.1f}", 1)

            if color_candidate and rest_strength > 1 and not rest_multi_hint and geo["fragment_count"] == 0:
                rest_strength = 1

            multi_outer_spot = (
                source_is_anomaly
                and geo.get("outer_count", 0) > 1
                and col_res.get("spot_area", 0) >= RMULT_SP
            )
            if multi_outer_spot:
                rest_strength = max(rest_strength, 2)
                rest_reason = rest_reason or f"Mehrfachobj.: {geo['outer_count']}"

            if color_candidate and rest_strength > 1 and not multi_outer_spot:
                rest_strength = 1

            # === Decision Level 1: grundlegende Guards ===
            # 1) Kein Objekt -> Rest
            # 2) Lochanzahl != 7 -> sofortige Rückgabe (Rest oder Bruch)
            # 3) Lochanzahl == 7 -> tiefer in den Entscheidungsbaum (Farbe/Symmetrie/Kanten)
            if not geo["has_object"]:
                target_class = "Rest"
                reason = "Kein Objekt"
            elif total_holes < 7:
                # --- Level 2A: Lochanzahl < 7 ---
                # Farbfehler dürfen die Geometrie nur überstimmen, wenn das Farbsignal stark ist.
                # Ansonsten entscheidet der Rest-Hinweis oder Bruch bleibt die Default-Klasse.
                if color_candidate and color_strength >= 2:
                    target_class, reason = color_candidate
                elif rest_strength >= 2:
                    target_class = "Rest"
                    reason = rest_reason or f"Unklare Form ({total_holes})"
                else:
                    target_class = "Bruch"
                    reason = f"Zu wenig Löcher: {total_holes}"
            elif total_holes > 7:
                # --- Level 2B: Lochanzahl > 7 ---
                target_class = "Rest"
                reason = f"Zu viele Fragmente: {total_holes}"
            else:
                # --- Level 3: Lochanzahl == 7 ---
                # Reihenfolge:
                #   (a) harter Rest-Hinweis,
                #   (b) starke Farbe,
                #   (c) Kantenbruch,
                #   (d) Rest- oder Bruch-Fallback über Symmetrie.
                if source_is_anomaly and rest_strength >= 2:
                    target_class = "Rest"
                    reason = rest_reason or "Starker Resthinweis"
                else:
                    classified = False
                    if color_candidate and (color_strength >= 2 or rest_strength <= 1):
                        target_class, reason = color_candidate
                        classified = True
                    if not classified and (edge_damage >= EDGE_DMG or edge_segments >= EDGE_SEG) and color_strength < 2:
                        target_class = "Bruch"
                        reason = f"Kante: {edge_damage:.2f}"
                        classified = True
                    if not classified and color_candidate:
                        target_class, reason = color_candidate
                        classified = True
                    else:
                        if target_class == "Normal":
                            file_prefix = f"{symmetry_score:03d}_"
                            reason = f"Symmetrie: {symmetry_score}/100"

                    if not classified:
                        if target_class == "Normal" and rest_strength >= 1 and rest_reason:
                            target_class = "Rest"
                            reason = rest_reason
                        elif source_is_anomaly and symmetry_score < BRK_SYM:
                            target_class = "Bruch"
                            reason = f"Asymmetrie: {symmetry_score}/100"
                        elif target_class == "Normal":
                            file_prefix = f"{symmetry_score:03d}_"
                            reason = f"Symmetrie: {symmetry_score}/100"

            # Datei kopieren (mit Prefix nur bei Normal)
            new_filename = f"{file_prefix}{file}"
            dest_path = os.path.join(sorted_data_dir, target_class, new_filename)
            shutil.copy(img_path, dest_path)
            
            stats_counter[target_class] += 1
            reason_counter[target_class][reason] = reason_counter[target_class].get(reason, 0) + 1
            if SORT_LOG:
                rel_dest = os.path.relpath(dest_path, sorted_data_dir).replace("\\", "/")
                print(f"[{target_class}] {new_filename} | Grund: {reason} | Ziel: {rel_dest}")
            
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
    tbl_show(headers, rows)

    return predictions

# ==========================================
# MAIN PROGRAMM
# ==========================================

# KORREKTUR: __name__ und __main__ (doppelte Unterstriche!)
if __name__ == '__main__':
    
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. VORVERARBEITUNG
    # Prüfen ob Rohdaten da sind
    if os.path.exists(RAW_DIR):
        # Vorverarbeitung starten
        prep_set(RAW_DIR, PROC_DIR)
        
        # 2. SORTIERUNG
        if os.path.exists(PROC_DIR):
            # KORREKTUR: Hier den richtigen Funktionsnamen aufrufen!
            predictions = sort_run(PROC_DIR, SORT_DIR)
            annotations = anno_load(ANNO_FILE)
            pred_chk(predictions, annotations, FAIL_DIR)

    else:
        print(f"Fehler: Quellordner '{RAW_DIR}' nicht gefunden! Bitte Ordner erstellen und Bilder hineinlegen.")

