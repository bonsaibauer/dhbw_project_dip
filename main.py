import os
import numpy as np

from classification import run_pipeline
from validation import validate_predictions, load_annotations
from segmentation import process_directory

# ==========================================
# ANPASSBARE PARAMETER & VERZEICHNISSE
# ==========================================
# --- Pfad-Setup (prep_set, run_pipeline, validate_predictions, copy_mismatch, load_annotations) ---
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

# --- Geometrie-Features (extract_hierarchy, compute_geometry, run_pipeline) ---
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

# --- Farb-/Spotprüfung (detect_spots, run_pipeline) ---
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

# --- Kantenschaden & Symmetrie (run_pipeline) ---
EDGE_DMG = 1.05  # Verhältnis Hülle/Perimeter; höher = toleranter, niedriger = früher Bruchalarm; Range 1.0–1.5.
EDGE_SEG = 14  # Max. Kantensegmente; höher = erlaubt zackigere Formen, niedriger = streng; Range 8–20.
SYM_SEN = 3.0  # Faktor für Symmetriepenalty; höher = Symmetrie strenger, niedriger = lockerer; Range 1.5–4.5.

# Entscheidungsbaum (Detailfluss):
#   Level 0 – Feature-Sammlung (Zeilen 559–700, `run_pipeline`):
#       Erzeugt `geo` via `compute_geometry` (Zeilen 235–292) und Farbmetriken via `detect_spots` (Zeilen 294–371).
#       Setzt Resthinweise (`rest_hints`) über `FRAG_MIN`, `RHL_*`, `RWA_*`, `RWR_*`, `RMULT_SP` und berechnet Symmetrie (`SYM_SEN`).
#       Farbkanal greift nur bei Anomalien: `detect_spots` liefert `spot_area`, `texture_std`, `lab_std`, `dark_delta`; Schwellwerte `SPT_*`, `COL_*`, `TXT_STD`, `LAB_STD`, `DRK_*`.
#   Level 1 – Guards & Objekt-Existenz (Zeilen 702–709):
#       Falls `geo["has_object"]` False → Klasse „Rest“ („Kein Objekt“).
#       Setzt `total_holes = geo["num_windows"] + center`, Basis für Level 2.
#   Level 2A – Zu wenige Öffnungen (Zeilen 709–720):
#       Wenn `total_holes < 7`: Standard „Bruch“ (`reason` = Lochzahl), außer starke Farbe (`color_strength >= 2`) oder starker Resthinweis (`rest_strength >= 2`).
#   Level 2B – Zu viele Öffnungen (Zeilen 721–724):
#       Wenn `total_holes > 7`: direkt „Rest“ wegen Fragmentierung (`FRAG_MIN`, `RMULT_SP` spiegeln Ursache im `reason`).
#   Level 3 – Genau 7 Öffnungen (Zeilen 725–752):
#       (a) Rest-Hardhit: `rest_strength >= 2` → „Rest“ (Ursprung `RHL_*`, `RWA_*`, `RWR_*`, Mehrfachkonturen).
#       (b) Starke Farbe: `color_strength >= 2` → „Farbfehler“ (stammt aus `detect_spots`-Schwellen `COL_STR`, `COL_SPT`, `COL_LAB`).
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


PREPROCESSING_PARAMS = {
    "HSV_LO": HSV_LO,
    "HSV_HI": HSV_HI,
    "CNT_MINA": CNT_MINA,
    "WARP_SZ": WARP_SZ,
    "TGT_W": TGT_W,
    "TGT_H": TGT_H,
}

GEOMETRY_PARAMS = {
    "EPS_FACT": EPS_FACT,
    "HOLE_MIN": HOLE_MIN,
    "WIND_MIN": WIND_MIN,
    "CTR_MAXA": CTR_MAXA,
    "FRAG_MIN": FRAG_MIN,
}

SPOT_PARAMS = {
    "ERO_KN": ERO_KN,
    "ERO_ITER": ERO_ITER,
    "BKH_KN": BKH_KN,
    "BKH_CON": BKH_CON,
    "NOI_KN": NOI_KN,
    "SPT_MIN": SPT_MIN,
    "SPT_RAT": SPT_RAT,
    "FERO_ITR": FERO_ITR,
    "INER_ITR": INER_ITR,
    "INSP_RAT": INSP_RAT,
    "FSPT_RAT": FSPT_RAT,
    "SPT_FIN": SPT_FIN,
    "DRK_PCT": DRK_PCT,
}

CLASSIFIER_RULES = {
    "RWA_BASE": RWA_BASE,
    "RWA_STRG": RWA_STRG,
    "RWA_CMP": RWA_CMP,
    "RWA_LRG": RWA_LRG,
    "RHL_BASE": RHL_BASE,
    "RHL_STRG": RHL_STRG,
    "RWR_BASE": RWR_BASE,
    "RWR_STRG": RWR_STRG,
    "RMULT_SP": RMULT_SP,
    "COL_STR": COL_STR,
    "COL_SPT": COL_SPT,
    "TXT_STD": TXT_STD,
    "COL_SYM": COL_SYM,
    "COL_LAB": COL_LAB,
    "LAB_STD": LAB_STD,
    "DRK_DLT": DRK_DLT,
    "SYM_SEN": SYM_SEN,
    "EDGE_DMG": EDGE_DMG,
    "EDGE_SEG": EDGE_SEG,
    "BRK_SYM": BRK_SYM,
}

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(RAW_DIR):
        process_directory(RAW_DIR, PROC_DIR, PREPROCESSING_PARAMS)

        if os.path.exists(PROC_DIR):
            predictions = run_pipeline(
                PROC_DIR,
                SORT_DIR,
                GEOMETRY_PARAMS,
                SPOT_PARAMS,
                CLASSIFIER_RULES,
                SORT_LOG,
            )
            annotations = load_annotations(ANNO_FILE, LABEL_PRIORITIES, LABEL_CLASS_MAP)
            validate_predictions(
                predictions,
                annotations,
                FAIL_DIR,
                LABEL_PRIORITIES,
                LABEL_CLASS_MAP,
            )
    else:
        print(f"Fehler: Quellordner '{RAW_DIR}' nicht gefunden! Bitte Ordner erstellen und Bilder hineinlegen.")
