"""
config.py

Enthält alle Konfigurationsvariablen, Pfade und Schwellenwerte für das Projekt.
"""

import numpy as np
from pathlib import Path

# --- 1. Projekt-Pfade ---

# Basisverzeichnis des Projekts (der Ordner, in dem config.py liegt)
ROOT_DIR = Path(__file__).parent.resolve()

# Eingabe-Pfade
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "Images"
NORMAL_IMAGES_DIR = IMAGE_DIR / "Normal"
ANOMALY_IMAGES_DIR = IMAGE_DIR / "Anomaly"
ANNOTATION_FILE = DATA_DIR / "image_anno.csv" # Pfad zur CSV für Ground Truth

# Ausgabe-Pfade
OUTPUT_DIR = ROOT_DIR / "Ergebnisse"

# Liste der Klassen für die Ordnererstellung
CLASSES = ["Normal", "Farbfehler", "Bruch", "Rest", "Falsch"]


# --- 2. Vorverarbeitungs-Parameter ---

# Einheitliche Größe (Breite, Höhe) für alle ausgeschnittenen Bilder
RESIZE_DIM = (200, 200)

# HSV Farbbereich für den grünen Hintergrund ("Greenscreen")
# Diese Werte müssen evtl. experimentell angepasst werden.
HSV_LOWER_GREEN = np.array([35, 50, 50])
HSV_UPPER_GREEN = np.array([85, 255, 255])

# Kernel-Größe für morphologische Operationen (Masken-Säuberung)
MORPH_KERNEL_SIZE = (5, 5)


# --- 3. Klassifizierungs-Schwellenwerte ---
# HINWEIS: Diese Werte sind Schätzungen und müssen 
#          wahrscheinlich durch Tests optimiert werden!

# "Rest" (Zirkularität: 4*pi*Area / Perimeter^2)
# Ein perfekter Kreis ist 1.0. Alles unter 0.75 wird als "Rest" eingestuft.
CIRCULARITY_THRESHOLD_REST = 0.75 

# "Bruch" (Solidität: Area / ConvexHullArea)
# Ein Objekt ohne Einbuchtungen ist 1.0. 
# Alles unter 0.95 (d.h. >5% Abweichung) wird als "Bruch" gewertet.
SOLIDITY_THRESHOLD_BRUCH = 0.95 

# "Farbfehler" (Standardabweichung der Helligkeit im V-Kanal)
# Ein hoher Wert bedeutet starke Helligkeitsunterschiede (z.B. verbrannt).
COLOR_STD_DEV_THRESHOLD = 35