import cv2
import numpy as np
import os
import shutil

# ==========================================
# EINSTELLUNGEN
# ==========================================
# Quellordner definieren
SRC_NORMAL_DIR = os.path.join("output", "processed", "Normal")
SRC_ANOMALY_DIR = os.path.join("output", "processed", "Anomaly")

# Zielordner für die Rankings
RANKING_NORMAL_DIR = os.path.join("output", "Symmetrie-Ranking-Normal")
RANKING_ANOMALY_DIR = os.path.join("output", "Symmetrie-Ranking-Anomaly")


# ==========================================
# DIE SYMMETRIE-FUNKTION (DEINE METHODE)
# ==========================================
def calculate_asymmetry_score(img):
    """
    Testet die 6-fache Rotationssymmetrie, indem alle 6 Positionen
    (0, 60, 120, 180, 240, 300 Grad) verglichen werden.
    
    Gibt die Gesamtfläche (Anzahl der Pixel) zurück, die NICHT
    in allen 6 Rotationen vorhanden ist ("rote Pixel").
    """
    if img is None:
        return -1 # Fehler beim Laden

    # In Graustufen umwandeln und Maske erstellen (Objekt vs. Hintergrund)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 1. Mittelpunkt (Centroid) finden
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return -1 # Kein Objekt gefunden
        
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    (h, w) = img.shape[:2]

    # Wir starten mit der Original-Maske (0 Grad) als "Kern"
    core_mask = mask.copy()
    
    # 2. Schleife durch die 5 Rotationen (60° bis 300°)
    for i in range(1, 6): # 1, 2, 3, 4, 5
        angle = i * 60
        R = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, R, (w, h))
        
        # Den "Kern" mit jeder Runde kleiner schrumpfen (Schnittmenge)
        core_mask = cv2.bitwise_and(core_mask, rotated_mask)

    # 3. Asymmetrie-Teile finden
    # 'asymmetric_parts_mask' sind alle Pixel, die nicht zum "Kern" gehören
    asymmetric_parts_mask = cv2.subtract(mask, core_mask)
    
    # 4. Score berechnen und zurückgeben
    # Der Score ist die Anzahl der Pixel in der asymmetrischen Maske
    score = cv2.countNonZero(asymmetric_parts_mask)
    
    return score

# ==========================================
# HAUPTSKRIPT: BATCH-VERARBEITUNG
# ==========================================

def process_and_rank_folder(source_dir, target_dir):
    """
    Eine Helferfunktion, die alle Bilder aus einem Quellordner
    analysiert, sortiert und in einen Zielordner kopiert.
    """
    print(f"\nStarte Ranking für: {source_dir}")
    print(f"Ziel: {target_dir}")

    # Zielordner vorbereiten
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    image_files = []
    try:
        # Alle Bilddateien finden
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(filename)
    except FileNotFoundError:
        print(f"--- FEHLER: Quellordner '{source_dir}' nicht gefunden! ---")
        return

    if not image_files:
        print("Keine Bilder im Quellordner gefunden.")
        return

    print(f"Analysiere {len(image_files)} Bilder...")
    
    results = [] # Hier speichern wir (score, full_path, filename)
    
    # 1. Alle Bilder analysieren und Scores berechnen
    for idx, filename in enumerate(image_files, 1):
        full_path = os.path.join(source_dir, filename)
        img = cv2.imread(full_path)
        
        if img is None:
            print(f"Konnte Bild nicht laden: {filename}")
            continue
            
        score = calculate_asymmetry_score(img)
        
        if score >= 0:
            results.append((score, full_path, filename))
        
        # Fortschrittsanzeige
        if idx % 25 == 0 or idx == len(image_files):
            print(f"  ...verarbeitet: {idx}/{len(image_files)} (Bild: {filename}, Score: {score})")

    # 2. Nach Score sortieren (der niedrigste Score = beste Symmetrie)
    results.sort(key=lambda x: x[0])
    
    print(f"\nRanking für '{source_dir}' abgeschlossen. Kopiere sortierte Dateien...")

    # 3. Sortierte Dateien in neuen Ordner kopieren
    for rank, (score, full_path, filename) in enumerate(results, 1):
        
        # Neuen Dateinamen erstellen: 001_Score-00500_...
        new_filename = f"{rank:03d}_Score-{score:05d}_{filename}"
        dest_path = os.path.join(target_dir, new_filename)
        
        shutil.copy(full_path, dest_path)

    print(f"Fertig! {len(results)} Bilder wurden nach '{target_dir}' kopiert.")


# --- Skript starten ---
if __name__ == '__main__':
    # Führe das Ranking für beide Ordner nacheinander aus
    process_and_rank_folder(SRC_NORMAL_DIR, RANKING_NORMAL_DIR)
    process_and_rank_folder(SRC_ANOMALY_DIR, RANKING_ANOMALY_DIR)
    
    print("\nAlle Operationen abgeschlossen.")