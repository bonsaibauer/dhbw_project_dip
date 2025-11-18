import cv2
import numpy as np
import os
import glob
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# --- KONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, 'output', 'processed')

# Ausgabe-Ordner definieren
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, 'output', 'results')
OK_FOLDER = os.path.join(RESULTS_FOLDER, 'OK')
DEFECT_FOLDER = os.path.join(RESULTS_FOLDER, 'DEFEKT')

# Ordner erstellen, falls sie nicht existieren
os.makedirs(OK_FOLDER, exist_ok=True)
os.makedirs(DEFECT_FOLDER, exist_ok=True)

# Schwellenwerte
OUTER_BREAK_SENSITIVITY = 0.75  # Wenn Radius < 75% des Median-Radius -> Bruch
INNER_AREA_TOLERANCE = 0.25     # Wenn Fläche > 25% vom Mittelwert abweicht -> Bruch

def analyze_snack_pellet(image_path):
    filename = os.path.basename(image_path)
    # Optional: Name des Unterordners in Dateinamen integrieren, um Überschreiben zu verhindern
    # parent_dir = os.path.basename(os.path.dirname(image_path))
    # save_name = f"{parent_dir}_{filename}"
    save_name = filename
    
    # 1. Bild laden
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Graustufen und Maske
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Ergebnis-Bild (Kopie zum Einzeichnen)
    output_img = img.copy()
    errors = []

    # ---------------------------------------------------------
    # PHASE 1: Äußere Brucherkennung
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if contours:
        outer_contour = max(contours, key=cv2.contourArea)
        
        # Schwerpunkt
        M = cv2.moments(outer_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        # Radius-Profil
        distances = []
        for point in outer_contour:
            px, py = point[0]
            dist = np.sqrt((px - cX)**2 + (py - cY)**2)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Glätten
        window_size = 15
        kernel = np.ones(window_size) / window_size
        distances_smooth = np.convolve(distances, kernel, mode='same')
        
        median_radius = np.median(distances)
        threshold_radius = median_radius * OUTER_BREAK_SENSITIVITY
        
        break_indices = np.where(distances_smooth < threshold_radius)[0]
        
        if len(break_indices) > 10:
            errors.append("AEUSSERER BRUCH")
            cv2.circle(output_img, (cX, cY), int(median_radius), (0, 255, 0), 1) 
            for i in break_indices:
                pt = outer_contour[i][0]
                cv2.circle(output_img, (pt[0], pt[1]), 2, (0, 0, 255), -1)

    # ---------------------------------------------------------
    # PHASE 2: Innere Brucherkennung
    # ---------------------------------------------------------
    contours_all, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hole_areas = []
    hole_contours = []
    
    if hierarchy is not None:
        for i, contour in enumerate(contours_all):
            # Index 3 ist Parent. Wenn != -1, ist es ein inneres Loch
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(contour)
                hole_areas.append(area)
                hole_contours.append(contour)

    # Sortieren nach Größe
    sorted_indices = np.argsort(hole_areas)[::-1]
    potential_windows = []
    
    for idx in sorted_indices:
        c_area = hole_areas[idx]
        c_cnt = hole_contours[idx]
        if c_area < 50: continue
        potential_windows.append({'area': c_area, 'cnt': c_cnt})

    # Auswertung
    if len(potential_windows) < 6:
        errors.append(f"INNERER FEHLER ({len(potential_windows)} Fenster)")
    elif len(potential_windows) >= 6:
        top_6_windows = potential_windows[:6]
        areas = [w['area'] for w in top_6_windows]
        mean_area = np.mean(areas)
        
        for win in top_6_windows:
            area = win['area']
            deviation = abs(area - mean_area)
            
            # Zeichnen
            cv2.drawContours(output_img, [win['cnt']], -1, (255, 0, 0), 1)
            
            if deviation > (mean_area * INNER_AREA_TOLERANCE):
                errors.append(f"ABWEICHUNG")
                cv2.drawContours(output_img, [win['cnt']], -1, (0, 0, 255), 2)

    # ---------------------------------------------------------
    # SPEICHERN & AUSGABE
    # ---------------------------------------------------------
    status = "OK" if not errors else "DEFEKT"
    
    # Text ins Bild schreiben
    color = (0, 255, 0) if status == "OK" else (0, 0, 255)
    cv2.putText(output_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Pfad bestimmen
    if status == "OK":
        target_path = os.path.join(OK_FOLDER, save_name)
    else:
        target_path = os.path.join(DEFECT_FOLDER, save_name)
        # Optional: Fehlermeldung ins Bild schreiben
        y_pos = 60
        for err in errors:
             cv2.putText(output_img, err, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
             y_pos += 20

    # Bild speichern
    cv2.imwrite(target_path, output_img)


# --- Main Loop ---
if __name__ == "__main__":
    print(f"Suche Bilder in: {INPUT_FOLDER} und Unterordnern...")
    print(f"Ergebnisse werden gespeichert in: {RESULTS_FOLDER}")

    types = ('**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.PNG', '**/*.JPG')
    files_grabbed = []
    for file_type in types:
        search_pattern = os.path.join(INPUT_FOLDER, file_type)
        files_grabbed.extend(glob.glob(search_pattern, recursive=True))
    
    files_grabbed = list(set(files_grabbed))
    
    if not files_grabbed:
        print("FEHLER: Keine Bilder gefunden.")
    else:
        print(f"{len(files_grabbed)} Bilder gefunden. Starte Analyse...")
        for file_path in files_grabbed:
            analyze_snack_pellet(file_path)