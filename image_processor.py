"""
image_processor.py

Enthält alle Funktionen zur Bildverarbeitung:
- Vorverarbeitung (Greenscreen, Zuschneiden)
- Merkmalsextraktion und Klassifizierung
- Symmetrie-Berechnung
"""

import cv2
import numpy as np
import config # Importiert unsere Einstellungs-Datei

def preprocess_image(image_path):
    """
    Lädt ein Bild, entfernt den grünen Hintergrund, findet das Objekt,
    schneidet es aus und skaliert es auf eine einheitliche Größe.

    Returns:
        resized_img: Das ausgeschnittene Objekt (200x200)
        resized_mask: Die zugehörige Maske (200x200)
        main_contour: Die Kontur des Objekts (für Formanalyse)
    """
    # 1. Bild einlesen
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Fehler: Bild konnte nicht geladen werden: {image_path}")
        return None, None, None

    # 2. Greenscreening (HSV-Farbraum)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, config.HSV_LOWER_GREEN, config.HSV_UPPER_GREEN)
    
    # 3. Maske invertieren (wir wollen das Objekt, nicht den Hintergrund)
    obj_mask = cv2.bitwise_not(mask)

    # 4. Maske säubern (Rauschen entfernen)
    kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Größte Kontur (das Objekt) finden
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Nichts gefunden in diesem Bild
        return None, None, None
        
    main_contour = max(contours, key=cv2.contourArea)

    # 6. Objekt ausschneiden (Bounding Box)
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = obj_mask[y:y+h, x:x+w] # Auch die Maske zuschneiden

    # 7. Auf einheitliche Größe skalieren
    resized_img = cv2.resize(cropped_img, config.RESIZE_DIM, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(cropped_mask, config.RESIZE_DIM, interpolation=cv2.INTER_NEAREST)

    return resized_img, resized_mask, main_contour


def classify_image(resized_img, resized_mask, main_contour):
    """
    Analysiert das vorverarbeitete Bild und gibt eine Klasse zurück.
    Die Reihenfolge der Prüfungen ist wichtig!
    """
    
    # --- 1. Form-Merkmale berechnen (von der Original-Kontur) ---
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    
    if perimeter == 0:
        return "Rest" # Ungültige Form

    # Zirkularität (für "Rest")
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # Solidität (für "Bruch")
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return "Rest" # Ungültige Form
        
    solidity = float(area) / hull_area

    # --- 2. Farb-Merkmal berechnen (vom skalierten Bild) ---
    # Wir nutzen nur Pixel *innerhalb* der Objektmaske
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_img[:, :, 2] # Helligkeits-Kanal
    
    # Standardabweichung der Helligkeit
    mean_val, std_dev_val = cv2.meanStdDev(v_channel, mask=resized_mask)
    color_std_dev = std_dev_val[0][0]

    
    # --- 3. Klassifizierungs-Logik (Pipeline) ---
    
    # Prüfung 1: Ist es "Rest"? (d.h. nicht rund)
    if circularity < config.CIRCULARITY_THRESHOLD_REST:
        return "Rest"

    # Prüfung 2: Ist es "Bruch"? (d.h. hat Einbuchtungen)
    if solidity < config.SOLIDITY_THRESHOLD_BRUCH:
        return "Bruch"
        
    # Prüfung 3: Ist es "Farbfehler"? (d.h. ungleichmäßige Farbe)
    if color_std_dev > config.COLOR_STD_DEV_THRESHOLD:
        return "Farbfehler"

    # Prüfung 4: Wenn nichts davon zutrifft -> Normal
    return "Normal"


def get_symmetry_score(resized_img, resized_mask):
    """
    Berechnet ein Symmetrie-Maß für "Normal"-Bilder.
    Ein niedrigerer Score bedeutet höhere Symmetrie.
    """
    # 1. Bild um 180° rotieren
    (h, w) = resized_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated_img = cv2.warpAffine(resized_img, M, (w, h))
    
    # 2. Maske auch rotieren (um nur relevante Pixel zu vergleichen)
    rotated_mask = cv2.warpAffine(resized_mask, M, (w, h))
    
    # 3. Schnittmenge der Masken (wo beide Bilder Objektpixel haben)
    comparison_mask = cv2.bitwise_and(resized_mask, rotated_mask)

    # 4. Absoluten Unterschied berechnen
    diff = cv2.absdiff(resized_img, rotated_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # 1-Kanal-Bild

    # 5. Mittleren Unterschied *nur innerhalb der Schnittmenge* berechnen
    mean_diff = cv2.mean(diff_gray, mask=comparison_mask)[0]
    
    # Als Integer zurückgeben (schöner für Dateinamen)
    return int(mean_diff)