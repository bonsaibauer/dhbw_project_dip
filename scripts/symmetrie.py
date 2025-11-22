import cv2
import numpy as np
import os

def get_symmetry_score(image_bgr):
    """
    Berechnet einen skaleninvarianten Score für 6-fache Rotationssymmetrie.
    
    Args:
        image_bgr: Das ausgeschnittene, farbige Bild (BGR).
        
    Returns:
        float: Score zwischen 0.0 (asymmetrisch) und 100.0 (perfekt symmetrisch).
    """
    # 1. Vorbereitung: Graustufen & Binärmaske
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Threshold 10 reicht, da der Hintergrund ja schwarz (0) ist
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Sicherheitscheck: Ist überhaupt ein Objekt da?
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0

    # 2. Schwerpunkt (Zentroid) finden
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 0.0
        
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    h, w = mask.shape[:2]
    
    # 3. Der "Symmetrische Kern" startet als Kopie des Originals
    symmetric_core = mask.copy()

    # 4. Rotations-Loop (60, 120, 180, 240, 300 Grad)
    for angle in range(60, 360, 60):
        # Rotationsmatrix um den Schwerpunkt cx, cy berechnen
        rot_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        # Bild drehen (Wichtig: borderValue=0 sorgt für schwarzen Rand beim Drehen)
        rotated_mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        
        # Schnittmenge bilden: Nur Pixel behalten, die in BEIDEN Versionen da sind
        symmetric_core = cv2.bitwise_and(symmetric_core, rotated_mask)

    # 5. Differenz berechnen (Original - Kern = Asymmetrische Teile)
    asymmetric_mask = cv2.subtract(mask, symmetric_core)
    asymmetric_pixel_count = cv2.countNonZero(asymmetric_mask)

    # 6. Relativen Score berechnen (Skaleninvarianz!)
    # Anteil der Fehlerfläche an der Gesamtfläche
    error_ratio = asymmetric_pixel_count / total_pixels
    
    # Score umdrehen: 0% Fehler = 100% Score
    score = (1.0 - error_ratio) * 100.0
    
    # Auf 2 Nachkommastellen runden und begrenzen
    return max(0.0, min(100.0, round(score, 2)))

def run_symmetry_check(sorted_dir):
    """
    Geht durch den 'Normal'-Ordner, berechnet für jedes Bild den Symmetrie-Score
    und benennt die Datei um (Prefix = Score).
    """
    print("\n[symmetrie.py] Starte Symmetrie-Analyse für Klasse 'Normal'...")
    
    normal_path = os.path.join(sorted_dir, "Normal")
    
    if not os.path.exists(normal_path):
        print(f"Warnung: Ordner {normal_path} existiert nicht. Überspringe Symmetrie-Check.")
        return

    count = 0
    scores = []

    # Durch alle Bilder im Normal-Ordner iterieren
    for root, _, files in os.walk(normal_path):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            file_path = os.path.join(root, filename)
            image = cv2.imread(file_path)
            
            if image is None:
                continue
            
            # Score berechnen
            score = get_symmetry_score(image)
            scores.append(score)
            
            # Neue Datei benennen: "99.55_Originalname.jpg"
            # Formatierung {:06.2f} sorgt für führende Nullen, damit die Sortierung im Explorer stimmt (z.B. 09.50 vs 99.50)
            new_filename = f"{score:05.2f}_{filename}"
            new_path = os.path.join(root, new_filename)
            
            try:
                os.rename(file_path, new_path)
                count += 1
            except OSError as e:
                print(f"Fehler beim Umbenennen von {filename}: {e}")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"[symmetrie.py] Abgeschlossen. {count} Bilder bewertet und umbenannt.")
    print(f"   -> Durchschnittlicher Symmetrie-Score: {avg_score:.2f}")