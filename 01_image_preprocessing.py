import cv2
import numpy as np
import os
import shutil
from pathlib import Path

def run_preprocessing(image):
    """
    Segmentierung des Objekts basierend auf dem Code des Kollegen
    """
    image_copy = image.copy()
    image_work = image.copy()  # Arbeitskopie für Maskierung

    # --- HSV Hintergrundentfernung ---
    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    
    # Grün-Definition für Hintergrund (Anpassbar je nach Licht)
    lower_green = np.array([35, 40, 30])
    upper_green = np.array([85, 255, 255])
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_object = cv2.bitwise_not(mask_green)  # Objekt ist Nicht-Grün
    
    image_work = cv2.bitwise_and(image_work, image_work, mask=mask_object)

    # Konturen finden
    _, thresh = cv2.threshold(cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for ele in contours:
        # Nur große Objekte beachten
        if cv2.contourArea(ele) > 30000:
            rect = cv2.minAreaRect(ele)
            
            # Querformat erzwingen
            if rect[1][1] > rect[1][0]:
                rect = [rect[0], [rect[1][1], rect[1][0]], rect[2] - 90]
            
            boxf = cv2.boxPoints(rect)

            # Maskierung innerhalb der Box
            mask = np.zeros((image_copy.shape[0], image_copy.shape[1])).astype(np.uint8)
            cv2.drawContours(mask, [ele], -1, (1, 1, 1), cv2.FILLED, 8)
            image_work[mask == 0] = (0, 0, 0)

            # Perspektivische Transformation
            size = (600, 400)
            dst_pts = np.array([
                [-1, size[1]], 
                [-1, -1], 
                [size[0], -1], 
                [size[0], size[1]]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)
            
            # Zielgröße für unsere CNN-Pipeline
            target_width = 400
            target_height = 400
            
            warped = cv2.warpPerspective(image_work, M, (size[0], size[1]), cv2.INTER_CUBIC)
            warped = cv2.resize(warped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

            return warped
    
    return None

def process_all_images():
    """
    Verarbeitet alle Bilder und erstellt die Basis-Ordnerstruktur
    mit korrekter Klassenzuordnung aus der CSV
    """
    # Basis-Pfade - relativ zum Projektordner
    project_root = Path(".")
    data_dir = project_root / "data"
    
    input_normal = data_dir / "Images" / "Normal"
    input_anomaly = data_dir / "Images" / "Anomaly"
    
    # Ausgabe-Ordner für vorverarbeitete Bilder
    output_base = project_root / "processed_images"
    
    # Basis-Ordnerstruktur erstellen
    output_folders = [
        output_base / "Normal",
        output_base / "Bruch", 
        output_base / "Farbfehler",
        output_base / "Rest"
    ]
    
    # Alten Output löschen und neu erstellen
    if output_base.exists():
        shutil.rmtree(output_base)
    
    for folder in output_folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    print("Starte Bildvorverarbeitung...")
    
    # CSV-Datei für Klassenzuordnung lesen
    csv_file = data_dir / "image_anno.csv"
    class_mapping = {}
    
    if csv_file.exists():
        print("Lese Klassenzuordnung aus CSV...")
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Header überspringen
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    image_path = parts[0]
                    original_class = parts[1]
                    
                    # Vereinfachung der Klassen gemäß Aufgabenstellung
                    if "normal" in original_class:
                        simplified_class = "Normal"
                    elif "break" in original_class or "bruch" in original_class.lower():
                        simplified_class = "Bruch"
                    elif "colour" in original_class or "color" in original_class or "farb" in original_class.lower():
                        simplified_class = "Farbfehler"
                    else:
                        simplified_class = "Rest"
                    
                    # Dateiname extrahieren
                    filename = Path(image_path).name
                    class_mapping[filename] = simplified_class
        print(f"Klassenzuordnung für {len(class_mapping)} Bilder geladen")
    
    # Zähler für korrekte Statistik
    actual_counts = {"Normal": 0, "Bruch": 0, "Farbfehler": 0, "Rest": 0}
    
    # Normal-Bilder verarbeiten
    print("\nVerarbeite Normal-Bilder...")
    if input_normal.exists():
        
        # FIX: Nur ein glob-Aufruf. Auf Windows findet ".jpg" auch ".JPG"
        normal_files = list(input_normal.glob("*.jpg"))
        
        for img_file in normal_files:
            image = cv2.imread(str(img_file))
            if image is not None:
                processed = run_preprocessing(image)
                if processed is not None:
                    # Präfix "normal_" hinzufügen um Überschreiben zu vermeiden
                    output_filename = f"normal_{img_file.name}"
                    output_path = output_base / "Normal" / output_filename
                    cv2.imwrite(str(output_path), processed)
                    actual_counts["Normal"] += 1
                    if actual_counts["Normal"] % 50 == 0:
                        print(f"  {actual_counts['Normal']} Normal-Bilder verarbeitet...")
                else:
                    print(f"  ✗ Kein Objekt gefunden: {img_file.name}")
    
    # Anomalie-Bilder verarbeiten
    print("\nVerarbeite Anomalie-Bilder...")
    if input_anomaly.exists():
        
        # FIX: Nur ein glob-Aufruf.
        anomaly_files = list(input_anomaly.glob("*.jpg"))
        
        for img_file in anomaly_files:
            image = cv2.imread(str(img_file))
            if image is not None:
                processed = run_preprocessing(image)
                if processed is not None:
                    # Klasse aus CSV-Mapping bestimmen
                    target_class = class_mapping.get(img_file.name, "Rest")
                    
                    # Präfix "anomaly_" hinzufügen um Überschreiben zu vermeiden
                    output_filename = f"anomaly_{img_file.name}"
                    output_path = output_base / target_class / output_filename
                    cv2.imwrite(str(output_path), processed)
                    actual_counts[target_class] += 1
                    if sum(actual_counts.values()) % 10 == 0:
                        print(f"  {sum(actual_counts.values())} Bilder insgesamt verarbeitet...")
                else:
                    print(f"  ✗ Kein Objekt gefunden: {img_file.name}")
    
    # Statistik ausgeben - NUR die tatsächlichen Zähler verwenden
    print(f"\n=== VORVERARBEITUNG ABGESCHLOSSEN ===")
    print(f"Gespeichert in: {output_base}")
    
    for folder in output_folders:
        count = actual_counts[folder.name]  # Verwende die korrekten Zähler
        print(f"  {folder.name}: {count} Bilder")
    
    total_processed = sum(actual_counts.values())
    print(f"\nTotal verarbeitete Bilder: {total_processed}")

if __name__ == '__main__':
    process_all_images()