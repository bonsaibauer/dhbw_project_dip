"""
main.py

Haupt-Skript des Projekts.
1. Lädt die Ground Truth Daten.
2. Erstellt die Ausgabe-Ordner.
3. Iteriert über alle Bilder.
4. Ruft `image_processor` für Vorverarbeitung und Klassifizierung auf.
5. Speichert die Ergebnisse in den korrekten Ordnern.
6. Erstellt Statistiken.
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm # Optionale, aber coole Fortschrittsanzeige
import config         # Unsere Konfiguration
import image_processor as proc # Unsere Werkzeugkiste

def load_ground_truth(csv_path):
    """
    Lädt die 'image_anno.csv' und wandelt sie in ein
    einfach nutzbares Dictionary (Lexikon) um.
    
    Key:   'Images/Normal/000.JPG'
    Value: 'Normal'
    """
    print(f"Lade Ground Truth von: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"FEHLER: Ground Truth Datei nicht gefunden: {csv_path}")
        return None

    # Mapping-Funktion, wie in unserer Analyse besprochen
    def map_label_to_class(label_str):
        label_str = str(label_str).lower()
        if 'breakage' in label_str:
            return "Bruch"
        elif 'spot' in label_str or 'burnt' in label_str:
            return "Farbfehler"
        elif label_str == 'normal':
            return "Normal"
        else:
            return "Rest"

    # Spalten erstellen
    df['target_class'] = df['label'].apply(map_label_to_class)
    
    # Den einzigartigen Key erstellen (z.B. 'Images/Normal/000.JPG')
    df['unique_key'] = df['image'].str.replace(r'^fryum/Data/', '', regex=True)
    df['unique_key'] = df['unique_key'].str.replace('\\', '/')
    
    # Dictionary erstellen und zurückgeben
    ground_truth_dict = dict(zip(df['unique_key'], df['target_class']))
    print(f"{len(ground_truth_dict)} Ground Truth Einträge geladen.")
    return ground_truth_dict

def create_output_dirs():
    """
    Erstellt alle notwendigen Ausgabe-Ordner basierend auf config.CLASSES.
    """
    print(f"Erstelle Ausgabe-Ordner in: {config.OUTPUT_DIR}")
    for class_name in config.CLASSES:
        os.makedirs(config.OUTPUT_DIR / class_name, exist_ok=True)

def main():
    """
    Haupt-Ausführungsfunktion
    """
    # 1. Setup
    create_output_dirs()
    ground_truth = load_ground_truth(config.ANNOTATION_FILE)
    if ground_truth is None:
        return # Abbruch, wenn Ground Truth fehlt

    # 2. Alle Bildpfade sammeln (unterstützt .png und .JPG)
    extensions = ('*.png', '*.JPG', '*.jpg')
    all_paths = []
    for ext in extensions:
        all_paths.extend(config.NORMAL_IMAGES_DIR.glob(ext))
        all_paths.extend(config.ANOMALY_IMAGES_DIR.glob(ext))
        
    print(f"Starte Verarbeitung von {len(all_paths)} Bildern...")

    # 3. Statistiken für das Fazit
    stats = {'total': len(all_paths), 'processed': 0, 'failed': 0, 'correct': 0, 'incorrect': 0}

    # 4. Haupt-Schleife über alle Bilder (mit Fortschrittsbalken)
    for img_path in tqdm(all_paths, desc="Fortschritt", unit="Bild"):
        
        try:
            # --- Schritt A: Vorverarbeitung ---
            resized_img, resized_mask, main_contour = proc.preprocess_image(img_path)

            if resized_img is None:
                print(f"Warnung: Konnte kein Objekt in {img_path.name} finden. Übersprungen.")
                stats['failed'] += 1
                continue

            # --- Schritt B: Klassifizierung ---
            predicted_label = proc.classify_image(resized_img, resized_mask, main_contour)
            stats['processed'] += 1
            
            # --- Schritt C: Ground Truth Abgleich ---
            # Erzeuge den Key (z.B. "Images/Normal/000.JPG")
            relative_path = img_path.relative_to(config.DATA_DIR).as_posix()
            true_label = ground_truth.get(relative_path)
            
            if true_label is None:
                print(f"Warnung: Kein Ground Truth für {relative_path} gefunden. Übersprungen.")
                continue

            # --- Schritt D: Bild speichern ---
            filename = img_path.name
            save_name = filename
            
            # Sonderfall "Normal": Symmetrie-Score als Präfix
            if predicted_label == "Normal":
                score = proc.get_symmetry_score(resized_img, resized_mask)
                # Formatierung als "005_bild.png" (3-stellig mit führenden Nullen)
                save_name = f"{score:03d}_{filename}"

            # Speichere im Ordner der *erkannten* Klasse
            save_path = config.OUTPUT_DIR / predicted_label / save_name
            cv2.imwrite(str(save_path), resized_img)

            # --- Schritt E: "Falsch"-Ordner füllen ---
            if predicted_label == true_label:
                stats['correct'] += 1
            else:
                stats['incorrect'] += 1
                # Dateiname wie gefordert: "Korrekt-Label_Erkannt-Label_Dateiname"
                falsch_name = f"Korrekt-{true_label}_Erkannt-{predicted_label}_{filename}"
                falsch_path = config.OUTPUT_DIR / "Falsch" / falsch_name
                cv2.imwrite(str(falsch_path), resized_img) # Speichere das Original-Ausschnitt

        except Exception as e:
            print(f"FEHLER bei der Verarbeitung von {img_path.name}: {e}")
            stats['failed'] += 1

    # 5. Fazit / Statistik ausgeben
    print("\n--- Verarbeitung abgeschlossen ---")
    print(f"Gesamtbilder:       {stats['total']}")
    print(f"Erfolgreich verarb.: {stats['processed']}")
    print(f"Fehlgeschlagen:     {stats['failed']}")
    print("---------------------------------")
    print(f"Korrekt klassifiziert: {stats['correct']}")
    print(f"Falsch klassifiziert:  {stats['incorrect']}")
    
    if stats['processed'] > 0:
        accuracy = (stats['correct'] / stats['processed']) * 100
        print(f"Genauigkeit:         {accuracy:.2f}%")
    print("---------------------------------")


# Standard-Python-Konvention: Führe die main-Funktion aus, 
# wenn das Skript direkt gestartet wird.
if __name__ == "__main__":
    main()