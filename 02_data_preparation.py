import os
import shutil
import random
from pathlib import Path

def split_files(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Teilt Dateien in Train/Validation/Test auf - DISJUNKT!
    """
    # Stelle sicher, dass die Summe 1.0 ist (oder nahe dran)
    if not (train_ratio + val_ratio + test_ratio) == 1.0:
        print("Warnung: Ratios summieren sich nicht zu 1.0. Normiere...")
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio = 1.0 - train_ratio - val_ratio


    random.shuffle(files)
    n_total = len(files)
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Der Rest geht an Test, um Rundungsfehler auszugleichen
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    return train_files, val_files, test_files

def create_single_cnn_data():
    """
    Erstellt Daten für ein einziges 4-Klassen-CNN:
    - Normal (0)
    - Bruch (1) 
    - Farbfehler (2)
    - Rest (3)
    """
    print("Bereite Daten für 4-Klassen-CNN vor...")
    
    source_dir = Path("processed_images")
    target_dir = Path("data_single_cnn")
    
    # Zielordner für das einzelne CNN
    cnn_folders = [
        target_dir / "Lernen" / "Normal",
        target_dir / "Lernen" / "Bruch",
        target_dir / "Lernen" / "Farbfehler", 
        target_dir / "Lernen" / "Rest",
        target_dir / "Validation" / "Normal",
        target_dir / "Validation" / "Bruch",
        target_dir / "Validation" / "Farbfehler",
        target_dir / "Validation" / "Rest",
        target_dir / "Test" / "Normal",
        target_dir / "Test" / "Bruch", 
        target_dir / "Test" / "Farbfehler",
        target_dir / "Test" / "Rest"
    ]
    
    # Alte Daten löschen und neue Ordner erstellen
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    for folder in cnn_folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Alle 4 Klassen verarbeiten
    classes = ["Normal", "Bruch", "Farbfehler", "Rest"]
    total_stats = {}
    
    for class_name in classes:
        # Bilder für diese Klasse sammeln
        # FIX: Nur ein glob-Aufruf.
        class_files = list((source_dir / class_name).glob("*.jpg"))
        
        if not class_files:
            print(f"Warnung: Keine Bilder gefunden für Klasse {class_name}")
            total_stats[class_name] = {"total": 0, "train": 0, "val": 0, "test": 0}
            continue
            
        # Bilder aufteilen
        train_files, val_files, test_files = split_files(class_files)
        
        # Kopieren
        for file in train_files:
            shutil.copy2(file, target_dir / "Lernen" / class_name / file.name)
        for file in val_files:
            shutil.copy2(file, target_dir / "Validation" / class_name / file.name)
        for file in test_files:
            shutil.copy2(file, target_dir / "Test" / class_name / file.name)
        
        total_stats[class_name] = {
            "total": len(class_files),
            "train": len(train_files),
            "val": len(val_files), 
            "test": len(test_files)
        }
        
        print(f"{class_name}: {len(class_files)} Bilder")
        print(f"  Lernen: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
    
    # Gesamtstatistik
    print(f"\n=== GESAMTSTATISTIK ===")
    total_train = sum(stats["train"] for stats in total_stats.values())
    total_val = sum(stats["val"] for stats in total_stats.values())
    total_test = sum(stats["test"] for stats in total_stats.values())
    total_all = total_train + total_val + total_test
    
    if total_all == 0:
        print("FEHLER: Keine Bilder gefunden!")
        return

    print(f"Gesamt: {total_all} Bilder")
    print(f"Lernen: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"Validation: {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"Test: {total_test} ({total_test/total_all*100:.1f}%)")

def main():
    """
    Hauptfunktion für die Datenvorbereitung - EINZELNES CNN
    """
    print("=== DATENVORBEREITUNG FÜR 4-KLASSEN-CNN ===")
    print("Aufteilung: 70% Lernen, 15% Validation, 15% Test")
    
    # Überprüfen ob vorverarbeitete Bilder existieren
    if not Path("processed_images").exists():
        print("FEHLER: processed_images Ordner nicht gefunden!")
        print("Führe zuerst 01_image_preprocessing.py aus!")
        return
    
    # Alte CNN-Ordner löschen (falls vorhanden)
    for old_dir in ["data_cnn1", "data_cnn2", "data_cnn3"]:
        if Path(old_dir).exists():
            shutil.rmtree(old_dir)
            print(f"Alten Ordner gelöscht: {old_dir}")
    
    # Daten für das einzelne CNN erstellen
    create_single_cnn_data()
    
    print("\n=== DATENVORBEREITUNG ABGESCHLOSSEN ===")
    print("Ordnerstruktur erstellt: data_single_cnn/")
    print("Klassen: Normal, Bruch, Farbfehler, Rest")

if __name__ == '__main__':
    main()