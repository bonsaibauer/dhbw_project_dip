import os
import shutil
import csv

def get_true_label(raw_label):
    """
    Map the CSV labels to our 4 categories based on user definition.
    """
    labels = [l.strip().lower() for l in raw_label.split(',')]
    
    # 1. BRUCH CHECK
    for l in labels:
        if "breakage" in l or "bruch" in l:
            return "Bruch"
            
    # 2. REST CHECK (Stuck together, fragments)
    for l in labels:
        if "stuck together" in l or "fragment" in l or "rest" in l or "other" in l or "scratches" in l:
            return "Rest"

    # 3. FARBFEHLER CHECK
    for l in labels:
        if "spot" in l or "burnt" in l or "farbfehler" in l:
            return "Farbfehler"
            
    # 4. NORMAL CHECK
    if "normal" in labels:
        return "Normal"

    return "Rest"

def evaluate_results(sorted_dir, csv_path):
    print(f"\n[ergebnis.py] Starte Verifizierung mit {csv_path}...")

    # 1. Ground Truth aus CSV laden
    ground_truth = {} 
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_path_csv = row['image']
                
                # Wir bauen den Schlüssel "Ordner/Datei.jpg" (z.B. "normal/000.jpg")
                parts = full_path_csv.split('/')
                if len(parts) >= 2:
                    key = f"{parts[-2]}/{parts[-1]}".lower()
                else:
                    key = os.path.basename(full_path_csv).lower()
                
                label_raw = row['label']
                true_cat = get_true_label(label_raw)
                
                ground_truth[key] = true_cat
                
    except Exception as e:
        print(f"Fehler beim Lesen der CSV: {e}")
        return

    # 2. Auswertung
    categories = ["Normal", "Bruch", "Farbfehler", "Rest"]
    
    stats = {
        "soll": {c: 0 for c in categories},
        "hits": {c: 0 for c in categories},
        "misses": 0
    }
    
    # SOLL-Werte zählen
    for tc in ground_truth.values():
        if tc in stats["soll"]:
            stats["soll"][tc] += 1

    falsch_dir = os.path.join(sorted_dir, "Falsch")
    if os.path.exists(falsch_dir): shutil.rmtree(falsch_dir)
    os.makedirs(falsch_dir)

    processed_count = 0
    
    # Durch alle Ergebnis-Ordner gehen
    for current_folder in categories:
        folder_path = os.path.join(sorted_dir, current_folder)
        if not os.path.exists(folder_path): continue
            
        # os.walk verwenden, falls doch Unterordner existieren
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): continue
                
                # --- NEU: Prefix entfernen (Symmetrie-Score) ---
                # Dateiname könnte jetzt so aussehen: "99.50_001.JPG"
                # Wir splitten am Unterstrich. Wenn der erste Teil eine Zahl ist, ignorieren wir ihn.
                filename_clean = filename.lower()
                if "_" in filename_clean:
                    parts = filename_clean.split('_', 1) # Nur beim ersten _ splitten
                    # Prüfen ob der erste Teil wie ein Score aussieht (z.B. "99.50")
                    try:
                        float(parts[0])
                        # Es ist eine Zahl -> Wir nehmen den Rest als echten Namen
                        filename_clean = parts[1]
                    except ValueError:
                        # Keine Zahl -> Wahrscheinlich "Folder_File.jpg" aus bruch.py -> lassen wir so
                        pass

                # -----------------------------------------------
                
                # --- MATCHING LOGIK ---
                found_true_cat = None
                
                # Versuch 1: Direkter Match mit bereinigtem Namen
                
                # Versuch 2: Match mit Umbenennung "Ordner_Datei.jpg" -> "Ordner/Datei.jpg"
                # Hier nutzen wir filename_clean
                reconstructed_key = filename_clean.replace('_', '/', 1)
                
                if reconstructed_key in ground_truth:
                    found_true_cat = ground_truth[reconstructed_key]
                else:
                    # Fallback: Suche nach Dateinamen-Endung im Ground Truth
                    matches = [val for k, val in ground_truth.items() if k.endswith(f"/{filename_clean}") or k == filename_clean]
                    if len(matches) > 0:
                        found_true_cat = matches[0]
                
                if found_true_cat is None:
                    # Bild nicht in CSV gefunden (z.B. Testbild)
                    continue

                processed_count += 1
                
                # ABGLEICH
                if found_true_cat == current_folder:
                    stats["hits"][found_true_cat] += 1
                else:
                    stats["misses"] += 1
                    # Verschieben
                    new_name = f"SOLL_{found_true_cat}_IST_{current_folder}_{filename}"
                    try:
                        shutil.move(os.path.join(root, filename), os.path.join(falsch_dir, new_name))
                    except Exception as e:
                        print(f"Fehler beim Verschieben von {filename}: {e}")

    # AUSGABE
    print("\n" + "="*65)
    print("   ERGEBNIS EVALUIERUNG (Vergleich mit Ground-Truth)")
    print("="*65)
    print(f"{'Kategorie':<15} | {'Soll (CSV)':<12} | {'Treffer (Ist)':<15} | {'Genauigkeit':<10}")
    print("-" * 65)
    
    total_soll = 0
    total_hits = 0
    
    for cat in categories:
        s = stats["soll"][cat]
        h = stats["hits"][cat]
        acc = (h/s*100) if s > 0 else 0
        print(f"{cat:<15} | {s:<12} | {h:<15} | {acc:.1f}%")
        total_soll += s
        total_hits += h

    print("-" * 65)
    tot_acc = (total_hits/total_soll*100) if total_soll > 0 else 0
    
    missing = total_soll - processed_count
    print(f"{'GESAMT':<15} | {total_soll:<12} | {total_hits:<15} | {tot_acc:.1f}%")
    
    if missing > 0:
        print(f"\n[Info] {missing} Bilder aus der CSV wurden nicht in den Ordnern gefunden.")
        
    print(f"\nFalsch zugeordnete Bilder ({stats['misses']}) sind in '{falsch_dir}'")
    print("="*65)