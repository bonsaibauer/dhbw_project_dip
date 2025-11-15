import shutil
import cv2
import numpy as np
import os

# ==========================================
# TEIL 1: BILDVORVERARBEITUNG (SEGMENTIERUNG)
# ==========================================

def run_preprocessing(image, result):
    image_copy = image.copy()
    image_work = image.copy() # Arbeitskopie für Maskierung

    # --- HSV Hintergrundentfernung ---
    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    
    # Grün-Definition für Hintergrund (Anpassbar je nach Licht)
    lower_green = np.array([35, 40, 30])
    upper_green = np.array([85, 255, 255])
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_object = cv2.bitwise_not(mask_green) # Objekt ist Nicht-Grün
    
    image_work = cv2.bitwise_and(image_work, image_work, mask=mask_object)

    # Konturen finden
    _, thresh = cv2.threshold(cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    processed = False
    for ele in contours:
        # Nur große Objekte beachten (Filterung von Rauschen)
        if cv2.contourArea(ele) > 30000:
            rect = cv2.minAreaRect(ele)
            
            # Querformat erzwingen (Winkel und Dimensionen tauschen)
            # rect = ((center_x, center_y), (width, height), angle)
            if rect[1][1] > rect[1][0]:
                rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
            
            boxf = cv2.boxPoints(rect)
            boxf = np.int64(boxf) # Konvertierung zu Integer für drawContours

            # Maskierung innerhalb der Box
            mask = np.zeros((image_copy.shape[0], image_copy.shape[1])).astype(np.uint8)
            cv2.drawContours(mask, [ele], -1, (255), cv2.FILLED) # Weiß füllen
            
            # Alles außerhalb der Kontur schwarz machen
            image_work[mask == 0] = (0, 0, 0)

            # Perspektivische Transformation (Warp)
            size = (600, 400)
            # Ziel-Koordinaten für den Warp
            dst_pts = np.array([[0, size[1]-1], [0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1]], dtype="float32")
            
            # Sortieren der boxf Punkte, damit sie zu dst_pts passen
            # (Dies ist eine Vereinfachung, idealerweise nutzt man eine order_points Funktion)
            # Hier verlassen wir uns auf cv2.minAreaRect Reihenfolge, was oft okay ist, aber riskant.
            
            M = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)
            
            target_width = 400
            target_height = 400
            
            warped = cv2.warpPerspective(image_work, M, (size[0], size[1]), cv2.INTER_CUBIC)
            warped = cv2.resize(warped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

            result.append({"name": "Result", "data": warped})
            processed = True
            
    return processed

def prepare_dataset(source_dir, target_dir):
    """Liest Bilder ein, segmentiert sie und speichert sie in target_dir."""
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    print(f"Starte Vorverarbeitung von {source_dir} nach {target_dir}...")
    
    for root, dirs, files in os.walk(source_dir):
        files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not files: continue
        
        # Struktur beibehalten (Unterordner)
        class_name = os.path.basename(root)
        save_path = os.path.join(target_dir, class_name)
        os.makedirs(save_path, exist_ok=True)
        
        for name in files:
            full_path = os.path.join(root, name)
            image = cv2.imread(full_path)
            
            if image is not None:
                res = []
                has_result = run_preprocessing(image, res)
                
                if has_result:
                    for item in res:
                        if item["name"] == "Result":
                            cv2.imwrite(os.path.join(save_path, name), item["data"])
                            
    print("Vorverarbeitung abgeschlossen.")

# ==========================================
# TEIL 2: QUALITÄTSKONTROLLE & SORTIERUNG
# ==========================================

def get_contours_hierarchy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def analyze_geometry_features(contours, hierarchy):
    stats = {
        "has_object": False, "area": 0, "solidity": 0,
        "num_windows": 0, "has_center_hole": False, "main_contour": None,
        "window_areas": []  # <--- NEU: Liste für die Flächengrößen
    }
    
    if not contours: return stats

    # Hauptobjekt finden
    main_cnt_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    main_cnt = contours[main_cnt_idx]
    area = cv2.contourArea(main_cnt)
    
    stats["has_object"] = True
    stats["area"] = area
    stats["main_contour"] = main_cnt

    # Löcher analysieren
    epsilon_factor = 0.04
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            
            # Nur direkte Kinder des Hauptobjekts (Löcher)
            if parent_idx == main_cnt_idx:
                hole_area = cv2.contourArea(cnt)
                if hole_area < 100: continue 

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
                corners = len(approx)

                # Unterscheidung Fenster vs. Mittelloch
                if 3 <= corners <= 5 and hole_area > 500:
                    stats["num_windows"] += 1
                    stats["window_areas"].append(hole_area) # <--- NEU: Fläche speichern
                elif corners > 5 and hole_area < 3000:
                    stats["has_center_hole"] = True
                    
    return stats

def detect_defects(image, spot_threshold=15, debug=False):
    """
    Nutzt Morphological Black-Hat, um dunkle Flecken unabhängig von
    Schattenverläufen zu finden.
    """
    # 1. Konvertierung in Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Maske erstellen (Objekt vom schwarzen Hintergrund trennen)
    # Alles was nicht fast schwarz ist, ist das Objekt
    _, mask_obj = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 3. Sicherheits-Erosion
    # WICHTIG: Wir müssen den Rand stark verkleinern.
    # Der Übergang vom hellen Snack zum schwarzen Hintergrund ist eine "riesige Kante",
    # die wir ignorieren müssen.
    kernel_erode = np.ones((5, 5), np.uint8)
    mask_analysis = cv2.erode(mask_obj, kernel_erode, iterations=5) 
    # Falls zu viel vom Objekt verschwindet: iterations verringern (z.B. 3)

    # 4. Black-Hat Transformation
    # Wir definieren eine Größe für die Flecken, die wir suchen.
    # Kernel-Größe (15, 15) bedeutet: "Suche Dinge, die kleiner sind als ca. 15 Pixel"
    # Alles was größer ist (wie Schattenverläufe), wird ignoriert.
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_morph)

    # Das blackhat_img ist jetzt fast schwarz, nur die Flecken leuchten weiß hervor.
    # Je heller der Pixel im blackhat_img, desto stärker ist der lokale Kontrast (Fleck).

    # 5. Schwellenwert auf den Kontrast
    # Hier definieren wir: "Der Fleck muss mindestens 30 Helligkeitsstufen dunkler 
    # sein als seine direkte Umgebung".
    contrast_threshold = 30 
    _, mask_defects = cv2.threshold(blackhat_img, contrast_threshold, 255, cv2.THRESH_BINARY)

    # 6. Nur Defekte IM Objekt betrachten
    valid_defects = cv2.bitwise_and(mask_defects, mask_defects, mask=mask_analysis)
    
    # (Optional) Rauschen entfernen (Punkte kleiner als 2px weg)
    valid_defects = cv2.morphologyEx(valid_defects, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    # 7. Ergebnis
    spot_area = cv2.countNonZero(valid_defects)
    is_defective = spot_area > spot_threshold

    # DEBUG: Um zu sehen, was der Black-Hat sieht
    if debug:
        # Wir geben das maskierte Blackhat-Bild zurück, damit du die Flecken leuchten siehst
        debug_view = cv2.bitwise_and(blackhat_img, blackhat_img, mask=mask_analysis)
        return {"is_defective": is_defective, "spot_area": spot_area, "debug_image": debug_view}

    return {
        "is_defective": is_defective,
        "spot_area": spot_area
    }

def sort_dataset_manual_rules(source_data_dir, sorted_data_dir):
    print("\nStarte Sortierung und Symmetrie-Berechnung...")
    CLASSES = ["Normal", "Bruch", "Farbfehler", "Rest"]
    
    if os.path.exists(sorted_data_dir):
        shutil.rmtree(sorted_data_dir)
    for cls in CLASSES:
        os.makedirs(os.path.join(sorted_data_dir, cls), exist_ok=True)

    stats_counter = {k: 0 for k in CLASSES}

    for root, dirs, files in os.walk(source_data_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            if image is None: continue

            contours, hierarchy = get_contours_hierarchy(image)
            geo = analyze_geometry_features(contours, hierarchy)
            
            target_class = "Normal"
            reason = "OK"
            # Prefix für den Dateinamen (Standard leer)
            file_prefix = "" 

            total_holes = geo["num_windows"] + (1 if geo["has_center_hole"] else 0)

            if not geo["has_object"]:
                target_class = "Rest"
                reason = "Kein Objekt"
            elif total_holes < 7:
                target_class = "Bruch"
                reason = f"Zu wenig Löcher: {total_holes}"
            elif total_holes > 7:
                target_class = "Rest"
                reason = f"Zu viele Fragmente: {total_holes}"
            else:
                # --- Geometrie ist OK (7 Löcher), jetzt Farbcheck ---
                col_res = detect_defects(image, spot_threshold=15)
                
                if col_res["is_defective"]:
                    target_class = "Farbfehler"
                    reason = f"Fleck: {col_res['spot_area']}px"
                else:
                    # --- ALLES OK -> SYMMETRIE BERECHNEN ---
                    # Wir berechnen, wie stark die Fensterflächen voneinander abweichen.
                    areas = geo["window_areas"]
                    
                    if len(areas) > 0:
                        mean_a = np.mean(areas)
                        std_a = np.std(areas) # Standardabweichung
                        
                        # Variationskoeffizient (Wie viel % weicht ein Fenster vom Durschnitt ab?)
                        # Kleiner ist besser.
                        cv = std_a / mean_a if mean_a > 0 else 0
                        
                        # Score invertieren: 100 = 0% Abweichung, 0 = 30%+ Abweichung
                        # Faktor 3.0 ist ein Skalierungsfaktor für Empfindlichkeit
                        symmetry_score = max(0, min(100, int(100 * (1 - (cv * 3.0)))))
                    else:
                        symmetry_score = 0

                    # Prefix erstellen: z.B. "098_" für sehr symmetrisch
                    file_prefix = f"{symmetry_score:03d}_"
                    reason = f"Symmetrie: {symmetry_score}/100"

            # Datei kopieren (mit Prefix nur bei Normal)
            new_filename = f"{file_prefix}{file}"
            shutil.copy(img_path, os.path.join(sorted_data_dir, target_class, new_filename))
            
            stats_counter[target_class] += 1
            print(f"[{target_class}] {new_filename} -> {reason}")

    print(f"\nFertig: {stats_counter}")

# ==========================================
# MAIN PROGRAMM
# ==========================================

# KORREKTUR: __name__ und __main__ (doppelte Unterstriche!)
if __name__ == '__main__':
    
    # PFADE (Bitte anpassen, falls nötig)
    raw_data_dir = os.path.join("data", "Images") 
    processed_data_dir = os.path.join("data", "processed") 
    sorted_data_dir = os.path.join("data", "sorted") 

    # 1. VORVERARBEITUNG
    # Prüfen ob Rohdaten da sind
    if os.path.exists(raw_data_dir):
        # Vorverarbeitung starten
        prepare_dataset(raw_data_dir, processed_data_dir)
        
        # 2. SORTIERUNG
        if os.path.exists(processed_data_dir):
            # KORREKTUR: Hier den richtigen Funktionsnamen aufrufen!
            sort_dataset_manual_rules(processed_data_dir, sorted_data_dir)

    else:
        print(f"Fehler: Quellordner '{raw_data_dir}' nicht gefunden! Bitte Ordner erstellen und Bilder hineinlegen.")