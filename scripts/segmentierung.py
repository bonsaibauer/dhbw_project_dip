import cv2
import numpy as np
import os
import shutil

def run_preprocessing(image, result):
    """
    Kernlogik: Entfernt grünen Hintergrund, findet das Objekt,
    führt die perspektivische Transformation (Warp) durch.
    """
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
            if rect[1][1] > rect[1][0]:
                rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)
            
            boxf = cv2.boxPoints(rect)
            boxf = np.int64(boxf) 

            # Maskierung innerhalb der Box
            mask = np.zeros((image_copy.shape[0], image_copy.shape[1])).astype(np.uint8)
            cv2.drawContours(mask, [ele], -1, (255), cv2.FILLED) 
            
            # Alles außerhalb der Kontur schwarz machen
            image_work[mask == 0] = (0, 0, 0)

            # Perspektivische Transformation (Warp)
            # Zielgröße für das ausgeschnittene Bild
            target_w, target_h = 400, 400
            
            # Ziel-Koordinaten für den Warp (Originalgröße vor Resize)
            size_warp = (600, 400)
            dst_pts = np.array([[0, size_warp[1]-1], [0, 0], [size_warp[0]-1, 0], [size_warp[0]-1, size_warp[1]-1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)
            
            warped = cv2.warpPerspective(image_work, M, (size_warp[0], size_warp[1]), cv2.INTER_CUBIC)
            warped = cv2.resize(warped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            result.append({"name": "Result", "data": warped})
            processed = True
            
    return processed

def prepare_dataset(source_dir, target_dir):
    """
    Liest Bilder aus source_dir, führt run_preprocessing aus 
    und speichert Ergebnisse in target_dir unter Beibehaltung der Ordnerstruktur.
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    print(f"[segmentierung.py] Starte Vorverarbeitung von {source_dir} nach {target_dir}...")
    
    counter = 0
    
    # os.walk geht durch alle Unterordner
    for root, dirs, files in os.walk(source_dir):
        
        # 1. Relativen Pfad berechnen (z.B. "anomaly" oder "normal")
        rel_path = os.path.relpath(root, source_dir)
        
        # 2. Ziel-Unterordner bestimmen (z.B. "data/processed/anomaly")
        current_target_subdir = os.path.join(target_dir, rel_path)
        
        # 3. Ziel-Unterordner erstellen, falls er noch nicht existiert
        os.makedirs(current_target_subdir, exist_ok=True)
        
        for name in files:
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, name)
                image = cv2.imread(full_path)
                
                if image is not None:
                    res = []
                    has_result = run_preprocessing(image, res)
                    
                    if has_result:
                        for item in res:
                            if item["name"] == "Result":
                                # Speichern im korrekten Unterordner
                                # Da die Ordner getrennt sind, behalten wir den Originalnamen bei
                                save_path = os.path.join(current_target_subdir, name)
                                cv2.imwrite(save_path, item["data"])
                                counter += 1
                                
    print(f"[segmentierung.py] Abgeschlossen. {counter} Bilder verarbeitet.")