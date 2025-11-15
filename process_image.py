import os
import cv2
import numpy as np
import shutil
import csv

# --- ⚙️ EINSTELLUNGEN (Die goldene Mitte v14) ⚙️ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "output", "test_crops")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "sorted_results")
CSV_PATH = os.path.join(BASE_DIR, "data", "image_anno.csv")

# 1. GEOMETRIE
# Wir entspannen minimal (von 0.863 auf 0.862).
# Das rettet "Grenzgänger", fängt aber immer noch die 0.852 Anomalie.
THRESH_CIRCULARITY = 0.862  
THRESH_SOLIDITY = 0.71     

# 2. FARBE
# 0.00085 war zu tödlich (463). 0.0012 war zu nett (503).
# Wir nehmen exakt die Mitte: 0.0010.
THRESH_COLOR_HARD = 0.0010   
COLOR_SENSITIVITY = 67     

# 3. OBERFLÄCHE (Black-Hat)
# 0.0025 war zu tödlich. Wir gehen auf 0.0035 hoch.
THRESH_SURFACE_HARD = 0.0035 

# 4. MATRIX (Kombi-Check) - HIER ENTSCHÄRFEN WIR!
# Wir heben die "Soft"-Grenzen an, damit nicht jeder Schatten bestraft wird.
THRESH_COLOR_SOFT = 0.0007    # War 0.0005
THRESH_SURFACE_SOFT = 0.0020  # War 0.0007 (Viel zu streng!)
THRESH_CIRC_SOFT = 0.875

MIN_HOLES = 5 
MIN_AREA = 1000            

# --- 🛠️ FUNKTIONEN 🛠️ ---

def get_features(img):
    # --- A. GEOMETRIE ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    pixel_area = cv2.countNonZero(thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, 0, 0, 0, 0, 0
    
    main_cnt_index = -1
    max_area = 0
    for i, c in enumerate(contours):
        a = cv2.contourArea(c)
        if a > max_area: max_area, main_cnt_index = a, i
            
    if main_cnt_index == -1: return 0, 0, 0, 0, 0, 0

    c = contours[main_cnt_index]
    hull = cv2.convexHull(c)
    
    contour_area_filled = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(hull)
    
    solidity = float(pixel_area) / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * contour_area_filled) / (perimeter ** 2) if perimeter > 0 else 0

    hole_count = 0
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] == main_cnt_index:
                if cv2.contourArea(contours[i]) > 200: hole_count += 1

    # --- B. FARBE (Red Sniper Logic) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_cookie = v > 10
    
    # Standard Dunkel
    dark_spots = (v < COLOR_SENSITIVITY) & (v > 10)
    # Rote Stifte (Hue > 160)
    red_pen = (h > 160) & (s > 60) & (v > 20)
    
    error_pixels = cv2.bitwise_or(dark_spots.astype(np.uint8), red_pen.astype(np.uint8))
    
    color_score = 0.0
    if np.sum(mask_cookie) > 0:
        color_score = cv2.countNonZero(error_pixels) / np.sum(mask_cookie)

    # --- C. SURFACE (Blackhat) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, defect_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    mask_eroded = cv2.erode(thresh, np.ones((5,5), np.uint8), iterations=3)
    defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=mask_eroded)
    defect_pixels = cv2.countNonZero(defect_mask)
    surface_score = float(defect_pixels) / pixel_area if pixel_area > 0 else 0

    return circularity, solidity, color_score, pixel_area, hole_count, surface_score

def load_csv_labels():
    labels = {}
    if not os.path.exists(CSV_PATH): return labels
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    full_path = row[0]
                    parts = full_path.split("/")
                    if len(parts) >= 2:
                        key = f"{parts[-2]}/{parts[-1]}"
                        labels[key] = row[1].lower()
    except: pass
    return labels

def map_csv_to_class(csv_label):
    l = csv_label
    if "normal" in l: return "Normal"
    if "burnt" in l or "colour" in l or "spot" in l: return "Farbfehler"
    if "breakage" in l or "stuck" in l or "scratch" in l or "corner" in l: return "Bruch"
    return "Rest"

def main():
    print("--- 🚀 FINAL RUN v14: The Golden Mean ---")
    
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    classes = ["Normal", "Bruch", "Farbfehler", "Rest", "Falsch"]
    for cls in classes: os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    ground_truth = load_csv_labels()
    if not os.path.exists(INPUT_DIR): return
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")]
    
    stats = {k: 0 for k in classes}
    false_counter = 0

    for idx, filename in enumerate(files):
        path = os.path.join(INPUT_DIR, filename)
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: continue

        circ, solid, color, area, holes, surface = get_features(img)
        
        detected = "Normal"
        
        if area < MIN_AREA:
            detected = "Rest"
        # 1. Harte Kriterien
        elif holes < MIN_HOLES:
            detected = "Bruch"
        elif circ < THRESH_CIRCULARITY:
            detected = "Bruch"
        elif solid < THRESH_SOLIDITY:
            detected = "Bruch"
        elif color > THRESH_COLOR_HARD:
            detected = "Farbfehler"
        elif surface > THRESH_SURFACE_HARD:
            detected = "Bruch" # Starke Kratzer
            
        # 2. Matrix (Kombi)
        elif (color > THRESH_COLOR_SOFT) and (surface > THRESH_SURFACE_SOFT):
            detected = "Farbfehler" 
        elif (circ < THRESH_CIRC_SOFT) and (surface > THRESH_SURFACE_SOFT):
            detected = "Bruch"

        else:
            detected = "Normal"

        # Validierung
        parts = filename.split("_")
        if len(parts) >= 2:
            folder_part = parts[0]
            file_part = parts[1].replace(".png", ".JPG")
            lookup_key = f"{folder_part}/{file_part}"
            
            if lookup_key in ground_truth:
                true_class = map_csv_to_class(ground_truth[lookup_key])
                if detected != true_class:
                    # Speichert jetzt auch die Werte im Dateinamen für bessere Analyse
                    falsch_name = f"{true_class}_als_{detected}_C{color:.4f}_S{surface:.4f}_{filename}"
                    shutil.copy(path, os.path.join(OUTPUT_DIR, "Falsch", falsch_name))
                    false_counter += 1

        final_name = filename
        if detected == "Normal":
            final_name = f"{circ:.3f}_{filename}"
            
        shutil.copy(path, os.path.join(OUTPUT_DIR, detected, final_name))
        stats[detected] += 1

        if idx % 100 == 0: print(f"   ... {idx}")

    print("-" * 30)
    print("✅ FERTIG!")
    print(f"📊 Statistik: {stats}")
    print(f"❌ Falsch zugeordnet: {false_counter}")

if __name__ == "__main__":
    main()