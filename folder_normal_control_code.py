import os
import cv2
import numpy as np

# --- PFAD ZUM AKTUELLEN NORMAL-ORDNER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_FOLDER = os.path.join(BASE_DIR, "output", "sorted_results", "Normal")

# --- DIE EXAKTE LOGIK VOM LETZTEN LAUF (v12) ---
def get_features(img):
    # A. Geometrie
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

    # B. Farbe (v12 Red Sniper Logic)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_cookie = v > 10
    
    # Standard Dunkel
    dark_spots = (v < 67) & (v > 10)
    # Red Sniper (Rotstich)
    red_pen = (h > 160) & (s > 60) & (v > 20)
    
    error_pixels = cv2.bitwise_or(dark_spots.astype(np.uint8), red_pen.astype(np.uint8))
    
    color_score = 0.0
    if np.sum(mask_cookie) > 0:
        color_score = cv2.countNonZero(error_pixels) / np.sum(mask_cookie)

    # C. Surface (Blackhat)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, defect_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    mask_eroded = cv2.erode(thresh, np.ones((5,5), np.uint8), iterations=3)
    defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=mask_eroded)
    defect_pixels = cv2.countNonZero(defect_mask)
    surface_score = float(defect_pixels) / pixel_area if pixel_area > 0 else 0

    return circularity, solidity, color_score, pixel_area, hole_count, surface_score

def spy():
    if not os.path.exists(NORMAL_FOLDER):
        print("❌ Normal-Ordner nicht gefunden! Pfad prüfen.")
        return

    files = [f for f in os.listdir(NORMAL_FOLDER) if f.lower().endswith(".png")]
    
    anomalies = []
    normals = []

    print(f"🕵️‍♂️ Untersuche {len(files)} Bilder im Ordner 'Normal'...\n")
    print(f"{'DATEINAME':<35} | {'CIRC':<8} | {'SOLID':<8} | {'COLOR':<8} | {'SURFACE':<8}")
    print("-" * 85)

    for filename in files:
        path = os.path.join(NORMAL_FOLDER, filename)
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: continue

        circ, solid, color, area, holes, surface = get_features(img)
        
        # Wir trennen die echten Normalen von den eingeschmuggelten Anomalien
        if "Anomaly" in filename:
            anomalies.append((filename, circ, solid, color, surface))
            # Drucke die U-Boote sofort aus
            print(f"🚨 {filename:<32} | {circ:.4f}   | {solid:.4f}   | {color:.5f}  | {surface:.5f}")
        else:
            normals.append((circ, solid, color, surface))

    print("-" * 85)
    print(f"Gefundene Anomalien im Normal-Ordner: {len(anomalies)}")
    
    if len(normals) > 0:
        # Berechne Durchschnittswerte der GUTEN Kekse als Referenz
        n_circ = np.mean([x[0] for x in normals])
        n_solid = np.mean([x[1] for x in normals])
        n_color = np.mean([x[2] for x in normals])
        n_surf = np.mean([x[3] for x in normals])
        
        print("\n✅ REFERENZ-WERTE (Durchschnitt der echten Normalen):")
        print(f"   Circularity: {n_circ:.4f}")
        print(f"   Solidity:    {n_solid:.4f}")
        print(f"   Color:       {n_color:.5f}")
        print(f"   Surface:     {n_surf:.5f}")

if __name__ == "__main__":
    spy()