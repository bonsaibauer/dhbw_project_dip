import os
import cv2
import numpy as np

# --- PFADE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_FOLDER = os.path.join(BASE_DIR, "output", "sorted_results", "Normal")

# --- 1:1 KOPIE DEINER AKTUELLEN FEATURE-LOGIK ---
# (Damit wir exakt sehen, was das Hauptprogramm sieht)
def get_features(img):
    # Vorbereitung
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    pixel_area = cv2.countNonZero(thresh)

    # Geometrie
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

    # Farbe (Settings aus dem Hauptprogramm: 67)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    mask_cookie = v > 10
    dark_spots = (v < 67) & (v > 10)
    color_score = np.sum(dark_spots) / np.sum(mask_cookie) if np.sum(mask_cookie) > 0 else 0

    # Surface (Blackhat)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, defect_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    mask_eroded = cv2.erode(thresh, np.ones((5,5), np.uint8), iterations=3)
    defect_mask = cv2.bitwise_and(defect_mask, defect_mask, mask=mask_eroded)
    defect_pixels = cv2.countNonZero(defect_mask)
    surface_score = float(defect_pixels) / pixel_area if pixel_area > 0 else 0

    return circularity, solidity, color_score, pixel_area, hole_count, surface_score

def analyze_enemies():
    print(f"🕵️‍♂️ Untersuche 'Normal'-Ordner: {NORMAL_FOLDER}")
    
    if not os.path.exists(NORMAL_FOLDER):
        print("❌ Ordner nicht gefunden!")
        return

    files = [f for f in os.listdir(NORMAL_FOLDER) if f.endswith(".png")]
    found_anomalies = 0
    
    print("\n--- 🚨 GEFUNDENE U-BOOTE (Anomalien im Normal-Ordner) ---")
    print(f"{'Dateiname':<35} | {'Circ (Ziel >0.863)':<18} | {'Solid (>0.71)':<15} | {'Color (<0.0013)':<15} | {'Surf (<0.007)':<15}")
    print("-" * 110)

    for filename in files:
        # Wir suchen nur Dateien, die "Anomaly" im Namen haben
        if "Anomaly" in filename:
            found_anomalies += 1
            path = os.path.join(NORMAL_FOLDER, filename)
            
            stream = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            
            if img is None: continue
            
            circ, solid, color, area, holes, surface = get_features(img)
            
            # Rote Flagge, wenn ein Wert kritisch nah an der Grenze ist
            print(f"{filename:<35} | {circ:.4f}             | {solid:.4f}          | {color:.5f}         | {surface:.5f}")

    print("-" * 110)
    print(f"Gefundene falsche Normale: {found_anomalies}")
    print("Nutze diese Werte, um die Grenzwerte im Hauptprogramm anzupassen!")

if __name__ == "__main__":
    analyze_enemies()