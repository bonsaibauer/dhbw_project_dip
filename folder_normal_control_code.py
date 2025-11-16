import os
import cv2
import numpy as np

# --- PFAD ZUM AKTUELLEN NORMAL-ORDNER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_FOLDER = os.path.join(BASE_DIR, "output", "sorted_results", "Normal")

# --- v17: get_features MIT LAPLACE-FILTER ---
def get_features(img):
    # --- A. GEOMETRIE (Standard) ---
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
    
    dark_spots = (v < 67) & (v > 10)
    red_pen = (h > 160) & (s > 60) & (v > 20)
    error_pixels = cv2.bitwise_or(dark_spots.astype(np.uint8), red_pen.astype(np.uint8))
    
    color_score = 0.0
    if np.sum(mask_cookie) > 0:
        color_score = cv2.countNonZero(error_pixels) / np.sum(mask_cookie)

    # --- C. NEU: LAPLACE (Kratzer-Detektor) ---
    # Wir filtern das Graubild, um Kanten/Kratzer zu finden
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Wir messen die Varianz (Streuung) des Bildes, ABER nur innerhalb der Keks-Maske
    # Eine hohe Varianz bedeutet viele Kanten/Kratzer.
    (mean, stddev) = cv2.meanStdDev(laplacian, mask=thresh)
    laplace_variance = stddev[0][0] ** 2

    # Rückgabe mit 6 Werten (Laplace statt Blackhat)
    return circularity, solidity, color_score, pixel_area, hole_count, laplace_variance

def spy():
    if not os.path.exists(NORMAL_FOLDER):
        print("❌ Normal-Ordner nicht gefunden!")
        return

    files = [f for f in os.listdir(NORMAL_FOLDER) if f.lower().endswith(".png")]
    
    anomalies = []
    normals = []

    print(f"🕵️‍♂️ Untersuche {len(files)} Bilder im Ordner 'Normal' mit Laplace-Filter...\n")
    print(f"{'DATEINAME':<35} | {'COLOR':<8} | {'LAPLACE (Kratzer)':<18}")
    print("-" * 85)

    for filename in files:
        path = os.path.join(NORMAL_FOLDER, filename)
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: continue

        # 6 Werte empfangen
        circ, solid, color, area, holes, laplace_var = get_features(img)
        
        if "Anomaly" in filename:
            anomalies.append((color, laplace_var))
            # Drucke die U-Boote
            print(f"🚨 {filename:<32} | {color:.5f}  | {laplace_var:<18.2f}")
        else:
            normals.append((color, laplace_var))

    print("-" * 85)
    print(f"Gefundene Anomalien im Normal-Ordner: {len(anomalies)}")
    
    if len(normals) > 0:
        # Durchschnittswerte der GUTEN Kekse
        n_color = np.mean([x[0] for x in normals])
        n_laplace = np.mean([x[1] for x in normals])
        
        print("\n✅ REFERENZ-WERTE (Durchschnitt der echten Normalen):")
        print(f"   Color:       {n_color:.5f}")
        print(f"   Laplace Var: {n_laplace:.2f}")
        
    if len(anomalies) > 0:
        # Durchschnittswerte der SCHLECHTEN Kekse
        a_color = np.mean([x[0] for x in anomalies])
        a_laplace = np.mean([x[1] for x in anomalies])
        
        print("\n🚨 DURCHSCHNITT der U-Boote (im Normal-Ordner):")
        print(f"   Color:       {a_color:.5f}")
        print(f"   Laplace Var: {a_laplace:.2f}")

if __name__ == "__main__":
    spy()