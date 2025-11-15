import cv2
import numpy as np
import os

def calculate_shape_features(img):
    """
    Berechnet Rundheit UND Solidität.
    
    Solidität = (Anzahl weißer Pixel) / (Fläche der konvexen Hülle)
    Das ist der entscheidende Wert für deine zerbrochenen Räder!
    """
    # 1. Graustufen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # --- WICHTIG: Echte Masse zählen (Pixel) ---
    pixel_area = cv2.countNonZero(thresh)

    # 2. Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, 0.0

    # Größte Kontur nehmen
    c = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    # 3. Convex Hull (Das Gummiband drumherum)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)

    # --- BERECHNUNGEN ---
    
    # A) Solidität (Fülldichte)
    solidity = 0.0
    if hull_area > 0:
        # Wir teilen die echte Pixel-Masse durch den Umriss
        solidity = float(pixel_area) / hull_area

    # B) Rundheit (Circularity)
    circularity = 0.0
    if perimeter > 0:
        circularity = (4 * np.pi * contour_area) / (perimeter ** 2)
    
    return circularity, solidity

def check_color_anomaly(img):
    """ Sucht nach dunklen/verbrannten Stellen. """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    mask_cookie = v > 10 
    total_pixels = np.sum(mask_cookie)
    
    if total_pixels == 0: return 0.0

    # Dunkle Stellen (aber kein Hintergrund)
    dark_spots = (v < 60) & (v > 10)
    dark_pixel_count = np.sum(dark_spots)
    
    return dark_pixel_count / total_pixels

# --- TEST BEREICH ---
if __name__ == "__main__":
    
    # 1. HIER DATEINAMEN ÄNDERN!
    # Teste einmal einen HEILEN (Normal_...) und einmal den KAPUTTEN (0.873_071.jpg)
    test_file = "Anomaly_058.png" 
    
    # Pfad basteln (sucht in deinem output Ordner)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Falls du das Bild direkt neben das Skript gelegt hast, nutze:
    # img_path = os.path.join(base_dir, test_file)
    
    # Falls es noch im Output-Ordner liegt:
    img_path = os.path.join(base_dir, "output", "test_crops", test_file) 
    # (ACHTUNG: Pfad ggf. anpassen, je nachdem wo das Bild gerade liegt!)

    print(f"🔍 Untersuche: {img_path}")
    
    if os.path.exists(img_path):
        # Sicher laden
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Fehler beim Laden.")
        else:
            # Neue Funktion aufrufen
            circ, solid = calculate_shape_features(img)
            dark_score = check_color_anomaly(img)
            
            print("-" * 30)
            print(f"🟢 Rundheit (Circularity): {circ:.4f}")
            print(f"🔷 Solidität (Fülldichte): {solid:.4f} <--- DAS IST WICHTIG")
            print(f"🔴 Farb-Score (Dunkel):    {dark_score:.4f}")
            print("-" * 30)
            
            cv2.imshow("Analyse", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("❌ Datei nicht gefunden! Pfad prüfen.")