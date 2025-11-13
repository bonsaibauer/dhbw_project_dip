import os
import cv2
import numpy as np

# --- KONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Wir wollen alle Bilder in data/Images finden (egal ob Normal oder Anomaly)
SEARCH_DIR = os.path.join(BASE_DIR, "data", "Images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "test_crops")

print(f"📂 Arbeitsverzeichnis: {BASE_DIR}")

# --- HILFSFUNKTIONEN ---

def load_image_windows_safe(path):
    """ Lädt Bilder auch mit Sonderzeichen im Pfad (Windows-Fix). """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Ladefehler: {e}")
        return None

def process_image(img_path, show_debug=False):
    """ Schneidet den Keks aus (Hintergrund weg). """
    img = load_image_windows_safe(img_path)
    if img is None: return None

    # 1. Preprocessing (Weichzeichnen + HSV)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. Maske (Grün entfernen)
    lower_green = np.array([30, 40, 40]) 
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_object = cv2.bitwise_not(mask_green)

    # 3. Aufräumen
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_object, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Crop (Größtes Objekt)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500: return None # Zu klein (Rauschen)

    x, y, w, h = cv2.boundingRect(c)
    
    # Bild maskieren (Schwarzer Hintergrund)
    img_masked = cv2.bitwise_and(img, img, mask=mask_clean)

    # Zuschneiden mit kleinem Rand
    pad = 10
    h_img, w_img = img.shape[:2]
    y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
    x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
    
    return img_masked[y1:y2, x1:x2]

def process_all_images():
    print(f"🚀 Starte Suche in: {SEARCH_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Output-Ordner erstellt: {OUTPUT_DIR}")

    image_files = []
    # Rekursiv durchsuchen (findet Normal UND Anomaly)
    for root, dirs, files in os.walk(SEARCH_DIR):
        folder_name = os.path.basename(root) # "Normal" oder "Anomaly"
        for file in files:
            if file.lower().endswith(".jpg"):
                image_files.append((os.path.join(root, file), folder_name))

    total = len(image_files)
    print(f"📦 {total} Bilder gefunden.")
    
    success = 0
    for i, (path, folder) in enumerate(image_files):
        # Fortschrittsanzeige
        if i % 50 == 0: print(f"   Processing {i}/{total}...")
        
        crop = process_image(path)
        
        if crop is not None:
            # Dateiname: "Normal_001.png" oder "Anomaly_050.png"
            orig_name = os.path.splitext(os.path.basename(path))[0]
            save_name = f"{folder}_{orig_name}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            # Speichern (Windows-Safe)
            is_ok, buf = cv2.imencode(".png", crop)
            if is_ok:
                with open(save_path, "wb") as f: f.write(buf)
                success += 1
    
    print("-" * 30)
    print(f"✅ Fertig! {success} von {total} Bildern gespeichert.")
    print(f"👉 Schau in den Ordner: {OUTPUT_DIR}")

# --- DER STARTKNOPF ---
if __name__ == "__main__":
    # Hier wird die Funktion endlich aufgerufen!
    process_all_images()