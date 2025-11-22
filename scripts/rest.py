import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURATION
# ==========================================
# Schwellenwert für die Summe der Kantenpixel im BINÄRBILD.
MAX_EDGE_SUM = 3031 
MIN_EDGE_SUM = 2740

# NEU: Minimale Größe in Pixeln, damit ein Objekt als "Hauptobjekt" zählt.
# Alles darunter wird als "Artefakt/Krümel" ignoriert.
MIN_OBJECT_AREA = 250  

def remove_small_artifacts(binary_img, min_area):
    """
    Entfernt alle weißen Bereiche (Objekte), die kleiner als min_area sind.
    Gibt das bereinigte Binärbild zurück.
    """
    # Finde alle zusammenhängenden Komponenten (Blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    # Erstelle eine leere Maske
    clean_binary = np.zeros_like(binary_img)
    
    # Iteriere über alle gefundenen Objekte (start bei 1, da 0 der Hintergrund ist)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            # Wenn das Objekt groß genug ist, kopiere es in das saubere Bild
            clean_binary[labels == i] = 255
            
    return clean_binary

def calculate_edge_sum(image):
    """
    Berechnet Kantenlänge und gibt zusätzlich das Binärbild zurück.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Glättung (Morphology)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Canny auf der Maske
    edges = cv2.Canny(binary, 50, 150)
    total_edge_length = cv2.countNonZero(edges)

    # WICHTIG: Wir geben jetzt auch 'binary' zurück, um es später filtern zu können
    return total_edge_length, edges, binary

def create_edge_report(image_data, output_file="complexity_report.png"):
    """
    Angepasster Bericht, der Original vs. Bereinigte Version zeigt.
    image_data ist eine Liste von Tupeln: (pfad, edges_original, edges_clean)
    """
    num_images = min(5, len(image_data))
    if num_images == 0: return

    print(f"[rest.py] Erstelle Bericht für {num_images} Bilder...")
    
    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    if num_images == 1: 
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, (img_path, edges_orig, edges_clean) in enumerate(image_data[:num_images]):
        # ZEILE 1: Originale Kanten (mit Artefakten)
        axes[0, idx].imshow(edges_orig, cmap='gray')
        axes[0, idx].set_title(f"Original\n{os.path.basename(img_path)}", fontsize=8)
        axes[0, idx].axis('off')
        
        # ZEILE 2: Bereinigte Kanten (nur Hauptobjekte)
        axes[1, idx].imshow(edges_clean, cmap='gray')
        axes[1, idx].set_title("Bereinigt (Ohne Artefakte)", fontsize=8)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    plt.close()
    print(f"[rest.py] Bericht gespeichert: {output_file}")

def run_complexity_check(sorted_dir):
    print("\n[rest.py] Starte intelligente Komplexitäts-Prüfung...")
    print(f"   - Limit: {MAX_EDGE_SUM} Kanten-Pixel")
    print(f"   - Artefakt-Filter: Objekte unter {MIN_OBJECT_AREA}px werden ignoriert")

    rest_dir = os.path.join(sorted_dir, "Rest")
    os.makedirs(rest_dir, exist_ok=True)

    check_classes = ["Normal", "Bruch"]
    moved_count = 0
    kept_count = 0

    for cls in check_classes:
        class_path = os.path.join(sorted_dir, cls)
        if not os.path.exists(class_path): continue

        for root, _, files in os.walk(class_path):
            for file_name in files:
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

                file_path = os.path.join(root, file_name)
                image = cv2.imread(file_path)
                if image is None: continue

                # 1. ERSTE PRÜFUNG (Schnell)
                edge_sum, edges_orig, binary_orig = calculate_edge_sum(image)

                # --- FALL A: ZU WENIG KANTEN (Fragment / Halbes Teil) ---
                if edge_sum < MIN_EDGE_SUM:
                    target_path = os.path.join(rest_dir, file_name)
                    shutil.move(file_path, target_path)
                    moved_count += 1
                    print(f"   -> REST (Fragment): {file_name} (Sum: {edge_sum} < {MIN_EDGE_SUM})")
                    continue # Nächstes Bild, wir sind fertig hier

                # --- FALL B: ZU VIELE KANTEN (Chaos / Schmutz / Überlagerung) ---
                elif edge_sum > MAX_EDGE_SUM:
                    # 2. ZWEITE PRÜFUNG (Detail-Check: Sind es nur Artefakte?)
                    
                    # Entferne kleine Objekte aus der Binärmaske
                    binary_clean = remove_small_artifacts(binary_orig, MIN_OBJECT_AREA)
                    
                    # Berechne Kanten auf dem BEREINIGTEN Bild neu
                    edges_clean = cv2.Canny(binary_clean, 50, 150)
                    clean_edge_sum = cv2.countNonZero(edges_clean)

                    # Entscheidung treffen
                    if clean_edge_sum > MAX_EDGE_SUM:
                        # Auch ohne Artefakte noch zu komplex -> Wirklich Rest
                        target_path = os.path.join(rest_dir, file_name)
                        if os.path.exists(target_path):
                            base, ext = os.path.splitext(file_name)
                            target_path = os.path.join(rest_dir, f"{base}_complex{ext}")

                        shutil.move(file_path, target_path)
                        moved_count += 1
                        print(f"   -> REST (Chaos): {file_name} (Clean Sum: {clean_edge_sum})")
                    else:
                        # War nur wegen Krümeln hoch -> Behalten
                        kept_count += 1
                        print(f"   -> BEHALTEN: {file_name} (Original: {edge_sum} -> Clean: {clean_edge_sum})")
                        

    print(f"[rest.py] Fertig. {moved_count} verschoben. {kept_count} vor fälschlicher Verschiebung gerettet.")
