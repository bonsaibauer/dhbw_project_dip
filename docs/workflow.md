# Workflow & Entscheidungsbaum  

Dieses Dokument fasst den gesamten Ablauf in `main.py` zusammen. Jeder Abschnitt beschreibt:
- **Welche Funktionen beteiligt sind** und was sie leisten.
- **Wie der Ablauf** (Reihenfolge, Zwischenschritte) aussieht.
- **Welche Entscheidungsregeln** greifen und auf welchen Parametern sie beruhen.

Alle Angaben beziehen sich auf den heuristischen Modus.

---

## 1. Einstellungen & Ordner vorbereiten

- **Konstanten & Pfade**: `RAW_DATA_DIR`, `OUTPUT_DIR`, `PROCESSED_DATA_DIR`, `SORTED_DATA_DIR`, `FALSCH_DIR`, `ANNOTATION_FILE`.  
- **Parametergruppen**  
  - Segmentierung: `LOWER_GREEN`, `UPPER_GREEN`, `CONTOUR_AREA_MIN`, `WARP_SIZE`, `TARGET_WIDTH`, `TARGET_HEIGHT`.  
  - Geometrie: `EPSILON_FACTOR`, `MIN_HOLE_AREA`, `MIN_WINDOW_AREA`, `MAX_CENTER_HOLE_AREA`, `FRAGMENT_AREA_MIN`, `EDGE_DAMAGE_THRESHOLD`, `EDGE_MAX_SEGMENTS`.  
  - Farbe & Textur: `EROSION_*`, `BLACKHAT_*`, `DEFECT_SPOT_THRESHOLD`, `FINE_*`, `TEXTURE_STD_THRESHOLD`, `LAB_A_STD_THRESHOLD`.  
  - Entscheidungsbaum: `REST_*`, `FARBFEHLER_*`, `BRUCH_SYMMETRY_THRESHOLD`, `SYMMETRY_SENSITIVITY`.  
  - Validierung: `LABEL_PRIORITIES`, `LABEL_CLASS_MAP`, `CLASS_DESCRIPTIONS`.  
- Beim Programmstart werden alle Ausgabeverzeichnisse erzeugt (`os.makedirs(..., exist_ok=True)`), sodass die folgenden Schritte direkt wegschreiben können.

---

## 2. Bildliste & Segmentierung (`collect_image_files`, `prepare_dataset`, `run_preprocessing`)

1. **`collect_image_files(source_dir)`**  
   - listet rekursiv alle `.jpg/.jpeg/.png`. Die Reihenfolge bestimmt später den Fortschritt.

2. **`print_progress(prefix, current, total)`**  
   - einfacher CLI-Balken, der während der Segmentierung aktualisiert wird.

3. **`prepare_dataset(source_dir, target_dir)`**  
   - steuert die komplette Vorverarbeitung: Dateien sammeln, Bilder laden, `run_preprocessing()` aufrufen, Ergebnisse unter `output/processed/<Klasse>/...` speichern.

4. **`run_preprocessing(image, result)`**  
   - Schritte: HSV-Maske (entfernt grünen Hintergrund) → Konturen suchen und filtern (`CONTOUR_AREA_MIN`) → größte Kontur entzerren (`cv2.minAreaRect` + `cv2.getPerspectiveTransform`) → auf `TARGET_WIDTH × TARGET_HEIGHT` skalieren → Ergebnis in `result` ablegen.  
   - Parameter wie `EROSION_ITERATIONS` können oben zentral angepasst werden.

Ergebnis: `output/processed` enthält für jede Quelle (Normal/Anomaly) die segmentierten, ausgerichteten Snacks.

---

## 3. Feature-Ermittlung (Geometrie & Farbanalyse)

1. **`get_contours_hierarchy(image)`**  
   - erzeugt ein Binärbild, sucht Konturen plus Hierarchie → Grundlage für Lochzählung, Fragmente, Fensteranalyse.

2. **`analyze_geometry_features(contours, hierarchy)`**  
   - liefert: Hauptkontur, Fläche, Konvexhülle, `edge_damage`, `edge_segments`, `num_windows`, `window_areas`, `has_center_hole`, `fragment_count`, `outer_count`.  
   - Diese Daten treiben später die Rest-/Bruch-Hints sowie Fenster-Score und Kantenerkennung.

3. **`detect_defects(image, spot_threshold)`**  
   - komplette Farbanalyse: Maskierung → Erosion (`EROSION_*`) → Black-Hat (`BLACKHAT_*`) → Schwellenwert & Rausch-Filter → Kennzahlen (`spot_area`, `texture_std`, `lab_std`, `dark_delta`, `median_intensity`, `is_defective`).  
   - Aus diesen Werten entsteht `color_strength`, das über Farbhints („Farbfehler“) entscheidet.

---

## 4. Sortieren via Entscheidungsbaum (`sort_dataset_manual_rules`)

### 4.1 Ablauf
1. **Verzeichnis vorbereiten** (`sorted_data_dir` leeren, Klasse-Unterordner anlegen).  
2. **Iteration** über `output/processed`: Geometrie- und Farbmerkmale pro Bild berechnen.  
3. **Decision Levels** (siehe unten) bestimmen `target_class` + `reason`.  
4. **Datei kopieren** → `output/sorted/<Klasse>/<Prefix optional>/<Name>`. Bei `Normal` wird standardmäßig `"{symmetry_score:03d}_"` vorangestellt.  
5. **Statistik** (`stats_counter`, `reason_counter`) speisen später die Ausgabe.

### 4.2 Decision Level 1 – Guards
- Kein Objekt (`geo["has_object"] == False`) → `Rest`.  
- Lochanzahl < 7 → potentielle Bruchfälle.  
- Lochanzahl > 7 → sofort `Rest` („zu viele Fragmente“).  
- Lochanzahl = 7 → weiter zu Level 3.

### 4.3 Decision Level 2 – Lochanzahl < 7
1. **Starke Farbe** (`color_strength ≥ 2`) → `Farbfehler`.  
2. **Starker Rest-Hinweis** (`rest_strength ≥ 2`, z. B. Fragmente, mehrere äußere Konturen) → `Rest`.  
3. **Sonst** → `Bruch` („zu wenig Löcher“).  
Hinweis: rein fensterbasierte Hinweise werden auf Stärke 1 begrenzt (keine sofortige Rest-Klassifikation).

### 4.4 Decision Level 3 – Lochanzahl = 7
Reihenfolge der Checks:
1. **Rest** (`rest_strength ≥ 2`).  
2. **Farbe** (`color_strength ≥ 2` oder `rest_strength ≤ 1`).  
3. **Kantenbruch** (`edge_damage ≥ EDGE_DAMAGE_THRESHOLD` oder `edge_segments ≥ EDGE_MAX_SEGMENTS`) und `color_strength < 2`.  
4. **Fenster-Score** (`window_size_variance_score < BRUCH_WINDOW_VARIANCE_THRESHOLD` → `Bruch`, sonst `Normal` mit Prefix).  
Damit entsteht eine klare Priorisierung: klebrige Restfälle vorn, starke Farbe vor Kanten, Kanten vor hohem Fenster-Score.

### 4.5 Weitere Helfer im Umfeld
- `normalize_relative_path(path)` – vereinheitlicht Pfade (wichtig für Logging/Validierung).  
- `resolve_priority_label(raw_label)` – falls nötig bei Mehrfachlabels.  
- `describe_priority_chain()` – liefert den String für die Validierungsanzeige.

---

## 5. Übersichtstabellen (`print_table`)

- Nach Abschluss der Sortierung: Tabelle mit Klasse/Anzahl/Anteil/Beschreibung/Häufigstem Grund.  
- Während `validate_predictions()`: Tabellen für Gesamtstatistik und Klassenübersicht.  
- `print_table()` sorgt für feste Spaltenbreiten, sodass die Ausgaben im Terminal sauber lesbar sind.

---

## 6. Validierung & Fehlbilder

### 6.1 `load_annotations(annotation_file)`
- liest `data/image_anno.csv`, verwendet `normalize_relative_path()` + `resolve_priority_label()` und mappt über `LABEL_CLASS_MAP` auf eine der vier Zielklassen.

### 6.2 `validate_predictions(predictions, annotations, falsch_dir)`
1. Vergleicht jede Prognose (`pred["predicted"]`) mit dem Annotationseintrag.  
2. Baut Klassenstatistiken (`per_class`).  
3. Druckt zwei Tabellen: Gesamtstatistik (Bewertet, Treffer, Genauigkeit, Falsch) und Klassenübersicht (Erwartet, Treffer, Genauigkeit %).  
4. Kopiert Fehlbilder via `copy_misclassified()` nach `output/sorted/Falsch` (Dateiname enthält `gt-...`/`pred-...`).  
- Ergebnis: Der Fehlordner enthält alle Abweichungen inkl. Grund.

---

## 7. Hilfsfunktionen & Logging

- `print_progress(...)` – Fortschrittsanzeige bei der Segmentierung.  
- `collect_image_files(...)` – Inputliste für `prepare_dataset`.  
- `print_table(...)` – Tabellenformatierung.  
- `describe_priority_chain()` – Textausgabe der Label-Prioritäten.  
- `copy_misclassified(...)` – kopiert Fehlbilder mit aussagekräftigem Dateinamen.

---

## 8. Hauptprogramm (Schritte im Überblick)

1. **Konfiguration** laden, `output/` anlegen.  
2. **Segmentierung** (Schritt 2): Rohbilder zu `output/processed`.  
3. **Sortierung** (Schritt 4): Decision Levels -> `output/sorted`.  
4. **Übersicht** (Schritt 5): Tabellen ausgeben.  
5. **Validierung** (Schritt 6): Annotierte Wahrheiten prüfen, Fehlbilder sammeln.  

Damit erfüllt der Code alle Vorgaben: Hintergrund entfernen, in vier Klassen einsortieren, Fehlbilder dokumentieren und Normalbilder nach dem Symmetry-Score prefixed abspeichern - rein heuristisch.
