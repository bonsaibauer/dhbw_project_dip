# 03_classification.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `load_json(path, error_msg)` | Generischer JSON-Loader. | Liest Dateien wie `path.json` oder `classification.json`, meldet fehlende Files mit einer erklärenden `FileNotFoundError` und liefert andernfalls das geparste Objekt. |
| `load_path_config()` | Cached Loader für die Pfadkonfiguration. | Ruft `load_json` für `path.json` auf und cached das Ergebnis, damit Pfade nur einmal von der Festplatte gelesen werden. |
| `class_config()` | Cached Loader für `classification.json`. | Stellt Label-Mapping, Prioritäten und Regeldefinitionen zentral bereit. |
| `norm_path(path_value)` | Normalisiert Pfade. | Nutzt `os.path.normpath`, um Pfadangaben aus der Konfiguration in ein OS-kompatibles Format zu bringen. |
| `read_path_section(cfg_name)` | Greift auf Abschnitte aus `path.json` zu. | Dient `load_paths`, um z. B. den `paths`-Block abzurufen; fehlende Einträge liefern `{}`. |
| `load_paths()` | Baut das Pfad-Lexikon. | Normalisiert alle Werte im `paths`-Abschnitt und speichert sie in einem Dictionary (für CSV, Sortierordner etc.). |
| `map_labels()` | Liest das Label-zu-Klasse-Mapping. | Stellt ein Dictionary `label -> Klassenbezeichnung` bereit, damit Regelresultate in Pipeline-Klassennamen übersetzt werden. |
| `rank_labels()` | Liest die Prioritäten der Labels/Klassen. | Liefert ein Dictionary, das zur Sortierung konkurrierender Entscheidungen verwendet wird (niedrige Zahlen = hohe Priorität). |
| `rule_list()` | Zugriff auf sämtliche Regeldefinitionen. | Gibt die `label_rules`-Liste aus der Konfiguration zurück, sodass die Engine jede Regel prüfen kann. |
| `read_rows(csv_path)` | Lädt die Pipeline-CSV. | Öffnet die Datei, liest sie mit `DictReader`, sammelt alle Zeilen und Fieldnames und informiert über fehlende Dateien. |
| `ensure_cols(fieldnames, required)` | Ergänzt fehlende CSV-Spalten. | Fügt neue Spaltennamen wie `target_label` oder `reason` hinzu, falls sie nicht bereits existieren. |
| `write_rows(csv_path, fieldnames, rows)` | Persistiert die aktualisierte CSV. | Schreibt Header und alle Zeilen mithilfe eines `DictWriter`, nachdem die Klassifikationsergebnisse hinzugefügt wurden. |
| `show_progress(prefix, current, total, bar_len)` | Visualisiert den Klassifikationsfortschritt. | Rendert einen einzeiligen Balken für lange CSVs, sofern mindestens eine Zeile existiert. |
| `parse_flag(value)` | Wandelt flexible Eingaben in boolsche Werte. | Akzeptiert Bool, Strings oder Zahlen und interpretiert `"1"`, `"true"`, `"yes"` (case-insensitive) als `True`, alles andere als `False`. |
| `parse_float(value, default)` | Robuste Fließkommakonvertierung. | Versucht `float(value)` und fällt bei Fehlern auf den angegebenen Default zurück. |
| `parse_int(value, default)` | Robuste Integerkonvertierung. | Castet Werte (ggf. via float) zu `int` oder nutzt einen Default, wenn dies nicht möglich ist. |
| `window_size_variance_score(areas, sensitivity)` | Bewertet die Gleichmäßigkeit von Fensterflächen. | Berechnet Mittelwert/Std-Abweichung der Flächenliste, ermittelt daraus einen normierten Score zwischen 0 und 100 und dämpft ihn über den Sensitivitätsfaktor. |
| `extract_metrics(row)` | Erstellt den Feature-Vektor für eine CSV-Zeile. | Parsed Fensterflächen/-anzahl, Mittellochstatus, Geometrie-/Farb-/Symmetrieparameter, berechnet abgeleitete Kennzahlen (z. B. Varianzscore, Hull-Ratio) und liefert ein Dictionary, das die Regel-Engine nutzt. |
| `match_metric(value, condition)` | Prüft, ob ein Feature eine Bedingung erfüllt. | Unterstützt Operatoren wie `>=`, `<=`, `between`, `in` etc. und bildet damit die flexible Grundlage für Regelbedingungen. |
| `score_rule(rule, features)` | Bewertet eine einzelne Regel. | Startet mit `base_score`, iteriert sämtliche Bedingungen, addiert Gewichte bei erfüllten Kriterien, sammelt optionale Begründungstexte und gibt bei Erreichen von `min_score` eine Entscheidung zurück. |
| `eval_rules(features)` | Führt alle konfigurierten Regeln aus. | Ruft `score_rule` für jede Regel auf, konvertiert Labels zum angezeigten Klassennamen und sammelt alle positiven Entscheidungen in einer Liste. |
| `pick_decision(decisions)` | Wählt die beste Regelentscheidung. | Sortiert Entscheidungen nach Score (absteigend), danach nach Priorität (gemäß `rank_labels`) und Labelname, um eine deterministische Auswahl zu treffen. |
| `fallback_pick(features)` | Liefert eine Standardklasse, falls keine Regel greift. | Gibt „rest“ oder „normal“ (abhängig vom Anomalie-Flag) inklusive Klasse/Score/Begründung zurück. |
| `format_reason(decision)` | Formatiert die Begründung für die CSV. | Kombiniert den Titel-Case des Labels mit der hinterlegten Reason oder einem Standardtext, um verständliche Strings wie „Bruch: Fensterabweichung …“ zu erzeugen. |
| `classify_row(row)` | Klassifiziert eine einzelne CSV-Zeile. | Extrahiert Features, evaluiert Standard- und Bruchregeln, wählt die beste Entscheidung (oder Fallback) und liefert Entscheidung plus Feature-Vektor (z. B. für weiterführende Logik) zurück. |
| `bruch_decisions(features)` | Ergänzt Bruch-spezifische Entscheidungsregeln. | Prüft Anomalie-Bilder mit geringer Spotfläche auf Außen- und Innenbruch via Radien-/Fensterabweichungsmetriken, erzeugt ggf. mehrere Entscheidungen mit hohen Scores und begründet diese textlich. |
| `classify_csv(csv_path, sort_log)` | Stapelt den Klassifikationsprozess über die ganze CSV. | Lädt Zeilen, ruft `classify_row` pro Eintrag auf, schreibt Label/Klasse/Reason/Varianzscore zurück in die CSV, sammelt Vorhersagen und zeigt optional einen Fortschrittsbalken plus Abschlusszeile an. |
| `classify_cli()` | CLI-Einstieg in die Klassifikation. | Startet `classify_csv` mit den Pfaden aus der Konfiguration und dem globalen Logging-Flag und bildet damit die dritte Pipeline-Stufe. |

## Konfiguration: classification.json

`config/classification.json` definiert Prioritäten, Label-Mapping und Regelkatalog.

### Operator-Referenz

| Operator | Bedeutung | Beispiel |
| --- | --- | --- |
| `==` / `!=` | Wert muss gleich bzw. verschieden sein. | `pipeline_has_anomaly_flag == true` prüft, ob das Bild als Anomalie markiert ist. |
| `>=` / `<=` | Wert muss größer/gleich bzw. kleiner/gleich sein. | `color_spot_area >= 80` markiert große Flecken. |
| `between` | Wert liegt innerhalb eines Intervalls (inklusive). | `color_spot_area between 5 und 80` erkennt moderate Kratzer. |

### label_priorities

| Label | Priorität | Bedeutung |
| --- | --- | --- |
| `normal` | 0 | Normal-Fälle haben höchste Priorität und dürfen andere Regeln verdrängen. |
| `farbfehler` | 1 | Farbfehler werden vor Bruch/Rest behandelt. |
| `bruch` | 2 | Bruchregeln greifen vor Rest. |
| `rest` | 3 | Rest-Kategorien haben den geringsten Vorrang. |

### label_class_map

| Label | Zielklasse | Kommentar |
| --- | --- | --- |
| `normal` | Normal | Saubere Snacks mit korrekter Geometrie. |
| `different colour spot`, `similar colour spot`, `burnt`, `farbfehler` | Farbfehler | Verschiedene Farbheuristiken mit spezifischen Grenzwerten. |
| `middle breakage`, `corner or edge breakage`, `bruch` | Bruch | Innen- und Außenbruchindikatoren. |
| `fryum stuck together`, `small scratches`, `other`, `fragment`, `rest` | Rest | Sammeln Fragmente, Doppelstrukturen und sonstige Anomalien. |

### label_rules

Jede Regel besitzt einen `min_score` und mehrere Bedingungen (`metric`, `op`, Grenzwert). Erfüllte Bedingungen addieren ihr `weight` zum Score.

#### fragment (min_score = 3.0)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `geometry_fragment_count` | `>=` | 1 | Mindestens ein Fragment vorhanden. |
| `geometry_outer_contour_count` | `>=` | 3 | Mehrere Außenkonturen deuten auf Bruchstücke hin. |
| `geometry_total_hole_count` | `>=` | 8 | Zu viele Löcher weisen auf zerstörte Strukturen hin. |

#### fryum stuck together (min_score = 2.5)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `geometry_window_area_ratio` | `>=` | 2.1 | Fensterflächenverhältnis stark erhöht. |
| `geometry_window_area_avg` | `<=` | 3800 | Durchschnittliche Fensterfläche sinkt. |
| `geometry_window_size_variance_score` | `<=` | 85 | Varianzscore fällt ab. |
| `geometry_outer_contour_count` | `>=` | 2 | Mehrere Außenkonturen bestätigen Doppelstrukturen. |

#### small scratches (min_score = 2.0)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `color_spot_area` | `between` | 5 – 80 | Kleine Flecken. |
| `color_texture_stddev` | `between` | 10 – 16 | Moderate Texturänderung. |
| `geometry_window_size_variance_score` | `>=` | 93 | Fenster bleiben gleichmäßig. |
| `geometry_window_area_ratio` | `<=` | 1.3 | Nur leichte Geometrieabweichungen. |

#### other (min_score = 2.0)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `pipeline_has_anomaly_flag` | `==` | true | Nur Anomalie-Bilder erlaubt. |
| `geometry_window_area_ratio` | `between` | 1.4 – 2.2 | Fensterverhältnis in auffälligem Bereich. |
| `geometry_window_size_variance_score` | `between` | 82 – 90 | Mittelmäßiger Varianzscore. |

#### rest (min_score = 2.0)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `geometry_has_primary_object` | `==` | false | Kein Hauptobjekt. |
| `geometry_total_hole_count` | `!=` | 7 | Falsche Lochanzahl. |
| `geometry_window_size_variance_score` | `<=` | 82 | Fenster stark ungleichmäßig. |
| `geometry_window_area_ratio` | `>=` | 2.0 | Fensterflächen zu unterschiedlich. |

#### middle breakage (min_score = 2.5)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `geometry_total_hole_count` | `<=` | 6 | Fenster fehlen (Innenbruch). |
| `geometry_window_area_ratio` | `>=` | 1.5 | Fensterflächen streuen. |
| `geometry_window_size_variance_score` | `<=` | 90 | Varianzscore sinkt. |
| `geometry_window_area_deviation_max` | `>=` | 0.15 | Mindestens ein Fenster weicht deutlich ab. |
| `geometry_window_area_deviation_count` | `>=` | 1 | Anzahl abweichender Fenster ≥ 1. |

#### corner or edge breakage (min_score = 2.0)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `geometry_edge_segment_count` | `<=` | 6 | Wenige Kontursemente → Außenbruch. |
| `geometry_edge_damage_ratio` | `<=` | 0.92 | Konvexität weist auf Kantenverlust hin. |
| `geometry_window_size_variance_score` | `>=` | 90 | Fenster stabil → Bruch liegt außen. |
| `geometry_window_area_ratio` | `<=` | 1.4 | Fensterverhältnis bleibt normal. |
| `geometry_outer_radius_min_ratio` | `<=` | 0.83 | Minimaler Außenradius fällt ab. |
| `geometry_outer_radius_low_fraction` | `>=` | 0.08 | Großer Anteil der Kontur < 75 % des Medianradius. |

#### bruch (min_score = 2.5)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `geometry_window_size_variance_score` | `<=` | 82 | Fenster sehr ungleichmäßig. |
| `geometry_total_hole_count` | `<=` | 6 | Fehlende Fenster. |
| `geometry_window_area_ratio` | `>=` | 2.0 | Fensterflächen streuen stark. |
| `geometry_outer_radius_min_ratio` | `<=` | 0.80 | Außenradius bricht ein. |
| `geometry_outer_radius_low_fraction` | `>=` | 0.06 | Großer Anteil der Kontur unter 75 %. |
| `geometry_window_area_deviation_max` | `>=` | 0.18 | Fensterabweichung stark erhöht. |

#### burnt (min_score = 2.0)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `color_issue_detected` | `==` | true | Farbfehler erkannt. |
| `color_spot_area` | `>=` | 350 | Sehr große dunkle Flecken. |
| `color_dark_delta` | `>=` | 32 | Stark erhöhter Dunkelanteil. |
| `color_texture_stddev` | `>=` | 20 | Hohe Texturstreuung durch Verbrennung. |

#### different colour spot (min_score = 2.5)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `color_issue_detected` | `==` | true | Farbabweichung vorhanden. |
| `color_spot_area` | `>=` | 80 | Fleckfläche signifikant. |
| `color_lab_stddev` | `>=` | 5.2 | LAB-Stddev hoch (starker Farbstich). |
| `color_dark_delta` | `<=` | 34 | Dunkelanteil bleibt unter verbranntem Niveau. |
| `color_texture_stddev` | `<=` | 18 | Texturvariation moderat. |

#### similar colour spot (min_score = 2.0)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `color_issue_detected` | `==` | true | Farbabweichung erkannt. |
| `color_spot_area` | `>=` | 60 | Fleckfläche sichtbar. |
| `color_texture_stddev` | `>=` | 14 | Textur streut leicht. |
| `color_lab_stddev` | `<=` | 6.2 | LAB-Abweichung begrenzt – ähnliche Farbe. |
| `color_dark_delta` | `<=` | 35 | Dunkelanteil bleibt moderat. |

#### farbfehler (min_score = 1.5)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `color_issue_detected` | `==` | true | Allgemeine Farbabweichung. |
| `color_spot_area` | `>=` | 40 | Flecken über Grundschwelle. |
| `geometry_window_size_variance_score` | `>=` | 85 | Fenstergeometrie weitgehend stabil – Fokus auf Farbe. |

#### normal (min_score = 4.5)

| Metric | Operator | Grenzwert(e) | Bedeutung |
| --- | --- | --- | --- |
| `pipeline_has_anomaly_flag` | `==` | false | Kein Anomalie-Marker. |
| `color_issue_detected` | `==` | false | Keine Farbabweichung. |
| `geometry_fragment_count` | `==` | 0 | Keine Fragmente. |
| `geometry_total_hole_count` | `==` | 7 | Exakt sieben Fenster. |
| `geometry_window_size_variance_score` | `>=` | 90 | Fenster sehr gleichmäßig. |
| `geometry_window_area_ratio` | `<=` | 1.6 | Fensterflächen liegen eng zusammen. |
