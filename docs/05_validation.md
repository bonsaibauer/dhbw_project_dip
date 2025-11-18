# 05_validation.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `load_json(path, error_msg)` | Allgemeiner JSON-Loader. | Öffnet Dateien wie `path.json` oder `classification.json`, liefert das geparste Objekt und meldet fehlende Dateien via `FileNotFoundError` mit Kontext. |
| `load_config()` | Cached Loader für `path.json`. | Stellt alle Pfade (`paths`-Abschnitt) bereit und hält sie durch `lru_cache` im Speicher. |
| `class_config()` | Cached Loader für `classification.json`. | Macht Label-Prioritäten und Mapping für die Validierungslogik verfügbar. |
| `norm_path(path_value)` | Normalisiert Pfade aus der Konfiguration. | Nutzt `os.path.normpath`, um konsistente Pfade zu erhalten; fehlende Werte werden als Leerstring zurückgegeben. |
| `load_paths()` | Erstellt das Pfad-Lexikon für die Validierung. | Liest den `paths`-Block, normalisiert die Werte und liefert u. a. CSV-, Annotation- und Fehlerordnerpfad. |
| `rank_labels()` | Holt die Label-Prioritäten. | Gibt ein Dictionary `label -> Priorität` zurück, das zur Auflösung mehrdeutiger Annotationen dient. |
| `map_labels()` | Holt das Label-zu-Klasse-Mapping. | Translatiert Annotationstexte (z. B. „bruch“) in Pipeline-Klassen wie „Bruch“. |
| `normalize_path(path)` | Vereinheitlicht Pfade gemäß Annotationen. | Ersetzt Backslashes, entfernt den Präfix `Data/Images/` und liefert relative Pfade ohne führende Slashes, damit Vorhersagen mit Annotationen abgeglichen werden können. |
| `render_table(headers, rows, indent)` | Gibt formatierte Tabellen aus. | Berechnet Spaltenbreiten, zeichnet Header/Divider und listet Zeilen – genutzt für die Übersicht zur Validierungsgenauigkeit. |
| `select_label(raw_label, label_priorities)` | Wählt das relevanteste Label bei Mehrfachangaben. | Zerlegt das Annotationfeld, trimmt Einträge, sortiert sie nach Priorität und gibt den höchstrangigen Labelstring in Kleinbuchstaben zurück. |
| `load_annos(annotation_file, label_priorities, label_class_map)` | Lädt und mappt Annotationen. | Liest die Annotation-CSV, normalisiert Pfade, wählt per `select_label` das wichtigste Label, mappt es auf die Zielklasse und baut ein Dictionary `rel_path -> erwartete Klasse`; fehlende Dateien werden mit einem Hinweis quittiert. |
| `build_chain(label_priorities, label_class_map)` | Erstellt eine Priorisierungskette. | Kombiniert die niedrigste Priorität je Klasse in einen String wie „Bruch > Farbfehler > …“, der später zur Einordnung ausgegeben wird. |
| `copy_miss(pred_entry, expected_label, falsch_dir)` | Kopiert Fehlklassifikationen zur Nachkontrolle. | Erstellt den Fehlerordner, baut einen Dateinamen aus Pfad, Ground-Truth und Prediction und kopiert das betroffene Bild dorthin. |
| `check_preds(predictions, annotations, falsch_dir, label_priorities, label_class_map)` | Vergleicht Pipelinevorhersagen mit Annotationen. | Löscht optional alte Fehlerordner, iteriert Vorhersagen, zählt Treffer/Mismatches pro Klasse, berechnet Gesamt- und Klassengenauigkeit, rendert eine Tabelle, zeigt die Priorisierungskette an und kopiert abweichende Bilder via `copy_miss`; bei fehlenden bzw. nicht passenden Annotationen wird der Vorgang übersprungen. |
| `load_preds(csv_path)` | Lädt Pipeline-Vorhersagen aus der CSV. | Liest `pipeline_csv_path`, filtert Zeilen ohne `target_class` heraus und sammelt relative Pfade, predicted Klassen, Quellpfade und Begründungen für die Validierung. |
| `validate_cli()` | CLI-Einstieg für die Validierung. | Lädt Vorhersagen und Annotationen, ruft `check_preds` mit allen Pfad-/Konfigurationsinformationen auf und bildet damit die abschließende Pipeline-Stufe. |
