# 04_sorting.py – Funktionsübersicht

| Funktion | Kurzbeschreibung | Detailbeschreibung |
| --- | --- | --- |
| `load_json(path, error_msg)` | Allgemeiner JSON-Loader. | Öffnet Konfigurationsdateien wie `path.json`, gibt das geparste Objekt zurück und meldet fehlende Dateien mit einer aussagekräftigen Exception. |
| `load_path_config()` | Cached Loader für `path.json`. | Ruft `load_json` genau einmal auf und stellt die Pfaddefinitionen anschließend aus dem Cache bereit. |
| `norm_path(path_value)` | Normalisiert Pfadstrings. | Gibt OS-konforme Pfade oder leere Strings zurück, damit die restliche Logik unabhängig vom Betriebssystem funktioniert. |
| `load_paths()` | Erstellt das Pfad-Mapping. | Liest den `paths`-Abschnitt aus `path.json`, normalisiert alle Werte und liefert u. a. den CSV-Eingabepfad und den Sortieroutput. |
| `render_table(headers, rows, indent)` | Gibt tabellarische Auswertungen aus. | Berechnet Spaltenbreiten, druckt Header, Trenner und Zeilen mit korrekter Ausrichtung und dient hauptsächlich zur Ausgabe der Sortierstatistik. |
| `show_progress(prefix, current, total, bar_len)` | Visualisiert den Kopierfortschritt. | Rendert einen einzeiligen Balken, solange Dateien verschoben werden, und aktualisiert ihn pro Iteration. |
| `read_rows(csv_path)` | Lädt die Pipeline-CSV. | Öffnet die Datei, liest sie mit `DictReader`, sammelt alle Zeilen sowie Header und informiert, falls die Datei fehlt. |
| `ensure_cols(headers, required)` | Ergänzt Kopfzeilen um fehlende Spaltennamen. | Fügt Einträge wie `sorted_path` oder `destination_filename` bei Bedarf hinzu, bevor die CSV erneut geschrieben wird. |
| `write_rows(csv_path, headers, rows)` | Schreibt die aktualisierte CSV zurück. | Nutzt einen `DictWriter`, um Header und Zeilen nach dem Sortierlauf zu persistieren. |
| `try_float(value)` | Konvertiert tolerant zu float. | Liefert `None`, wenn der Wert nicht numerisch interpretierbar ist; notwendig für die optionale Score-Präfixierung. |
| `base_filename(row)` | Ermittelt den Ausgangsdateinamen. | Nimmt bevorzugt `row["filename"]`, fällt sonst auf den Basename des `source_path` zurück, um einen Startpunkt für den Zielnamen zu haben. |
| `prefixed_name(row, base_name, target_class)` | Präfixiert Normal-Klassen mit Symmetriescore. | Gibt für Nicht-Normal-Klassen den Basenamen unverändert zurück; andernfalls wird ein Null-gefüllter `symmetry_score` vorangestellt, sofern der Wert numerisch vorliegt. |
| `resolve_destination_name(row)` | Bestimmt den finalen Dateinamen. | Nutzt einen bereits vorhandenen `destination_filename` oder baut ihn aus Klasseninfo + Symmetriescore-Präfix, sodass spätere Kopierschritte den richtigen Namen verwenden. |
| `clear_folder(folder)` | Löscht den Ausgabeordner robust. | Entfernt vorhandene Strukturen rekursiv und hebt bei Bedarf Schreibschutz auf, um jeden Sortierlauf mit einer leeren Zielhierarchie zu starten. |
| `sort_images(csv_path, sorted_dir, log_progress)` | Kopiert Bilder in Klassenordner und protokolliert Ergebnisse. | Lädt CSV-Zeilen, leert das Ziel, erstellt Standardklassenordner, kopiert jedes Bild anhand `target_class` und Zielnamen, sammelt Zählwerte pro Klasse, zeigt optional Fortschritt, rendert eine Übersichtstabelle und schreibt `sorted_path`/`destination_filename` zurück in die CSV. |
| `sort_cli()` | CLI-Einstieg in die Sortierung. | Ruft `sort_images` mit den konfigurierten Pfaden und dem globalen Logging-Flag auf und bildet damit die vierte Pipeline-Stufe. |
