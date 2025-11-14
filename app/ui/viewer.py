"""Tkinter user interface for browsing inspection results."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Dict, List, Tuple

from PIL import Image, ImageTk, UnidentifiedImageError

from .. import config
from ..models.records import ProcessRecord
from ..utils.logger import get_log_history, register_listener

THUMB_SIZE = (160, 160)
ImageSlotResolver = Callable[[ProcessRecord], Tuple[str | None, str | None]]
DETAIL_IMAGE_SLOTS: List[Tuple[str, str, ImageSlotResolver]] = [
    ("original_raw", "Original (roh)", lambda rec: (rec.original_path, None)),
    ("original_masked", "Original (maskiert)", lambda rec: (rec.inspection_paths.get("original"), None)),
    ("blurred", "Blur (maskiert)", lambda rec: (rec.inspection_paths.get("blurred"), None)),
    ("raw_mask", "Maske roh", lambda rec: (rec.inspection_paths.get("raw_mask"), None)),
    ("mask_clean", "Maske gereinigt", lambda rec: (rec.inspection_paths.get("mask"), None)),
    (
        "mask_overlay",
        "Masken-Overlay",
        lambda rec: (
            rec.inspection_paths.get("mask_overlay") or rec.inspection_paths.get("mask"),
            None,
        ),
    ),
    ("cropped", "Segmentierter Ausschnitt", lambda rec: (rec.inspection_paths.get("cropped"), None)),
    ("classified", "Klassifiziertes Ergebnis", lambda rec: (rec.classified_path, None)),
    ("classified_masked", "Ergebnis mit Maske", lambda rec: (rec.classified_path, rec.inspection_paths.get("mask"))),
]


def _format_hashtag(value: str) -> str:
    normalized = value.strip().lower().replace("/", " ")
    slug = "-".join(part for part in normalized.split() if part)
    return f"#{slug or 'tag'}"


def launch_viewer(records: List[ProcessRecord] | None = None) -> "PipelineViewer":
    viewer = PipelineViewer(records or [])
    return viewer


class PipelineViewer(tk.Tk):
    def __init__(self, records: List[ProcessRecord]) -> None:
        super().__init__()
        self.title("Fryum Inspection Dashboard")
        self.geometry("1400x900")
        self.records = records
        self.log_history = get_log_history()
        self.grouped_records: Dict[str, List[ProcessRecord]] = {}
        self.tab_widgets: Dict[str, Dict[str, object]] = {}
        self.notebook: ttk.Notebook | None = None
        self.status_var = tk.StringVar(value="Pipeline wird ausgefuehrt...")
        self._build_ui()
        self._refresh_tabs()

    def _init_state(self) -> None:
        self.log_history = get_log_history()
        grouped: Dict[str, List[ProcessRecord]] = {"Alle": []}
        for cls in config.TARGET_CLASSES:
            grouped[cls] = []
        for rec in self.records:
            grouped["Alle"].append(rec)
            categories = self._record_categories(rec)
            if not categories:
                grouped.setdefault(rec.prediction, []).append(rec)
                continue
            for category in categories:
                grouped.setdefault(category, []).append(rec)
        self.grouped_records = grouped

    def update_records(self, records: List[ProcessRecord]) -> None:
        self.records = records
        self.status_var.set("Pipeline abgeschlossen." if records else "Keine Daten gefunden.")
        self._refresh_tabs()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        log_frame = ttk.Frame(paned)
        paned.add(log_frame, weight=1)
        self._build_log_panel(log_frame)

        content_frame = ttk.Frame(paned)
        paned.add(content_frame, weight=3)
        status_label = ttk.Label(content_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=tk.X, padx=5, pady=3)
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self._refresh_tabs()

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Pipeline-Log").pack(anchor="w")
        text = tk.Text(parent, height=12, wrap="none")
        scroll = ttk.Scrollbar(parent, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        for line in self.log_history:
            text.insert(tk.END, line + "\n")
        text.see(tk.END)
        text.configure(state=tk.DISABLED)

        def _append(line: str) -> None:
            text.configure(state=tk.NORMAL)
            text.insert(tk.END, line + "\n")
            text.see(tk.END)
            text.configure(state=tk.DISABLED)

        register_listener(_append)

    def _refresh_tabs(self) -> None:
        if self.notebook is None:
            return
        self._init_state()
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        self.tab_widgets.clear()
        for tab in ["Alle", *config.TARGET_CLASSES]:
            records = self.grouped_records.get(tab, [])
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=f"{tab} ({len(records)})")
            widgets = self._build_tab(frame, records, tab)
            self.tab_widgets[tab] = widgets

    def _build_tab(self, parent: ttk.Frame, records: List[ProcessRecord], tab_name: str) -> Dict[str, object]:
        container = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(container, width=320)
        detail_frame = ttk.Frame(container)
        container.add(list_frame, weight=1)
        container.add(detail_frame, weight=3)

        widgets = self._build_detail_panel(detail_frame)

        summary_frame = ttk.LabelFrame(list_frame, text="Aufteilung", padding=5)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        summary_tree = ttk.Treeview(
            summary_frame,
            columns=("segment", "count"),
            show="headings",
            height=6,
            selectmode="browse",
        )
        summary_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        summary_tree.heading("segment", text="Segment")
        summary_tree.heading("count", text="Bilder")
        summary_tree.column("segment", anchor="w", stretch=True, width=200)
        summary_tree.column("count", anchor="center", stretch=False, width=70)

        summary_scroll = ttk.Scrollbar(summary_frame, command=summary_tree.yview)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        summary_tree.configure(yscrollcommand=summary_scroll.set)

        listbox_frame = ttk.LabelFrame(list_frame, text="Bilder", padding=5)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        listbox = tk.Listbox(listbox_frame)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(listbox_frame, command=listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=list_scroll.set)

        sections = self._build_tag_sections(tab_name, records)
        section_map = {section["key"]: section for section in sections}
        for section in sections:
            summary_tree.insert(
                "",
                tk.END,
                iid=section["key"],
                values=(section["label"], len(section["records"])),
            )

        current_subset: List[ProcessRecord] = []

        def _populate_listbox(subset: List[ProcessRecord]) -> None:
            nonlocal current_subset
            current_subset = subset
            listbox.delete(0, tk.END)
            for rec in subset:
                listbox.insert(tk.END, self._format_record_entry(rec))
            if subset:
                listbox.selection_clear(0, tk.END)
                listbox.selection_set(0)
                self._update_detail(widgets, subset[0])
            else:
                listbox.selection_clear(0, tk.END)

        def _on_summary_select(_event: tk.Event) -> None:
            selection = summary_tree.selection()
            if not selection:
                return
            key = selection[0]
            section = section_map.get(key)
            if not section:
                return
            _populate_listbox(section["records"])

        def _on_list_select(_event: tk.Event) -> None:
            if not listbox.curselection():
                return
            idx = listbox.curselection()[0]
            if idx < len(current_subset):
                self._update_detail(widgets, current_subset[idx])

        summary_tree.bind("<<TreeviewSelect>>", _on_summary_select)
        listbox.bind("<<ListboxSelect>>", _on_list_select)

        if sections:
            summary_tree.selection_set(sections[0]["key"])
            _populate_listbox(sections[0]["records"])

        return {
            "summary_tree": summary_tree,
            "listbox": listbox,
            "records": records,
            "sections": sections,
            **widgets,
        }

    def _build_detail_panel(self, parent: ttk.Frame) -> Dict[str, object]:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        grid_frame = ttk.Frame(parent)
        grid_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        grid_frame.columnconfigure((0, 1, 2), weight=1)

        grid_labels: Dict[str, tk.Label] = {}
        for idx, (slot_key, title, _resolver) in enumerate(DETAIL_IMAGE_SLOTS):
            row = idx // 3
            col = idx % 3
            label = tk.Label(grid_frame, text=title, bd=1, relief=tk.SUNKEN, compound="top", cursor="hand2")
            label.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            grid_frame.rowconfigure(row, weight=1)
            grid_labels[slot_key] = label

        meta_var = tk.StringVar()
        meta_label = ttk.Label(parent, textvariable=meta_var, justify=tk.LEFT)
        meta_label.grid(row=1, column=0, sticky="ew", padx=5, pady=2)

        steps = ttk.Treeview(parent, columns=("step", "status", "details"), show="headings", height=5)
        steps.heading("step", text="Schritt")
        steps.heading("status", text="Status")
        steps.heading("details", text="Details")
        steps.column("step", width=160, anchor="w")
        steps.column("status", width=120, anchor="center")
        steps.column("details", anchor="w")
        steps.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        return {
            "grid_labels": grid_labels,
            "meta_var": meta_var,
            "steps": steps,
        }

    def _record_tags(self, record: ProcessRecord) -> List[str]:
        tags = getattr(record, "tags", [])
        normalized: List[str] = []
        for tag in tags:
            if not tag:
                continue
            normalized.append(str(tag).strip().lower())
        return normalized

    def _record_categories(self, record: ProcessRecord) -> set[str]:
        categories: set[str] = set()
        for tag in self._record_tags(record):
            category = config.TAG_CATEGORY_LOOKUP.get(tag)
            if category:
                categories.add(category)
        if not categories and record.prediction:
            categories.add(record.prediction)
        return categories

    def _format_record_entry(self, record: ProcessRecord) -> str:
        tags = [self._format_tag_display(tag) for tag in self._record_tags(record)]
        tag_text = " ".join(tags) if tags else "–"
        return f"{record.filename} ({record.prediction})  {tag_text}"

    @staticmethod
    def _format_tag_display(tag: str) -> str:
        category = config.TAG_CATEGORY_LOOKUP.get(tag, "")
        hashtag = _format_hashtag(tag)
        return f"{hashtag}" if not category else f"{hashtag}[{category}]"

    def _build_tag_sections(self, tab_name: str, records: List[ProcessRecord]) -> List[Dict[str, object]]:
        sections: List[Dict[str, object]] = [
            {"key": f"{tab_name}-all", "label": f"Alle ({len(records)})", "records": records},
        ]
        if not records:
            return sections

        ordered_tags = list(config.CATEGORY_TAGS["Alle"])
        preferred = [tag for tag in config.CATEGORY_TAGS.get(tab_name, ()) if tag not in ordered_tags]
        ordered_tags.extend(preferred)

        seen = set()
        for tag in ordered_tags:
            tagged_records = [rec for rec in records if tag in self._record_tags(rec)]
            if not tagged_records:
                continue
            seen.add(tag)
            label = config.TAG_DISPLAY_NAMES.get(tag, tag)
            sections.append(
                {
                    "key": f"{tab_name}-{tag}",
                    "label": f"{label} ({len(tagged_records)})",
                    "records": tagged_records,
                }
            )

        # ensure category-specific tags appear even if count 0 (for completeness)
        for tag in config.CATEGORY_TAGS.get(tab_name, ()):
            if tag in seen:
                continue
            label = config.TAG_DISPLAY_NAMES.get(tag, tag)
            sections.append(
                {
                    "key": f"{tab_name}-{tag}",
                    "label": f"{label} (0)",
                    "records": [],
                }
            )
        return sections

    def _update_detail(self, widgets: Dict[str, object], record: ProcessRecord) -> None:
        meta_text = (
            f"Datei: {record.filename}\n"
            f"Klassifikation: {record.prediction} (Soll: {record.target})\n"
            + "\n".join(f"{key}: {value:.3f}" for key, value in record.metrics.items())
        )
        widgets["meta_var"].set(meta_text)

        for slot_key, title, resolver in DETAIL_IMAGE_SLOTS:
            label: tk.Label = widgets["grid_labels"][slot_key]  # type: ignore[assignment]
            path, override_mask = resolver(record)
            if path:
                applied_mask = override_mask if override_mask is not None else None
                self._set_image(label, path, THUMB_SIZE, applied_mask)
                label.configure(text=title)
            else:
                label.configure(text=f"{title}\n(nicht verfuegbar)", image="")
                label.image = None
                label.unbind("<Button-1>")

        steps_tree: ttk.Treeview = widgets["steps"]  # type: ignore[assignment]
        for row in steps_tree.get_children():
            steps_tree.delete(row)
        for step in record.process_steps:
            steps_tree.insert("", tk.END, values=(step["title"], step["status"], step["details"]))

    def _set_image(
        self,
        label: tk.Label,
        path: str,
        size: tuple[int, int],
        mask_path: str | None = None,
    ) -> None:
        img = _load_image(path, size, mask_path)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo

        def _open_full(_event: tk.Event) -> None:
            self._show_full_image(path)

        label.bind("<Button-1>", _open_full)

    def _show_full_image(self, path: str) -> None:
        top = tk.Toplevel(self)
        top.title(Path(path).name)
        img = _load_image(path)
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(top, image=photo)
        label.image = photo
        label.pack()


def _load_image(path: str, size: tuple[int, int] | None = None, mask_path: str | None = None) -> Image.Image:
    try:
        image = Image.open(path)
    except (FileNotFoundError, UnidentifiedImageError):
        image = Image.new("RGB", size or (400, 400), color="gray")
    if mask_path:
        try:
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != image.size:
                mask_img = mask_img.resize(image.size, Image.NEAREST)
            background = Image.new("RGB", image.size, color="black")
            image = Image.composite(image, background, mask_img)
        except (FileNotFoundError, UnidentifiedImageError):
            pass
    if size:
        image = image.resize(size, Image.LANCZOS)
    return image
