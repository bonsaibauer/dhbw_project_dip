"""Tkinter user interface for browsing inspection results."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Dict, List

from PIL import Image, ImageTk, UnidentifiedImageError

from .. import config
from ..models.records import ProcessRecord
from ..utils.logger import get_log_history, register_listener

THUMB_SIZE = (160, 160)
MAIN_SIZE = (380, 380)
MASK_SIZE = (320, 320)
INTERMEDIATE_LABELS = {
    "original": "Original",
    "blurred": "Blur",
    "raw_mask": "Maske roh",
    "mask": "Maske gereinigt",
    "mask_overlay": "Maske Overlay",
    "cropped": "Cropped",
    "classified": "Klassifiziert",
}


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
            grouped.setdefault(rec.prediction, []).append(rec)
            grouped["Alle"].append(rec)
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
            widgets = self._build_tab(frame, records)
            self.tab_widgets[tab] = widgets

    def _build_tab(self, parent: ttk.Frame, records: List[ProcessRecord]) -> Dict[str, object]:
        container = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(container, width=250)
        detail_frame = ttk.Frame(container)
        container.add(list_frame, weight=1)
        container.add(detail_frame, weight=3)

        listbox = tk.Listbox(list_frame)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=scrollbar.set)

        for rec in records:
            listbox.insert(tk.END, f"{rec.filename} ({rec.prediction})")

        widgets = self._build_detail_panel(detail_frame)

        def _on_select(event: tk.Event) -> None:
            if not listbox.curselection():
                return
            idx = listbox.curselection()[0]
            self._update_detail(widgets, records[idx])

        listbox.bind("<<ListboxSelect>>", _on_select)
        if records:
            listbox.selection_set(0)
            self._update_detail(widgets, records[0])
        return {"listbox": listbox, "records": records, **widgets}

    def _build_detail_panel(self, parent: ttk.Frame) -> Dict[str, object]:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        image_label = tk.Label(parent, text="Bild", bd=1, relief=tk.SUNKEN)
        image_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        mask_label = tk.Label(parent, text="Maske", bd=1, relief=tk.SUNKEN)
        mask_label.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        meta_var = tk.StringVar()
        meta_label = ttk.Label(parent, textvariable=meta_var, justify=tk.LEFT)
        meta_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

        steps = ttk.Treeview(parent, columns=("step", "status", "details"), show="headings", height=5)
        steps.heading("step", text="Schritt")
        steps.heading("status", text="Status")
        steps.heading("details", text="Details")
        steps.column("step", width=160, anchor="w")
        steps.column("status", width=120, anchor="center")
        steps.column("details", anchor="w")
        steps.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        intermediate_frame = ttk.Frame(parent)
        intermediate_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        return {
            "image_label": image_label,
            "mask_label": mask_label,
            "meta_var": meta_var,
            "steps": steps,
            "intermediate": intermediate_frame,
            "current_images": {},
        }

    def _update_detail(self, widgets: Dict[str, object], record: ProcessRecord) -> None:
        meta_text = (
            f"Datei: {record.filename}\n"
            f"Klassifikation: {record.prediction} (Soll: {record.target})\n"
            + "\n".join(f"{key}: {value:.3f}" for key, value in record.metrics.items())
        )
        widgets["meta_var"].set(meta_text)
        self._set_image(widgets["image_label"], record.classified_path, MAIN_SIZE, widgets)
        mask_path = record.inspection_paths.get("mask_overlay") or record.inspection_paths.get("mask")
        if mask_path:
            self._set_image(widgets["mask_label"], mask_path, MASK_SIZE, widgets)

        steps_tree: ttk.Treeview = widgets["steps"]  # type: ignore[assignment]
        for row in steps_tree.get_children():
            steps_tree.delete(row)
        for step in record.process_steps:
            steps_tree.insert("", tk.END, values=(step["title"], step["status"], step["details"]))

        self._populate_intermediates(widgets["intermediate"], record)

    def _set_image(
        self,
        label: tk.Label,
        path: str,
        size: tuple[int, int],
        widgets: Dict[str, object],
    ) -> None:
        img = _load_image(path, size)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo

        def _open_full(_event: tk.Event) -> None:
            self._show_full_image(path)

        label.bind("<Button-1>", _open_full)

    def _populate_intermediates(self, frame: ttk.Frame, record: ProcessRecord) -> None:
        for child in frame.winfo_children():
            child.destroy()
        paths = record.inspection_paths
        col = 0
        for key, title in INTERMEDIATE_LABELS.items():
            path = paths.get(key)
            if not path:
                continue
            img = _load_image(path, THUMB_SIZE)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(frame, image=photo, compound="top", text=title, cursor="hand2")
            lbl.image = photo
            lbl.grid(row=0, column=col, padx=5, pady=5)
            lbl.bind("<Button-1>", lambda _e, p=path: self._show_full_image(p))
            col += 1

    def _show_full_image(self, path: str) -> None:
        top = tk.Toplevel(self)
        top.title(Path(path).name)
        img = _load_image(path)
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(top, image=photo)
        label.image = photo
        label.pack()


def _load_image(path: str, size: tuple[int, int] | None = None) -> Image.Image:
    try:
        image = Image.open(path)
    except (FileNotFoundError, UnidentifiedImageError):
        image = Image.new("RGB", size or (400, 400), color="gray")
    if size:
        image = image.resize(size, Image.LANCZOS)
    return image
