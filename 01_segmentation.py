import json, os, shutil, stat
from functools import lru_cache

import cv2, numpy as np

base_dir = os.path.dirname(__file__)
path_path = os.path.join(base_dir, "path.json")
seg_path = os.path.join(base_dir, "segmentation.json")


@lru_cache(maxsize=1)
def load_path_config():
    if not os.path.exists(path_path):
        raise FileNotFoundError(f"Pfaddatei '{path_path}' nicht gefunden.")
    with open(path_path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_segmentation_config():
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentierungsdatei '{seg_path}' nicht gefunden.")
    with open(seg_path, encoding="utf-8") as handle:
        return json.load(handle)


def read_path_section(name):
    return load_path_config().get(name, {})


def norm_path(value):
    return os.path.normpath(value) if value else ""


def cast_value(cfg, key, default, fn=lambda value: value):
    return fn(cfg.get(key, default))


def load_paths():
    return {key: norm_path(value) for key, value in read_path_section("paths").items()}


def load_preproc():
    cfg = load_segmentation_config().get("preprocessing", {})
    return {
        "preprocess_hsv_lower": np.array(cast_value(cfg, "preprocess_hsv_lower", [0, 0, 0])),
        "preprocess_hsv_upper": np.array(cast_value(cfg, "preprocess_hsv_upper", [0, 0, 0])),
        "minimum_contour_area": cast_value(cfg, "minimum_contour_area", 0),
        "warp_frame_size": tuple(cast_value(cfg, "warp_frame_size", [0, 0])),
        "target_width": cast_value(cfg, "target_width", 0),
        "target_height": cast_value(cfg, "target_height", 0),
    }


path_map = load_paths()
raw_dir = path_map["raw_image_directory"]
proc_dir = path_map["processed_image_directory"]
preproc_cfg = load_preproc()


def show_progress(prefix, current, total, bar_len=30):
    if total > 0:
        ratio = min(max(current / total, 0), 1)
        filled = int(bar_len * ratio)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r{prefix.ljust(20)}[{bar}] {ratio * 100:5.1f}% ({current}/{total})", end="", flush=True)


def clear_folder(folder):
    if os.path.exists(folder):
        def _on_rm_error(func, path, exc_info):
            if isinstance(exc_info[1], PermissionError):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            else:
                raise
        shutil.rmtree(folder, onerror=_on_rm_error)


def scan_images(source_dir):
    suffixes = (".jpg", ".jpeg", ".png")
    for root, _, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(suffixes):
                yield root, name


def warp_segments(image, warp_cfg):
    mask = cv2.bitwise_not(cv2.inRange(
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        warp_cfg["preprocess_hsv_lower"],
        warp_cfg["preprocess_hsv_upper"],
    ))
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    warp_w, warp_h = warp_cfg["warp_frame_size"]
    target_size = (warp_cfg["target_width"], warp_cfg["target_height"])
    dst_pts = np.float32([[0, warp_h - 1], [0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1]])
    outputs, min_area = [], warp_cfg["minimum_contour_area"]

    for contour in (cnt for cnt in contours if cv2.contourArea(cnt) > min_area):
        rect = cv2.minAreaRect(contour)
        if rect[1][1] > rect[1][0]:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90)

        transform = cv2.getPerspectiveTransform(cv2.boxPoints(rect).astype("float32"), dst_pts)
        mask_fill = np.zeros_like(gray)
        cv2.drawContours(mask_fill, [contour], -1, 255, cv2.FILLED)
        warped = cv2.warpPerspective(cv2.bitwise_and(masked, masked, mask=mask_fill), transform, (warp_w, warp_h), cv2.INTER_CUBIC)
        outputs.append(cv2.resize(warped, target_size, interpolation=cv2.INTER_CUBIC))

    return outputs


def segment_folder(source_dir, target_dir, preproc_cfg):
    clear_folder(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    image_files = list(scan_images(source_dir))
    total_files = len(image_files)
    if total_files == 0:
        print(f"Keine Bilder in '{source_dir}' gefunden.")
        return

    for idx, (root, name) in enumerate(image_files, 1):
        image = cv2.imread(os.path.join(root, name))
        if image is None:
            continue

        warp_tiles = warp_segments(image, preproc_cfg)
        if warp_tiles:
            class_dir = os.path.join(target_dir, os.path.basename(root))
            os.makedirs(class_dir, exist_ok=True)
            for tile in warp_tiles:
                cv2.imwrite(os.path.join(class_dir, name), tile)

        show_progress("  Segmentierung", idx, total_files)

    print()


def segment_cli():
    if not os.path.exists(raw_dir):
        print(f"Fehler: Quellordner '{raw_dir}' nicht gefunden. Bitte Bilder bereitstellen.")
        return

    os.makedirs(os.path.dirname(proc_dir), exist_ok=True)
    segment_folder(raw_dir, proc_dir, preproc_cfg)


if __name__ == "__main__":
    segment_cli()
