#!/usr/bin/env python3
"""
Ablation‑ready HOG extraction pipeline — **modular, no‑CLI edition**
===================================================================

*Exact behaviour preserved; cleaner structure; no command‑line shim.*

Usage
-----
```py
from ablation_pipeline_modular import PipelineConfig, run_pipeline
cfg = PipelineConfig(train_dir="test", annot_csv="test/_annotations.csv")
run_pipeline(cfg)
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reference HOG implementation (unchanged logic)
# ---------------------------------------------------------------------------

from Custom_HOGFeatures import Hog_descriptor as _BaseHog  # noqa: N811 – keep name parity

# ---------------------------------------------------------------------------
# Gradient operator switchboard
# ---------------------------------------------------------------------------

def _sobel_grad(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang


def _scharr_grad(img: np.ndarray):
    gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang


def _prewitt_grad(img: np.ndarray):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = kernelx.T
    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang


def _roberts_grad(img: np.ndarray):
    kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang


def _dog_grad(img: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0):
    g1 = cv2.GaussianBlur(img, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2)
    gx = cv2.Sobel(dog, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(dog, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang


_GRADIENT_FN: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = {
    "sobel": _sobel_grad,
    "scharr": _scharr_grad,
    "prewitt": _prewitt_grad,
    "roberts": _roberts_grad,
    "dog": _dog_grad,
}

# ---------------------------------------------------------------------------
# HOG subclass with pluggable gradient & normaliser
# ---------------------------------------------------------------------------

class AblationHOG(_BaseHog):
    """Wrap *Custom_HOGFeatures.Hog_descriptor* with extra options."""

    def __init__(
        self,
        img: np.ndarray,
        *,
        cell_size: int,
        bin_size: int,
        unsigned: bool,
        gradient: str,
        block_norm: str,
    ) -> None:
        self.gradient_choice = gradient
        self.block_norm = block_norm
        self.unsigned = unsigned
        # Pre‑set angle_unit so parent asserts pass
        self.bin_size = bin_size
        self.angle_unit = 180 / bin_size if unsigned else 360 / bin_size
        super().__init__(img, cell_size=cell_size, bin_size=bin_size)

    # ------------ gradient override -------------------------------------
    def global_gradient(self):
        return _GRADIENT_FN[self.gradient_choice](self.img)

    # ------------ block normalisation -----------------------------------
    @staticmethod
    def _l2(vec: List[float]):
        mag = math.sqrt(sum(v * v for v in vec))
        return [v / mag for v in vec] if mag else vec

    @staticmethod
    def _l2hys(vec: List[float]):
        vec = AblationHOG._l2(vec)
        vec = [min(v, 0.2) for v in vec]
        return AblationHOG._l2(vec)

    @staticmethod
    def _l1(vec: List[float]):
        s = sum(abs(v) for v in vec)
        return [v / s for v in vec] if s else vec

    def _normalise_block(self, block_vector: List[float]):
        if self.block_norm == "none":
            return block_vector
        if self.block_norm == "l2":
            return self._l2(block_vector)
        if self.block_norm == "l2hys":
            return self._l2hys(block_vector)
        if self.block_norm == "l1":
            return self._l1(block_vector)
        raise ValueError(f"Unknown block_norm {self.block_norm}")

    # ------------ full extract() rewrite to use custom normaliser -------
    def extract(self):  # noqa: D401 – keep original signature
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        if self.unsigned:
            gradient_angle %= 180.0

        cell_gradient_vector = np.zeros(
            (height // self.cell_size, width // self.cell_size, self.bin_size)
        )

        # per‑cell histograms
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_mag = gradient_magnitude[
                    i * self.cell_size : (i + 1) * self.cell_size,
                    j * self.cell_size : (j + 1) * self.cell_size,
                ]
                cell_ang = gradient_angle[
                    i * self.cell_size : (i + 1) * self.cell_size,
                    j * self.cell_size : (j + 1) * self.cell_size,
                ]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_mag, cell_ang)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        # block normalisation (2×2 cells)
        hog_vector: List[List[float]] = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector: List[float] = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                hog_vector.append(self._normalise_block(block_vector))

        return hog_vector, hog_image

# ---------------------------------------------------------------------------
# Pre‑processing helpers
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray, cfg: "PipelineConfig") -> np.ndarray:
    """Convert BGR crop to 64×128 grayscale patch ready for HOG."""

    # colour‑space conversion
    cs = cfg.color_space
    if cs == "gray":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif cs == "rgb":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif cs == "lab":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    elif cs == "ycrcb":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        raise ValueError(f"Unknown COLOR_SPACE {cs}")

    img = img.astype(np.float32)

    # gamma correction
    if cfg.gamma != 1.0:
        img = ((img / 255.0) ** cfg.gamma) * 255.0

    # gaussian blur
    if cfg.gaussian_sigma > 0.0:
        ksize = int(4 * cfg.gaussian_sigma + 1) | 1
        img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=cfg.gaussian_sigma)

    img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineConfig:
    # dataset / I-O
    annot_csv: str = "train/_annotations.csv"
    train_dir: str = "train"
    out_dir: str = "trainHOG"      # single folder; T/F suffix encodes class
    max_images: int = 8_000

    # HOG core
    cell_size: int = 8
    bin_size: int = 9
    unsigned: bool = True
    gradient: str = "sobel"
    block_norm: str = "l2"

    # extras
    phog_levels: int = 0          # 0 = off

    # pre‑processing
    gamma: float = 1.0
    gaussian_sigma: float = 0.0
    color_space: str = "gray"

    # output
    save_descriptor: bool = False

    # output label logic
    label_column: str = "class"
    positive_labels: set[str] = frozenset({"person", "pedestrians"})

    # convenience
    def with_overrides(self, **kwargs) -> "PipelineConfig":
        return replace(self, **kwargs)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> None:
    """Iterate over annotation CSV and generate HOG files.

    Output naming scheme — identical to original scripts
    ----------------------------------------------------
    `{idx}_{filename}{T|F}[ _fd].txt`
    * `T` = person / pedestrians; `F` = everything else
    * `_fd.txt` if `save_descriptor=True`, otherwise plain `.txt` visualisation
    """

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(_yield_rows(Path(cfg.annot_csv), cfg.max_images), start=1):
        img_path = Path(cfg.train_dir) / row["filename"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️  Missing {img_path}")
            continue

        # crop to annotation box
        crop = img_bgr[
            int(row["ymin"]): int(row["ymax"]),
            int(row["xmin"]): int(row["xmax"]),
        ]
        if crop.size == 0:
            print(f"⚠️  Empty crop in {img_path}")
            continue

        img_gray = preprocess(crop, cfg)

        hog = AblationHOG(
            img_gray,
            cell_size=cfg.cell_size,
            bin_size=cfg.bin_size,
            unsigned=cfg.unsigned,
            gradient=cfg.gradient,
            block_norm=cfg.block_norm,
        )
        hog_fd, hog_vis = hog.extract()

        # optional pyramid HOG levels
        if cfg.phog_levels:
            for lvl in range(1, cfg.phog_levels + 1):
                cell = cfg.cell_size // (2 ** lvl)
                if cell < 2:
                    break
                ph = AblationHOG(
                    img_gray,
                    cell_size=cell,
                    bin_size=cfg.bin_size,
                    unsigned=cfg.unsigned,
                    gradient=cfg.gradient,
                    block_norm=cfg.block_norm,
                )
                fd_lvl, _ = ph.extract()
                hog_fd.extend(fd_lvl)

        # build output filename
        label = row[cfg.label_column].lower()
        suffix = "T" if label in cfg.positive_labels else "F"
        base = out_dir / f"{idx}_{row['filename']}{suffix}"

        if cfg.save_descriptor:
            vec = np.asarray(hog_fd, dtype=np.float32).ravel()
            _save_array(base.parent / f"{base.name}_fd.txt", vec)
        else:
            vis = (np.asarray(hog_vis, dtype=np.float32) / 255.0)
            _save_array(base.with_suffix(".txt"), vis)

        if idx % 500 == 0:
            print(f"Processed {idx} images…")

    print("Finished extracting!")





if __name__ == "__main__":

    '''
    ### READ THIS FIRST ###
    - Below is a guideline for how to use this pipeline
    '''

    from ablation_HOG import PipelineConfig, run_pipeline
    cfg = PipelineConfig(
    # ── dataset / I-O ────────────────────────────────────────────────
    annot_csv="train/_annotations.csv",     # Path to annotation CSV file
    train_dir="train",                      # Folder containing input images
    out_dir="trainHOG",                     # Output folder for results
    max_images=8000,                        # Max number of rows to process

    # ── HOG core ─────────────────────────────────────────────────────
    cell_size=8,                            # HOG cell size (pixels)
    bin_size=9,                             # Number of orientation bins
    unsigned=True,                          # Use 0–180° (True) or 0–360° (False)
    gradient="sobel",                       # sobel | scharr | prewitt | roberts | dog
    block_norm="l2",                        # l2 | l2hys | l1 | none

    # ── extras ───────────────────────────────────────────────────────
    phog_levels=0,                          # Spatial pyramid HOG (0 = off)

    # ── pre-processing ──────────────────────────────────────────────
    color_space="gray",                     # gray | rgb | lab | ycrcb
    gamma=1.0,                              # Gamma correction (1.0 = none)
    gaussian_sigma=0.0,                     # Gaussian blur sigma (0.0 = off)

    # ── output ──────────────────────────────────────────────────────
    save_descriptor=False,                  # Save *_fd.txt (True) or visual HOG .txt (False)

    # ── label logic ─────────────────────────────────────────────────
    # Modify this if changing datasets!!
    label_column="class",                   # Column to use for label ("T"/"F" decision)
    positive_labels={"person", "pedestrians"}  # Labels considered as positive ("T")
)

# run_pipeline(cfg)
