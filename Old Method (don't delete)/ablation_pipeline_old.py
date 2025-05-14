#!/usr/bin/env python3
"""
Ablation‑ready HOG extraction pipeline
=====================================

− Saves the HOG *visualisation image* (not the descriptor) as CSV into
  `people/` and `notpeople/`.

Dependencies ─────────────────────────────────────────────────────────────
* Python ≥ 3.8
* OpenCV‑Python (cv2)
* NumPy
* (optional, commented‑out) scikit‑image ⇢ LBP / Gabor

Folder layout ───────────────────────────────────────────────────────────
train/
  ├── _annotations.csv   ← CSV with xmin,xmax, …, class
  └── <images>
people/
notpeople/

"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from Custom_HOGFeatures import Hog_descriptor

# -----------------------------------------------------------------------------
# Configuration block – tweak here, or override on the CLI
# -----------------------------------------------------------------------------
CONFIG: dict[str, object] = {
    # Dataset ------------------------------------------------------------------
    "MAX_IMAGES": 8_000,          # int |  cap on rows read from CSV
    "ANNOT_CSV": "train/_annotations.csv",
    "TRAIN_DIR": "train",       # folder that contains the images
    "OUT_PEOPLE": "people",     # where to save <idx>_<filename>.txt
    "OUT_NOTPEOPLE": "notpeople",

    # HOG descriptor -----------------------------------------------------------
    "CELL_SIZE": 8,              # ↔ pixels_per_cell in skimage
    "BIN_SIZE": 9,               # ↔ orientations
    "UNSIGNED": True,            # use 0–180° (True) or 0–360° (False)
    "GRADIENT": "sobel",        # sobel | scharr | prewitt | roberts | dog
    "BLOCK_NORM": "l2",        # l2 | l2hys | l1 | none

    # Extended variants --------------------------------------------------------
    "PHOG_LEVELS": 0,            # 0 = OFF, 1 = add 1 pyramid level, …
    "GLOH": False,               # experimental log‑polar layout
    "CHOG": False,               # average opposite bins

    # Extra descriptors (late fusion) -----------------------------------------
    "LBP": False,                # requires scikit‑image
    "LBP_RADIUS": 1,
    "LBP_POINTS": 8,
    "GABOR": False,              # requires scikit‑image

    # Pre‑processing -----------------------------------------------------------
    "GAMMA": 1.0,                # 1.0 = OFF; e.g. 0.5 brightens shadows
    "GAUSSIAN_SIGMA": 0.0,       # 0.0 = OFF; 0.5–1.0 reduces sensor noise
    "COLOR_SPACE": "gray",      # gray | rgb | lab | ycrcb

    # Output -------------------------------------------------------------------
    "SAVE_DESCRIPTOR": False,    # also save the 1‑D feature vector
}

# -----------------------------------------------------------------------------
# Helper: gradient operator switchboard
# -----------------------------------------------------------------------------

def sobel_grad(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def scharr_grad(img: np.ndarray):
    gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def prewitt_grad(img: np.ndarray):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = kernelx.T
    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def roberts_grad(img: np.ndarray):
    kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def dog_grad(img: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0):
    g1 = cv2.GaussianBlur(img, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2)
    gx = cv2.Sobel(dog, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(dog, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

GRADIENT_FN = {
    "sobel": sobel_grad,
    "scharr": scharr_grad,
    "prewitt": prewitt_grad,
    "roberts": roberts_grad,
    "dog": dog_grad,
}

# -----------------------------------------------------------------------------
# Custom extension of Hog_descriptor that plugs in the gradient & norm variants
# -----------------------------------------------------------------------------

class AblationHOG(Hog_descriptor):
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
        # adapt angle_unit before super() so assertions pass
        super().__init__(img, cell_size=cell_size, bin_size=bin_size)
        if self.unsigned:
            self.angle_unit = 180 / self.bin_size
        # else keep 360 / bin_size from parent

    # override gradient step
    def global_gradient(self):
        fn = GRADIENT_FN[self.gradient_choice]
        return fn(self.img)

    # optional block normalisation tweaks
    @staticmethod
    def _l2(vec: list[float]):
        mag = math.sqrt(sum(v * v for v in vec))
        return [v / mag for v in vec] if mag else vec

    @staticmethod
    def _l2hys(vec: list[float]):
        vec = AblationHOG._l2(vec)
        vec = [min(v, 0.2) for v in vec]
        return AblationHOG._l2(vec)

    @staticmethod
    def _l1(vec: list[float]):
        s = sum(abs(v) for v in vec)
        return [v / s for v in vec] if s else vec

    def _normalise_block(self, block_vector: list[float]):
        if self.block_norm == "none":
            return block_vector
        if self.block_norm == "l2":
            return self._l2(block_vector)
        if self.block_norm == "l2hys":
            return self._l2hys(block_vector)
        if self.block_norm == "l1":
            return self._l1(block_vector)
        raise ValueError(f"Unknown block_norm {self.block_norm}")

    # re‑implement extract() to call chosen normaliser
    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        if self.unsigned:
            gradient_angle %= 180.0
        cell_gradient_vector = np.zeros(
            (height // self.cell_size, width // self.cell_size, self.bin_size)
        )
        # populate cells
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
        hog_vector: list[list[float]] = []
        # blocks of 2×2 cells ----------------------------------------------------
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                block_vector = self._normalise_block(block_vector)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

# -----------------------------------------------------------------------------
# Optional extra descriptors (LBP / Gabor) – OFF by default
# -----------------------------------------------------------------------------
# To enable, switch the CONFIG flags to True **and** pip‑install scikit‑image.
#
# from skimage.feature import local_binary_pattern
#
# def lbp_descriptor(img_gray: np.ndarray, radius: int, n_points: int) -> np.ndarray:
#     lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
#     # 59 uniform patterns + 1 misc bin
#     hist, _ = np.histogram(lbp.ravel(), bins=59 + 1, range=(0, n_points + 2))
#     hist = hist.astype(np.float32)
#     hist /= hist.sum() or 1
#     return hist
#
# from skimage.filters import gabor
#
# def gabor_descriptor(img_gray: np.ndarray) -> np.ndarray:
#     thetas = np.linspace(0, np.pi, 8, endpoint=False)
#     sigmas = (1, 3)
#     feats = []
#     for theta in thetas:
#         for sigma in sigmas:
#             filt_real, filt_imag = gabor(img_gray, frequency=0.2, theta=theta, sigma_x=sigma, sigma_y=sigma)
#             feats.append(filt_real.mean())
#             feats.append(filt_real.var())
#     return np.array(feats, dtype=np.float32)

# -----------------------------------------------------------------------------
# Pre‑processing helpers
# -----------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    """Returns the grayscale image‑patch ready for HOG."""
    # colour conversion --------------------------------------------------------
    cs = cfg["COLOR_SPACE"]
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
    # gamma correction ---------------------------------------------------------
    gamma = float(cfg["GAMMA"])
    if gamma != 1.0:
        img = ((img / 255.0) ** gamma) * 255.0
    # gaussian blur ------------------------------------------------------------
    sigma = float(cfg["GAUSSIAN_SIGMA"])
    if sigma > 0.0:
        ksize = int(4 * sigma + 1) | 1  # ensure odd
        img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
    # resize to 64×128 ----------------------------------------------------------
    img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.uint8)
    return img

# -----------------------------------------------------------------------------
# Main extractor loop
# -----------------------------------------------------------------------------

def run_for_csv(cfg: dict[str, object] = CONFIG):
    """Iterates over the annotation CSV and saves HOG images/feature vectors."""
    # ensure output dirs exist --------------------------------------------------
    Path(cfg["OUT_PEOPLE"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["OUT_NOTPEOPLE"]).mkdir(parents=True, exist_ok=True)

    with open(cfg["ANNOT_CSV"], newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader, start=1):
            if idx > int(cfg["MAX_IMAGES"]):
                break

            img_path = Path(cfg["TRAIN_DIR"]) / row["filename"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"⚠️  Could not read {img_path}")
                continue

            # crop using annotation box -----------------------------------------
            crop = img_bgr[
                int(row["ymin"]): int(row["ymax"]),
                int(row["xmin"]): int(row["xmax"]),
            ]
            if crop.size == 0:
                print(f"Empty crop for {img_path}")
                continue

            img_gray = preprocess(crop, cfg)

            # HOG ----------------------------------------------------------------
            hog = AblationHOG(
                img_gray,
                cell_size=int(cfg["CELL_SIZE"]),
                bin_size=int(cfg["BIN_SIZE"]),
                unsigned=bool(cfg["UNSIGNED"]),
                gradient=str(cfg["GRADIENT"]),
                block_norm=str(cfg["BLOCK_NORM"]),
            )
            hog_fd, hog_vis = hog.extract()

            # optional pyramidal HoG ------------------------------------------
            if cfg["PHOG_LEVELS"]:
                for lvl in range(1, int(cfg["PHOG_LEVELS"]) + 1):
                    cell = int(cfg["CELL_SIZE"]) // (2 ** lvl)
                    if cell < 2:
                        break  # stop if cells would be <2 px
                    ph = AblationHOG(
                        img_gray,
                        cell_size=cell,
                        bin_size=int(cfg["BIN_SIZE"]),
                        unsigned=bool(cfg["UNSIGNED"]),
                        gradient=str(cfg["GRADIENT"]),
                        block_norm=str(cfg["BLOCK_NORM"]),
                    )
                    fd_lvl, _ = ph.extract()
                    hog_fd.extend(fd_lvl)

            feature_vec = np.asarray(hog_fd, dtype=np.float32).ravel()
            hog_vis = np.asarray(hog_vis, dtype=np.float32)

            # late‑fusion extras --------------------------------------------------
            # if cfg["LBP"]:
            #     feature_vec = np.hstack([
            #         feature_vec,
            #         lbp_descriptor(img_gray, cfg["LBP_RADIUS"], cfg["LBP_POINTS"]),
            #     ])
            # if cfg["GABOR"]:
            #     feature_vec = np.hstack([feature_vec, gabor_descriptor(img_gray)])

            # save ---------------------------------------------------------------
            if row["class"].lower() in {"person", "pedestrians"}:
                out_base = Path(cfg["OUT_PEOPLE"]) / f"{idx}_{row['filename']}"
            else:
                out_base = Path(cfg["OUT_NOTPEOPLE"]) / f"{idx}_{row['filename']}"

            np.savetxt(f"{out_base}.txt", hog_vis, delimiter=",")
            if cfg["SAVE_DESCRIPTOR"]:
                np.savetxt(f"{out_base}_fd.txt", feature_vec, delimiter=",")

            if idx % 500 == 0:
                print(f"Processed {idx} images…")

    print("Finished extracting!")



'''
Example code to run in a notebook:

import importlib, ablation_pipeline as hogpipe      # ① import the module
importlib.reload(hogpipe)                           # ② pull in any edits

# ③ overwrite any defaults you care about
hogpipe.CONFIG.update(
    # ── HOG core ────────────────────────────────────────────────────────────
    GRADIENT="scharr",      # sobel | scharr | prewitt | roberts | dog
    UNSIGNED=True,          # 0–180° bins (True) vs. 0–360° (False)
    BIN_SIZE=9,             # number of orientation bins per cell
    CELL_SIZE=8,            # pixels per cell
    BLOCK_NORM="l2hys",     # l2 | l2hys | l1 | none
    PHOG_LEVELS=2,          # add 2 levels of pyramid-HOG (0 = off)

    # ── Pre-processing ─────────────────────────────────────────────────────
    COLOR_SPACE="gray",     # gray | rgb | lab | ycrcb
    GAMMA=1.0,              # 1.0 = off; <1 brightens shadows
    GAUSSIAN_SIGMA=0.0,     # 0.0 = no blur; 0.5–1.0 typical

    # ── Dataset / I/O tweaks ───────────────────────────────────────────────
    MAX_IMAGES=8000,       # cap rows read from the CSV
    SAVE_DESCRIPTOR=True,   # save the 1-D feature vector too
)

# ④ run the extractor
hogpipe.run_for_csv(hogpipe.CONFIG)
'''
