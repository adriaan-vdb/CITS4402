"""
custom_hog() Parameters Explained (Usage & Effects)
==================================================

custom_hog(img, orientations=9, pixels_per_cell=(8, 8),
               block_norm='l2hys', transform_sqrt=False,
               feature_vector=True, unsigned=True, gradient='sobel',
               gamma=1.0, gaussian_sigma=0.0):

orientations (int):
    - Defines how many bins to split gradient directions.
    - Higher values (e.g., 9, 16) capture finer edge direction info.
    - Lower values (e.g., 6) reduce detail, faster computation.
    - Default: 9 (standard from Dalal & Triggs HOG).

pixels_per_cell (tuple of int):
    - Size of the cell region in pixels, like (8, 8).
    - Smaller cells = more detailed texture, but larger feature vector.
    - Larger cells = more general, less sensitive to small edges.
    - Typical: (8, 8).

block_norm (str):
    - Method to normalize gradient histograms per block.
    - Options:
        'l2'     : Standard normalization.
        'l2hys'  : L2 + clipping outliers (default, robust).
        'l1'     : Normalizes using L1-norm.
        'none'   : No normalization (rarely used).
    - Accepts variations like 'L2-Hys', 'l2hys', etc.

transform_sqrt (bool):
    - Applies square root to pixel intensities before HOG.
    - Helps with uneven lighting & shadows.
    - Effectively a gamma correction with gamma=0.5.

feature_vector (bool):
    - If True: Output is flattened 1D feature vector (for classifiers like SVM).
    - If False: Output is kept as block-grid array (for visualization or advanced models).

unsigned (bool):
    - True  : Gradients are unsigned (0°–180°), ignores edge direction.
    - False : Uses full 0°–360°, detects signed edge direction.
    - Unsigned is standard for object detection.

gradient (str):
    - Selects the gradient filter used to compute edge directions.
    - Choices:
        'sobel'   : Balanced, general-purpose.
        'scharr'  : Stronger edges, small details.
        'prewitt' : Lighter, simpler filter.
        'roberts' : Focuses on diagonal edges.
        'dog'     : Multi-scale blob & edge detection (Difference of Gaussians).

gamma (float):
    - Gamma correction applied to input intensities.
    - >1.0 darkens bright regions (compress dynamic range).
    - <1.0 boosts shadows (brightens dark areas).
    - Default: 1.0 (no gamma correction).

gaussian_sigma (float):
    - Amount of Gaussian blur applied before gradients.
    - Helps reduce noise & textures that confuse edge detection.
    - 0.0 disables blurring (default).

Examples:
---------
# Typical pedestrian HOG feature extraction
hog_features = custom_hog(image_data, orientations=9, pixels_per_cell=(8, 8), block_norm='l2hys', gradient='prewitt')

# Enhance edges with Scharr & sqrt normalization (for dark images)
hog_features = custom_hog(image_data, gradient='scharr', block_norm='l2hys', transform_sqrt=True, feature_vector=True)

# Apply gamma correction & Gaussian blur to denoise before HOG
hog_features = custom_hog(image_data, block_norm='l2hys', gamma=0.8, gaussian_sigma=1.5, gradient='sobel')
"""


import cv2
import numpy as np

# ---------------------- Gradient Filters ------------------------

def sobel_grad(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def scharr_grad(img):
    gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def prewitt_grad(img):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = kernelx.T
    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def roberts_grad(img):
    kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(img, cv2.CV_64F, kernely)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

def dog_grad(img, sigma1=1.0, sigma2=2.0):
    g1 = cv2.GaussianBlur(img, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2)
    gx = cv2.Sobel(dog, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(dog, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, ang

GRADIENT_FILTERS = {
    "sobel": sobel_grad,
    "scharr": scharr_grad,
    "prewitt": prewitt_grad,
    "roberts": roberts_grad,
    "dog": dog_grad,
}

# ---------------------- HOG Descriptor ------------------------

class SimpleHOG:
    def __init__(self, img, cell_size, bin_size, unsigned, gradient, block_norm):
        self.img = img
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.unsigned = unsigned
        self.block_norm = block_norm.lower()
        self.angle_unit = 180 / bin_size if unsigned else 360 / bin_size
        gradient = gradient.lower()
        if gradient not in GRADIENT_FILTERS:
            raise ValueError(f"Unknown gradient type '{gradient}'. Choose from {list(GRADIENT_FILTERS.keys())}")
        self.gradient_func = GRADIENT_FILTERS[gradient]

    def extract(self):
        h, w = self.img.shape
        mag, ang = self.gradient_func(self.img)
        mag = np.abs(mag)
        if self.unsigned:
            ang %= 180.0

        cell_grad = np.zeros((h // self.cell_size, w // self.cell_size, self.bin_size))

        for i in range(cell_grad.shape[0]):
            for j in range(cell_grad.shape[1]):
                cell_mag = mag[i*self.cell_size:(i+1)*self.cell_size, j*self.cell_size:(j+1)*self.cell_size]
                cell_ang = ang[i*self.cell_size:(i+1)*self.cell_size, j*self.cell_size:(j+1)*self.cell_size]
                cell_grad[i, j] = self.cell_histogram(cell_mag, cell_ang)

        hog_vector = []
        for i in range(cell_grad.shape[0] - 1):
            for j in range(cell_grad.shape[1] - 1):
                block = np.concatenate((
                    cell_grad[i, j],
                    cell_grad[i, j+1],
                    cell_grad[i+1, j],
                    cell_grad[i+1, j+1]
                ))
                block = self.norm_block(block)
                hog_vector.append(block)

        return np.asarray(hog_vector)

    def cell_histogram(self, mags, angs):
        hist = np.zeros(self.bin_size)
        for i in range(mags.shape[0]):
            for j in range(mags.shape[1]):
                mag = mags[i, j]
                ang = angs[i, j]
                bin_idx = int(ang / self.angle_unit) % self.bin_size
                next_bin = (bin_idx + 1) % self.bin_size
                mod = ang % self.angle_unit
                hist[bin_idx] += mag * (1 - mod / self.angle_unit)
                hist[next_bin] += mag * (mod / self.angle_unit)
        return hist

    def norm_block(self, vec):
        if self.block_norm == "none":
            return vec
        elif self.block_norm == "l2":
            norm = np.linalg.norm(vec)
            return vec / norm if norm else vec
        elif self.block_norm == "l2hys":
            vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) else vec
            vec = np.clip(vec, 0, 0.2)
            norm = np.linalg.norm(vec)
            return vec / norm if norm else vec
        elif self.block_norm == "l1":
            norm = np.sum(np.abs(vec))
            return vec / norm if norm else vec
        else:
            raise ValueError(f"Unknown block_norm '{self.block_norm}'. Choose from ['l2', 'l2hys', 'l1', 'none']")

# ---------------------- Public API ------------------------

def custom_hog(img, orientations=9, pixels_per_cell=(8, 8),
               block_norm='l2hys', transform_sqrt=False,
               feature_vector=True, unsigned=True, gradient='sobel',
               gamma=1.0, gaussian_sigma=0.0):
    img = img.astype(np.float32)

    # --- Gamma Correction ---
    if gamma != 1.0:
        img = ((img / 255.0) ** gamma) * 255.0

    # --- Gaussian Blur ---
    if gaussian_sigma > 0.0:
        ksize = int(4 * gaussian_sigma + 1) | 1  # Ensure odd kernel size
        img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=gaussian_sigma)

    # --- Transform Sqrt ---
    if transform_sqrt:
        max_val = np.max(img)
        if max_val > 0:
            img = np.sqrt(img / max_val) * 255.0
        else:
            img = np.zeros_like(img)  # Black image stays black


    img = img.astype(np.uint8)

    hog = SimpleHOG(
        img,
        cell_size=pixels_per_cell[0],
        bin_size=orientations,
        unsigned=unsigned,
        gradient=gradient,
        block_norm=block_norm,
    )

    hog_vec = hog.extract()
    if feature_vector:
        hog_vec = hog_vec.ravel()

    return hog_vec
