"""Frame-0 SSIM tripwire: rendered lottie frame vs source PNG.

Same SSIM implementation as extract_parts.py (gaussian, sigma 1.5) so numbers
are comparable with the manifest's gates (0.986 baseline, 0.92 gate).

Usage: python3 ssim_check.py <frame.png> <source.png>
"""
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def ssim(a, b, sigma=1.5):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    var_a = gaussian_filter(a * a, sigma) - mu_a**2
    var_b = gaussian_filter(b * b, sigma) - mu_b**2
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    return float((num / den).mean())


def main():
    frame = cv2.imread(sys.argv[1])
    source = cv2.imread(sys.argv[2])
    if source.shape[:2] != frame.shape[:2]:
        source = cv2.resize(source, (frame.shape[1], frame.shape[0]))
    g1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    diff = np.abs(frame.astype(int) - source.astype(int)).mean()
    print(f"SSIM {ssim(g1, g2):.4f}  mean|d| {diff:.2f}")


if __name__ == "__main__":
    main()
