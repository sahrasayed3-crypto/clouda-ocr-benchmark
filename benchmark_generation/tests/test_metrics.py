import cv2
import numpy as np

from ocrbench.metrics import assess


def test_local_shadow_does_not_look_like_ink_flooding():
    clean = np.full((600, 400), 245, np.uint8)
    for y in range(80, 540, 38):
        cv2.line(clean, (45, y), (350, y), 25, 7)
    shadow = clean.astype(np.float32)
    shadow[:, :180] *= 0.62
    result = assess(
        shadow.astype(np.uint8), clean,
        min_contrast_ratio=0.18, min_sharpness_ratio=0.01,
        min_ink_ratio=0.25, max_ink_ratio=4.0,
    )
    assert result.passed, result.reasons


def test_sparse_faded_line_keeps_adaptive_ink_coverage():
    clean = np.full((180, 1000), 235, np.uint8)
    cv2.putText(clean, "OCR benchmark line", (55, 115), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 35, 5)
    faded = np.clip(clean.astype(np.float32) * 0.45 + 118, 0, 255).astype(np.uint8)
    result = assess(faded, clean, min_contrast_ratio=0.18, min_sharpness_ratio=0.01,
                    min_ink_ratio=0.25, max_ink_ratio=4.0)
    assert result.ink_change_ratio > 0.15


def test_scan_grain_does_not_dominate_text_sharpness():
    rng = np.random.default_rng(7)
    clean = np.full((700, 500), 245, np.float32)
    for y in range(80, 640, 42):
        cv2.line(clean, (40, y), (455, y), 30, 6)
    clean = np.clip(clean + rng.normal(0, 35, clean.shape), 0, 255).astype(np.uint8)
    small = cv2.resize(clean, None, fx=.33, fy=.33, interpolation=cv2.INTER_AREA)
    degraded = cv2.resize(small, (500, 700), interpolation=cv2.INTER_CUBIC)
    result = assess(degraded, clean, min_contrast_ratio=0.18, min_sharpness_ratio=0.01,
                    min_ink_ratio=0.25, max_ink_ratio=4.0)
    assert result.sharpness_ratio > 0.01
