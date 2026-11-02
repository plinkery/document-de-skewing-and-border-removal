"""
test_deskew.py - Person 3 runs this before merging
===================================================
Run: python3 tests/test_deskew.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.deskew import deskew_image


def test_return_type():
    raw    = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    binary = np.zeros((480, 640), dtype=np.uint8)
    result = deskew_image(raw, binary)
    assert isinstance(result, tuple) and len(result) == 2, f"Expected (ndarray, float) tuple"
    img, angle = result
    assert isinstance(img, np.ndarray), f"First item must be ndarray, got {type(img)}"
    assert isinstance(angle, float), f"Second item must be float, got {type(angle)}"
    print('PASS: return type is (np.ndarray, float)')

def test_output_is_colour():
    raw    = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    binary = np.zeros((480, 640), dtype=np.uint8)
    img, _ = deskew_image(raw, binary)
    assert img.ndim == 3, f"Expected colour (3D), got shape {img.shape}"
    print('PASS: output image is colour (3-channel BGR)')

if __name__ == '__main__':
    test_return_type()
    test_output_is_colour()
    print('All deskew tests passed.')
