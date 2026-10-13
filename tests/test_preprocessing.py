"""
test_preprocessing.py - Person 1 runs this before merging
==========================================================
Run: python3 tests/test_preprocessing.py
"""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.preprocessing import preprocess_image


def test_output_is_2d():
    fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(fake)
    assert result.ndim == 2, f"Expected 2D, got shape {result.shape}"
    print('PASS: output is 2D (grayscale/binary)')

def test_output_is_binary():
    fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(fake)
    unique = set(np.unique(result).tolist())
    assert unique.issubset({0, 255}), f"Expected only 0 and 255, got {unique}"
    print('PASS: output values are only 0 and 255')

def test_real_sample():
    path = 'data/samples/sample1.jpg'
    if not os.path.exists(path):
        print(f'SKIP: no sample at {path}')
        return
    img = cv2.imread(path)
    result = preprocess_image(img)
    assert result.ndim == 2
    print(f'PASS: real sample processed, shape = {result.shape}')

if __name__ == '__main__':
    test_output_is_2d()
    test_output_is_binary()
    test_real_sample()
    print('All preprocessing tests passed.')
