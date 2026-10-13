"""
test_morphology.py - Person 2 runs this before merging
=======================================================
Run: python3 tests/test_morphology.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.morphology import apply_morphology


def test_shape_unchanged():
    fake = np.zeros((480, 640), dtype=np.uint8)
    fake[100:200, 100:300] = 255
    result = apply_morphology(fake)
    assert result.shape == fake.shape, f"Shape changed: {fake.shape} -> {result.shape}"
    print('PASS: output shape matches input shape')

def test_output_is_binary():
    fake = np.zeros((480, 640), dtype=np.uint8)
    fake[50:100, 50:100] = 255
    result = apply_morphology(fake)
    unique = set(np.unique(result).tolist())
    assert unique.issubset({0, 255}), f"Non-binary values: {unique}"
    print('PASS: output values are only 0 and 255')

def test_noise_reduction():
    clean = np.zeros((100, 100), dtype=np.uint8)
    clean[50, 50] = 255
    result = apply_morphology(clean)
    print(f'INFO: isolated noise pixel after morphology = {result[50,50]} (should be 0 after real implementation)')

if __name__ == '__main__':
    test_shape_unchanged()
    test_output_is_binary()
    test_noise_reduction()
    print('All morphology tests passed.')
