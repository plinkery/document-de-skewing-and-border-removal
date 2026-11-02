"""
test_border_removal.py - Person 4 runs this before merging
===========================================================
Run: python3 tests/test_border_removal.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.border_removal import remove_borders


def test_output_is_colour():
    fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = remove_borders(fake)
    assert result.ndim == 3, f"Expected 3D output, got shape {result.shape}"
    print('PASS: output is a colour image')

def test_output_not_larger():
    fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = remove_borders(fake)
    h_in, w_in   = fake.shape[:2]
    h_out, w_out = result.shape[:2]
    assert h_out <= h_in and w_out <= w_in, f"Output {result.shape[:2]} > input {fake.shape[:2]}"
    print(f'PASS: output {result.shape[:2]} <= input {fake.shape[:2]}')

if __name__ == '__main__':
    test_output_is_colour()
    test_output_not_larger()
    print('All border removal tests passed.')
