"""
test_pipeline.py - Integration test, Person 4 runs this
========================================================
Run: python3 tests/test_pipeline.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import run_pipeline


def test_pipeline_end_to_end():
    path = 'data/samples/sample1.jpg'
    if not os.path.exists(path):
        print(f'SKIP: no sample at {path} - add a real image and retry')
        return
    result = run_pipeline(path)
    assert result is not None and isinstance(result, np.ndarray) and result.ndim == 3
    print(f'PASS: pipeline output shape = {result.shape}')

if __name__ == '__main__':
    test_pipeline_end_to_end()
    print('Integration test complete.')
