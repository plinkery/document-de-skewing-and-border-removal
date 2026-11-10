"""
test_pipeline.py - Integration test, Person 4 runs this
========================================================
Run: python3 tests/test_pipeline.py
"""
import numpy as np
import sys, os
import cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import discover_sample_images, run_pipeline


def test_pipeline_end_to_end():
    sample_paths = discover_sample_images()
    assert sample_paths, 'No supported sample images found under data/samples/'
    path = sample_paths[0]
    source = cv2.imread(path)
    assert source is not None, f'Could not read discovered sample: {path}'

    result = run_pipeline(path)
    assert isinstance(result, np.ndarray), f'Expected ndarray, got {type(result)}'
    assert result.ndim == 3 and result.shape[2] == 3, f'Invalid output shape: {result.shape}'
    assert result.dtype == np.uint8, f'Expected uint8, got {result.dtype}'
    assert 0 < result.shape[0] <= source.shape[0], f'Invalid output height: {result.shape[0]}'
    assert 0 < result.shape[1] <= source.shape[1], f'Invalid output width: {result.shape[1]}'
    assert result.flags.c_contiguous, 'Pipeline output must be C-contiguous'
    print(f'PASS: pipeline output shape = {result.shape}')

if __name__ == '__main__':
    test_pipeline_end_to_end()
    print('Integration test complete.')
