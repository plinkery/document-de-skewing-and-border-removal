"""
test_pipeline.py - Full End-to-End Pipeline Integration Test
============================================================
Run (quick single-image contract test):
    python3 tests/test_pipeline.py

Run (batch test on real images with 5-panel comparisons):
    python3 tests/test_pipeline.py --count 15 --debug
"""
import numpy as np
import sys, os
import cv2
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import discover_sample_images, run_pipeline, generate_comparison_panel
from modules.preprocessing import preprocess_image
from modules.morphology import apply_morphology
from modules.deskew import deskew_image
from modules.border_removal import remove_borders


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


def run_batch_integration(count: int = 15, debug: bool = True):
    """
    Run all 4 integrated modules across sample images and generate 5-panel comparisons.
    """
    sample_paths = discover_sample_images()
    if not sample_paths:
        print("ERROR: No sample images found under data/samples/")
        return

    test_paths = sample_paths[:count]
    out_dir = "data/outputs/comparisons"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Running Full Pipeline Integration Test on {len(test_paths)} Real Images")
    print("=" * 70)

    summary = []
    all_passed = True

    for idx, path in enumerate(test_paths, start=1):
        fname = os.path.basename(path)
        raw_img = cv2.imread(path)
        if raw_img is None:
            print(f"[{idx}/{len(test_paths)}] FAIL: Could not read {path}")
            all_passed = False
            continue

        try:
            # Step 1: Preprocessing (Person 1)
            b = preprocess_image(raw_img)
            assert b.ndim == 2 and b.dtype == np.uint8, "P1 contract violation"

            # Step 2: Morphology (Person 2)
            c = apply_morphology(b)
            assert c.shape == b.shape and c.dtype == np.uint8, "P2 contract violation"

            # Step 3: Deskew (Person 3)
            d, angle = deskew_image(raw_img, c)
            assert d.ndim == 3 and d.dtype == np.uint8, "P3 contract violation"

            # Step 4: Border Removal (Person 4)
            f = remove_borders(d)
            assert f.ndim == 3 and f.dtype == np.uint8, "P4 contract violation"

            status = "PASS"
            summary.append((fname, f"{raw_img.shape[1]}x{raw_img.shape[0]}", f"{angle:+.2f}°", f"{f.shape[1]}x{f.shape[0]}", status))

            if debug:
                base_name = os.path.splitext(fname)[0]
                out_path = os.path.join(out_dir, f"compare_{base_name}.png")
                generate_comparison_panel(raw_img, b, c, d, f, angle, out_path)

            print(f"[{idx}/{len(test_paths)}] PASS [{fname}]: skew={angle:+.2f}° | crop={raw_img.shape[:2]} -> {f.shape[:2]}")

        except Exception as e:
            print(f"[{idx}/{len(test_paths)}] FAIL [{fname}]: {e}")
            summary.append((fname, f"{raw_img.shape[1]}x{raw_img.shape[0]}", "N/A", "N/A", f"FAIL ({e})"))
            all_passed = False

    print("\n" + "=" * 70)
    print("  Integration Summary Table")
    print("=" * 70)
    print(f"{'Image':<18} {'Original':<14} {'Skew Angle':<12} {'Final Crop':<14} {'Status'}")
    print("-" * 70)
    for row in summary:
        print(f"{row[0]:<18} {row[1]:<14} {row[2]:<12} {row[3]:<14} {row[4]}")
    print("=" * 70)

    if debug:
        print(f"\n5-Panel comparison images saved to: {out_dir}/")
        print("View them in Windows at: \\\\wsl.localhost\\Ubuntu" + os.path.abspath(out_dir))

    if all_passed:
        print("\nSUCCESS: All sample images passed full end-to-end integration!")
    else:
        print("\nWARNING: One or more sample images failed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline integration tests")
    parser.add_argument('--count', type=int, default=None,
                        help='Number of real images to test in batch mode (e.g. 15)')
    parser.add_argument('--debug', action='store_true',
                        help='Generate 5-panel transition comparison images')
    args = parser.parse_args()

    if args.count is not None or args.debug:
        run_batch_integration(count=args.count or 15, debug=args.debug)
    else:
        test_pipeline_end_to_end()
        print('Integration test complete.')
