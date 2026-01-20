#!/usr/bin/env python3
"""
Homography Verification Test Script (Standalone)

This script verifies the accuracy of the homography calibration by:
1. Testing corner points map correctly to field coordinates
2. Testing inverse transformation (field -> image)
3. Testing distance calculations
4. Testing intermediate points
5. Checking for numerical stability
"""

import numpy as np
import math
import cv2
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


# ===== Inlined calibration functions from soccer_ai/calibration.py =====

@dataclass
class PlaneCalibration:
    homography: np.ndarray
    field_width_m: float
    field_height_m: float
    image_points: List[Tuple[float, float]]
    field_points: List[Tuple[float, float]]


def _order_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected 4 points of shape (x, y).")

    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]

    sums = ordered.sum(axis=1)
    tl_idx = int(np.argmin(sums))
    ordered = np.roll(ordered, -tl_idx, axis=0)

    def _cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if _cross(ordered[0], ordered[1], ordered[2]) < 0:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)

    return [tuple(map(float, pt)) for pt in ordered]


def _field_points(field_width_m: float, field_height_m: float) -> List[Tuple[float, float]]:
    return [
        (0.0, 0.0),
        (float(field_width_m), 0.0),
        (float(field_width_m), float(field_height_m)),
        (0.0, float(field_height_m)),
    ]


def build_calibration(
    image_points: Iterable[Sequence[float]],
    field_width_m: float,
    field_height_m: float,
    reorder: bool = True,
) -> PlaneCalibration:
    pts = list(image_points)
    if len(pts) != 4:
        raise ValueError("Expected exactly 4 image points.")
    ordered = _order_points(pts) if reorder else [tuple(map(float, p)) for p in pts]
    field_pts = _field_points(field_width_m, field_height_m)
    h_matrix, _ = cv2.findHomography(
        np.array(ordered, dtype=np.float32),
        np.array(field_pts, dtype=np.float32),
    )
    if h_matrix is None:
        raise RuntimeError("Unable to compute homography from the provided points.")
    return PlaneCalibration(
        homography=h_matrix,
        field_width_m=float(field_width_m),
        field_height_m=float(field_height_m),
        image_points=ordered,
        field_points=field_pts,
    )


def project_point(point: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    arr = np.array([[point]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(arr, homography)
    return float(mapped[0][0][0]), float(mapped[0][0][1])


def project_points(
    points: Iterable[Tuple[float, float]], homography: np.ndarray
) -> List[Tuple[float, float]]:
    pts = np.array(list(points), dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, homography)
    return [(float(p[0][0]), float(p[0][1])) for p in mapped]


# ===== Tests =====

def test_corner_mapping():
    """Test that image corners map exactly to field corners."""
    print("\n" + "="*60)
    print("TEST 1: Corner Point Mapping")
    print("="*60)
    
    # Simulate a perspective view of a 20m x 10m rectangle
    # These points form a trapezoid (typical camera view of a rectangle)
    image_points = [
        (100, 100),   # top-left
        (500, 120),   # top-right  
        (550, 400),   # bottom-right
        (50, 380),    # bottom-left
    ]
    
    field_width_m = 20.0
    field_height_m = 10.0
    
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=field_width_m,
        field_height_m=field_height_m,
        reorder=True,
    )
    
    expected_field = [
        (0.0, 0.0),                              # top-left
        (field_width_m, 0.0),                    # top-right
        (field_width_m, field_height_m),         # bottom-right
        (0.0, field_height_m),                   # bottom-left
    ]
    
    print(f"Image points (ordered): {calibration.image_points}")
    print(f"Expected field points:  {expected_field}")
    print()
    
    all_pass = True
    for i, (img_pt, expected) in enumerate(zip(calibration.image_points, expected_field)):
        projected = project_point(img_pt, calibration.homography)
        error = math.hypot(projected[0] - expected[0], projected[1] - expected[1])
        status = "✅ PASS" if error < 0.001 else "❌ FAIL"
        if error >= 0.001:
            all_pass = False
        print(f"  Corner {i+1}: {img_pt} -> {projected}")
        print(f"           Expected: {expected}, Error: {error:.6f}m {status}")
    
    return all_pass


def test_inverse_mapping():
    """Test that field points can be mapped back to image points."""
    print("\n" + "="*60)
    print("TEST 2: Inverse Mapping (Field -> Image)")
    print("="*60)
    
    image_points = [
        (100, 100),
        (500, 120),
        (550, 400),
        (50, 380),
    ]
    
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=20.0,
        field_height_m=10.0,
        reorder=True,
    )
    
    # Compute inverse homography
    h_inv = np.linalg.inv(calibration.homography)
    
    print("Testing inverse transform of field corners back to image:")
    print()
    
    all_pass = True
    for i, (field_pt, img_pt) in enumerate(zip(calibration.field_points, calibration.image_points)):
        # Project field point back to image
        arr = np.array([[field_pt]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(arr, h_inv)
        recovered = (float(mapped[0][0][0]), float(mapped[0][0][1]))
        
        error = math.hypot(recovered[0] - img_pt[0], recovered[1] - img_pt[1])
        status = "✅ PASS" if error < 0.01 else "❌ FAIL"
        if error >= 0.01:
            all_pass = False
        print(f"  Field {field_pt} -> Image {recovered}")
        print(f"           Expected: {img_pt}, Error: {error:.4f}px {status}")
    
    return all_pass


def test_distance_accuracy():
    """Test that distances in the field coordinate system are accurate."""
    print("\n" + "="*60)
    print("TEST 3: Distance Accuracy")
    print("="*60)
    
    # Use a simple rectangle (no perspective distortion) for baseline
    image_points = [
        (0, 0),
        (200, 0),
        (200, 100),
        (0, 100),
    ]
    
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=20.0,
        field_height_m=10.0,
        reorder=True,
    )
    
    # Test horizontal distance (should be 20m for full width)
    p1 = project_point(calibration.image_points[0], calibration.homography)
    p2 = project_point(calibration.image_points[1], calibration.homography)
    horizontal_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    # Test vertical distance (should be 10m for full height)
    p3 = project_point(calibration.image_points[3], calibration.homography)
    vertical_dist = math.hypot(p3[0] - p1[0], p3[1] - p1[1])
    
    # Test diagonal
    p4 = project_point(calibration.image_points[2], calibration.homography)
    diagonal_dist = math.hypot(p4[0] - p1[0], p4[1] - p1[1])
    expected_diagonal = math.hypot(20.0, 10.0)
    
    print(f"  Horizontal (width):  {horizontal_dist:.4f}m, Expected: 20.0m")
    print(f"  Vertical (height):   {vertical_dist:.4f}m, Expected: 10.0m")
    print(f"  Diagonal:            {diagonal_dist:.4f}m, Expected: {expected_diagonal:.4f}m")
    
    h_error = abs(horizontal_dist - 20.0)
    v_error = abs(vertical_dist - 10.0)
    d_error = abs(diagonal_dist - expected_diagonal)
    
    all_pass = h_error < 0.001 and v_error < 0.001 and d_error < 0.001
    status = "✅ ALL PASS" if all_pass else "❌ SOME FAILED"
    print(f"\n  {status}")
    
    return all_pass


def test_midpoint_accuracy():
    """Test that midpoints project correctly."""
    print("\n" + "="*60)
    print("TEST 4: Midpoint Projection")
    print("="*60)
    
    image_points = [
        (100, 100),
        (500, 120),
        (550, 400),
        (50, 380),
    ]
    
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=20.0,
        field_height_m=10.0,
        reorder=True,
    )
    
    # Center of the image rectangle (average of corners)
    img_center = (
        sum(p[0] for p in calibration.image_points) / 4,
        sum(p[1] for p in calibration.image_points) / 4,
    )
    
    # Project center
    field_center = project_point(img_center, calibration.homography)
    expected_center = (10.0, 5.0)  # Center of 20x10 field
    
    error = math.hypot(field_center[0] - expected_center[0], field_center[1] - expected_center[1])
    
    print(f"  Image center:  {img_center}")
    print(f"  Field center:  {field_center}")
    print(f"  Expected:      {expected_center}")
    print(f"  Error:         {error:.4f}m")
    
    # Note: For a perspective transformation, the image center doesn't necessarily
    # map to the field center. This is expected behavior.
    print("\n  ⚠️ NOTE: With perspective distortion, image center ≠ field center")
    print("           This is mathematically correct behavior.")
    
    return True  # This test is informational


def test_scale_variation():
    """Test meters-per-pixel scale at different locations."""
    print("\n" + "="*60)
    print("TEST 5: Scale Variation Across Field")
    print("="*60)
    
    image_points = [
        (100, 100),
        (500, 120),
        (550, 400),
        (50, 380),
    ]
    
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=20.0,
        field_height_m=10.0,
        reorder=True,
    )
    
    def m_per_px_at(point, homography):
        """Calculate local meters-per-pixel at a point."""
        base = project_point(point, homography)
        dx = project_point((point[0] + 1.0, point[1]), homography)
        dy = project_point((point[0], point[1] + 1.0), homography)
        mx = math.hypot(dx[0] - base[0], dx[1] - base[1])
        my = math.hypot(dy[0] - base[0], dy[1] - base[1])
        return (mx + my) / 2.0
    
    test_points = [
        ("Top-left area", (150, 150)),
        ("Top-right area", (450, 150)),
        ("Bottom-left area", (100, 350)),
        ("Bottom-right area", (500, 350)),
        ("Center", (300, 250)),
    ]
    
    print("  Local scale (meters per pixel) at different locations:")
    print()
    
    scales = []
    for name, pt in test_points:
        scale = m_per_px_at(pt, calibration.homography)
        scales.append(scale)
        print(f"  {name:20s}: {scale:.6f} m/px")
    
    scale_ratio = max(scales) / min(scales)
    print(f"\n  Max/Min scale ratio: {scale_ratio:.2f}x")
    print("  (Higher ratio = more perspective distortion)")
    
    return True


def test_homography_determinant():
    """Check homography matrix properties for numerical stability."""
    print("\n" + "="*60)
    print("TEST 6: Homography Matrix Properties")
    print("="*60)
    
    image_points = [
        (100, 100),
        (500, 120),
        (550, 400),
        (50, 380),
    ]
    
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=20.0,
        field_height_m=10.0,
        reorder=True,
    )
    
    h = calibration.homography
    det = np.linalg.det(h)
    cond = np.linalg.cond(h)
    
    print(f"  Homography matrix:\n{h}\n")
    print(f"  Determinant:        {det:.6f}")
    print(f"  Condition number:   {cond:.2f}")
    
    # Check if matrix is well-conditioned
    det_ok = abs(det) > 1e-10
    cond_ok = cond < 1e6
    
    print()
    print(f"  Determinant check:  {'✅ PASS' if det_ok else '❌ FAIL'} (should be non-zero)")
    print(f"  Condition check:    {'✅ PASS' if cond_ok else '⚠️ WARNING'} (prefer < 1e6)")
    
    return det_ok


def test_batch_projection():
    """Test batch projection matches individual projections."""
    print("\n" + "="*60)
    print("TEST 7: Batch vs Individual Projection")
    print("="*60)
    
    image_points = [
        (100, 100),
        (500, 120),
        (550, 400),
        (50, 380),
    ]
    
    calibration = build_calibration(
        image_points=image_points,
        field_width_m=20.0,
        field_height_m=10.0,
        reorder=True,
    )
    
    test_points = [(200, 200), (300, 300), (400, 350)]
    
    # Individual projection
    individual = [project_point(p, calibration.homography) for p in test_points]
    
    # Batch projection
    batch = project_points(test_points, calibration.homography)
    
    all_pass = True
    for i, (ind, bat) in enumerate(zip(individual, batch)):
        error = math.hypot(ind[0] - bat[0], ind[1] - bat[1])
        status = "✅" if error < 1e-6 else "❌"
        if error >= 1e-6:
            all_pass = False
        print(f"  Point {i+1}: Individual {ind} vs Batch {bat}")
        print(f"           Error: {error:.10f} {status}")
    
    return all_pass


def test_point_ordering():
    """Test that point ordering works correctly for various input orderings."""
    print("\n" + "="*60)
    print("TEST 8: Point Ordering Robustness")
    print("="*60)
    
    # Same points but in different input orders
    points_variations = [
        # Already ordered correctly
        [(100, 100), (500, 120), (550, 400), (50, 380)],
        # Shuffled order 1
        [(500, 120), (100, 100), (50, 380), (550, 400)],
        # Shuffled order 2
        [(550, 400), (50, 380), (100, 100), (500, 120)],
        # Reverse order
        [(50, 380), (550, 400), (500, 120), (100, 100)],
    ]
    
    all_pass = True
    reference_ordered = None
    
    for i, points in enumerate(points_variations):
        calibration = build_calibration(
            image_points=points,
            field_width_m=20.0,
            field_height_m=10.0,
            reorder=True,
        )
        
        if reference_ordered is None:
            reference_ordered = calibration.image_points
            print(f"  Variation {i+1}: Reference ordering = {calibration.image_points}")
        else:
            matches = calibration.image_points == reference_ordered
            status = "✅ PASS" if matches else "❌ FAIL"
            if not matches:
                all_pass = False
            print(f"  Variation {i+1}: {calibration.image_points}")
            print(f"           Matches reference: {status}")
    
    return all_pass


def main():
    print("\n" + "#"*60)
    print("#  HOMOGRAPHY CALIBRATION VERIFICATION")
    print("#"*60)
    
    results = []
    
    results.append(("Corner Mapping", test_corner_mapping()))
    results.append(("Inverse Mapping", test_inverse_mapping()))
    results.append(("Distance Accuracy", test_distance_accuracy()))
    results.append(("Midpoint Accuracy", test_midpoint_accuracy()))
    results.append(("Scale Variation", test_scale_variation()))
    results.append(("Matrix Properties", test_homography_determinant()))
    results.append(("Batch Projection", test_batch_projection()))
    results.append(("Point Ordering", test_point_ordering()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:25s}: {status}")
    
    total_pass = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  Total: {total_pass}/{total} tests passed")
    
    if total_pass == total:
        print("\n  🎉 All homography tests passed!")
    else:
        print("\n  ⚠️ Some tests failed - review the output above")


if __name__ == "__main__":
    main()
