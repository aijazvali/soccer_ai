# Homography Calibration System

This document explains how the homography-based calibration works in the Soccer AI project to convert pixel coordinates into real-world measurements (meters).

## Overview

The homography system allows accurate calculation of:
- **Player speed** in km/h or m/s (instead of pixels/second)
- **Distance traveled** in meters
- **Shot/pass speed** in real-world units
- **Jump height** in meters

## How It Works

### The Math Behind It

A **homography** is a 3×3 transformation matrix that maps points from one plane to another. In our case:
- **Source plane**: The video frame (pixels)
- **Destination plane**: A real-world ground rectangle (meters)

```
┌─────────────────┐         ┌─────────────────┐
│  Video Frame    │         │   Field (meters)│
│   (pixels)      │   H     │                 │
│  ┌───────────┐  │ ──────► │  (0,0)─────(20,0)
│  │ ●       ● │  │         │    │           │
│  │           │  │         │    │   20m     │
│  │ ●       ● │  │         │    │           │
│  └───────────┘  │         │  (0,10)───(20,10)
└─────────────────┘         └─────────────────┘
```

The transformation is: `[x', y', w] = H × [x, y, 1]`

Where the final coordinates are: `(x'/w, y'/w)`

---

## Key Files

| File | Purpose |
|------|---------|
| `soccer_ai/calibration.py` | Core calibration functions |
| `soccer_ai/config.py` | Default field dimensions (20m × 10m) |
| `soccer_ai/pipelines/touch.py` | Uses calibration for speed/distance |
| `streamlit_app.py` | UI for marking calibration points |

---

## Calibration Process

### Step 1: Mark 4 Corners

In the Streamlit UI, you mark 4 points on the video frame that form a rectangle of **known real-world dimensions**.

```python
# Example: User marks these pixel coordinates
image_points = [
    (100, 100),   # top-left corner
    (500, 120),   # top-right corner
    (550, 400),   # bottom-right corner
    (50, 380),    # bottom-left corner
]
```

### Step 2: Build Calibration

The `build_calibration()` function:

1. **Orders the points** correctly (TL → TR → BR → BL)
2. **Computes the homography matrix** using OpenCV's `cv2.findHomography()`

```python
from soccer_ai.calibration import build_calibration

calibration = build_calibration(
    image_points=image_points,
    field_width_m=20.0,   # Real width in meters
    field_height_m=10.0,  # Real height in meters
)
```

### Step 3: Project Points

Any pixel coordinate can now be converted to meters:

```python
from soccer_ai.calibration import project_point

# Convert a pixel position to field coordinates
pixel_pos = (300, 250)
field_pos = project_point(pixel_pos, calibration.homography)
# Returns: (9.92, 5.56) in meters
```

---

## Point Ordering

The `_order_points()` function automatically sorts input points:

```
Input (any order)          Output (always ordered)
     ●  ●                    1 ●──────● 2
     ●  ●          →           │      │
                               │      │
                             4 ●──────● 3

Order: top-left (1) → top-right (2) → bottom-right (3) → bottom-left (4)
```

This means users can click points in any order, and the system will correctly identify which is which.

---

## Scale Calculation

### Global Scale (Average)

At initialization, a global meters-per-pixel scale is computed:

```python
# From touch.py
width_px = average(top_edge_length, bottom_edge_length)
height_px = average(left_edge_length, right_edge_length)

global_m_per_px = average(
    field_width_m / width_px,
    field_height_m / height_px
)
```

### Local Scale (Per-Point)

Due to perspective, scale varies across the image. The `m_per_px_at()` function computes local scale:

```python
def m_per_px_at(point):
    # Project point and neighbors
    base = project_point(point, homography)
    dx = project_point((point[0] + 1, point[1]), homography)
    dy = project_point((point[0], point[1] + 1), homography)
    
    # Measure how far 1 pixel moves in meter-space
    scale_x = distance(base, dx)  # meters per pixel in X
    scale_y = distance(base, dy)  # meters per pixel in Y
    
    return (scale_x + scale_y) / 2
```

This handles perspective distortion where objects appear smaller when farther from the camera.

---

## Usage in Speed Calculation

### Without Calibration (pixels)
```python
# Speed in pixels per second
vx = (x2 - x1) / dt
vy = (y2 - y1) / dt
speed_px_s = sqrt(vx² + vy²) * fps
```

### With Calibration (meters)
```python
# Transform positions to field coordinates first
p1_field = project_point(p1_px, homography)
p2_field = project_point(p2_px, homography)

# Speed in meters per second
vx = (p2_field[0] - p1_field[0]) / dt
vy = (p2_field[1] - p1_field[1]) / dt
speed_mps = sqrt(vx² + vy²) * fps

# Convert to km/h
speed_kmh = speed_mps * 3.6
```

---

## Saving & Loading Calibration

### Save
```python
from soccer_ai.calibration import save_calibration

save_calibration("my_calibration.json", calibration)
```

### JSON Format
```json
{
  "field_width_m": 20.0,
  "field_height_m": 10.0,
  "image_points": [[100, 100], [500, 120], [550, 400], [50, 380]],
  "field_points": [[0, 0], [20, 0], [20, 10], [0, 10]],
  "homography": [[...3x3 matrix...]],
  "point_order": ["top-left", "top-right", "bottom-right", "bottom-left"]
}
```

### Load
```python
from soccer_ai.calibration import load_calibration

calibration = load_calibration("my_calibration.json")
```

---

## Configuration Options

In `config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CALIB_RECT_WIDTH_M` | 20.0 | Default rectangle width (meters) |
| `CALIB_RECT_HEIGHT_M` | 10.0 | Default rectangle height (meters) |
| `USE_HOMOGRAPHY` | True | Enable/disable homography globally |

In `options.py` / Streamlit sidebar:
- `use_homography`: Toggle between meter-based and pixel-based calculations
- `calibration_path`: Path to saved calibration JSON file

---

## Tips for Best Results

1. **Mark corners precisely** - Small errors in marking = larger errors in distant areas

2. **Use visible landmarks** - Field lines, cones, or tape that are clearly visible

3. **Keep rectangle on ground level** - All 4 points must be on the same flat plane

4. **Larger rectangles = better accuracy** - More pixels in the calibration area means better precision

5. **Avoid extreme angles** - Very oblique camera angles increase distortion and reduce accuracy

---

## Verifying Calibration

Run the test script to verify homography accuracy:

```bash
python3 test_homography.py
```

This tests:
- ✅ Corner mapping accuracy
- ✅ Inverse transformation
- ✅ Distance calculations
- ✅ Scale variation across field
- ✅ Numerical stability

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Speed shows as px/s | Calibration not loaded | Load a calibration file or mark points |
| Speeds seem wrong | Incorrect field dimensions | Verify the real-world size of your rectangle |
| Distant objects have wrong speed | Extreme perspective | Move camera higher or use larger calibration area |
| "Unable to compute homography" | Points are collinear | Ensure 4 points form a proper quadrilateral |

---

## Code Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit UI                            │
│  User marks 4 points → save_calibration() → calibration.json │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   run_touch_detection()                      │
│  load_calibration() → compute global_m_per_px                │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Per-Frame Processing                      │
│  ball_center_field = project_point(ball_center, homography)  │
│  speed_mps = compute_velocity(field_coords) * fps            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      FrameResult                             │
│  avg_speed_kmh, total_distance_m, shot_speed_mps, etc.       │
└──────────────────────────────────────────────────────────────┘
```
