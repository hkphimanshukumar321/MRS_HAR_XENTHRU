"""Purpose: Calculates the participant’s position from UWB measurements using trilateration. If the HAR setup includes multiple UWB radar sensors placed at known locations, this script uses their distance measurements to estimate the subject’s (x, y) coordinates. The output is sequential position data (which could correspond to the trajectory of the subject over time). What it does:
Defines a trilateration function that takes known anchor positions and distance measurements, and solves for the 2D coordinates of the target. This uses the geometric intersection of circles (one around each anchor with radius equal to measured distance).
The script can read a file containing distance measurements (for example, a CSV or NPZ with distances from each anchor over time) and computes the position for each time step.
The resulting positions are saved (e.g., in data/processed/positions.npz) for further analysis or for integration with the real-time dashboard.
Usage:
bash
Copy
python compute_position.py --input data/kaggle/distances.csv --output data/processed/positions.npz
You need to provide an input file with distance measurements. For example, a CSV where each row is [time, d_anchor1, d_anchor2, d_anchor3, ...]. The script will use the first row (or a specified reference) as needed for trilateration. Adjust anchor coordinates in the script to match your setup."""

#code starts from here

#!/usr/bin/env python3
"""
compute_position.py: Compute participant positions using trilateration from UWB distances.

Given distances from multiple UWB radar anchors to the participant, this script computes the (x, y) position of the participant at each time step.
Anchor positions (in 2D coordinates) should be defined in this script as per the deployment setup.

Usage:
    python compute_position.py --input <distances_file> --output <positions_file>
Example:
    python compute_position.py --input data/kaggle/distances.csv --output data/processed/positions.npz

The input file is expected to contain distance measurements from at least 3 anchors for each timestamp.
"""
import numpy as np
import argparse

# Define the known anchor positions (in meters, for example) in a 2D plane.
# Modify these coordinates to match the actual anchor placement in your environment.
anchor_positions = np.array([
    [0.0, 0.0],   # Anchor 1 at (x1, y1)
    [5.0, 0.0],   # Anchor 2 at (x2, y2)
    [2.5, 4.33]   # Anchor 3 at (x3, y3) forming roughly an equilateral triangle for robustness
    # Add more anchors here if available (the algorithm will use least-squares if >3 anchors)
])

def trilateration(anchors, distances):
    """
    Solve for (x, y) given anchor coordinates and distances using trilateration.
    - anchors: numpy array of shape (N, 2) with N anchor positions.
    - distances: numpy array of shape (N,) with distances from each anchor to the target.
    Returns: (x, y) coordinates of the target.
    """
    num_anchors = anchors.shape[0]
    if num_anchors < 2 or len(distances) < 2:
        raise ValueError("At least 2 anchors (and distances) are required for trilateration in 2D.")
    # Use the first anchor as reference and create equations relative to it.
    x1, y1 = anchors[0]
    d1 = distances[0]
    # Set up linear equations: for each other anchor i,
    # (x - x1)^2 + (y - y1)^2 = d1^2
    # (x - xi)^2 + (y - yi)^2 = di^2
    # Subtract second equation from first to eliminate x^2 and y^2 terms, yielding linear equation in x and y.
    A_list = []
    b_list = []
    for i in range(1, num_anchors):
        xi, yi = anchors[i]
        di = distances[i]
        # Linear coefficients for x and y
        A_x = 2 * (xi - x1)
        A_y = 2 * (yi - y1)
        # Right-hand side
        # Note: (x1^2 + y1^2 - xi^2 - yi^2) + (di^2 - d1^2) = 0 after moving terms; rearranged:
        # A_x * x + A_y * y = d1^2 - di^2 - (x1^2 + y1^2 - xi^2 - yi^2)
        b_val = d1**2 - di**2 - (x1**2 + y1**2 - xi**2 - yi**2)
        A_list.append([A_x, A_y])
        b_list.append(b_val)
    A = np.array(A_list)
    b = np.array(b_list)
    # Solve the linear system A * [x, y]^T = b in a least-squares sense
    try:
        position = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # If A is singular or not square (more equations than unknowns), use least squares
        position, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return position  # (x, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute positions from UWB anchor distances using trilateration.")
    parser.add_argument("--input", "-i", required=True, help="Input file with distance measurements (CSV or NPZ format).")
    parser.add_argument("--output", "-o", default="data/processed/positions.npz", help="Output file for computed positions (NPZ format).")
    args = parser.parse_args()
    
    # Load distance measurements
    if args.input.endswith(".npz"):
        data = np.load(args.input)
        # Expect distances under a known key or as an array
        if "distances" in data:
            dist_array = data["distances"]
        else:
            # If not labeled, assume first array in npz
            dist_array = data[list(data.files)[0]]
    else:
        # Assume CSV: each row has [time, d1, d2, d3, ...] or [d1, d2, d3, ...] if no time column
        raw = np.loadtxt(args.input, delimiter=',')
        # If first column looks like time (increasing values), drop it
        if raw.shape[1] > len(anchor_positions):
            dist_array = raw[:, 1:]  # drop the first column
        else:
            dist_array = raw
    
    # Ensure dist_array shape is (T, N_anchors)
    if dist_array.ndim == 1:
        # Single measurement case, make it 2D
        dist_array = dist_array.reshape(1, -1)
    
    positions = []
    for dist in dist_array:
        if np.any(np.isnan(dist)) or np.any(dist <= 0):
            # Skip or handle invalid distance measurements (nan or non-positive)
            positions.append([np.nan, np.nan])
        else:
            pos = trilateration(anchor_positions, dist)
            positions.append(pos.tolist())
    positions = np.array(positions)
    
    # Save positions to NPZ file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, positions=positions)
    print(f"Computed {len(positions)} positions. Saved to {args.output}.")


"""Comments: Adjust the anchor_positions array to reflect the actual coordinates of your UWB anchors in the environment (e.g., measuring in meters from a reference point). 
The script as given assumes 3 anchors; it will work with more anchors (using a least-squares solution if redundancy exists). 
The input data handling assumes a CSV with distances; you may need to tweak if your Kaggle data provides distances differently or if the distances need to be extracted from the radar signals (which is a complex task on its own). 
The trilateration function can be imported by other modules. For instance, you might use it in the real-time inference loop to track the subject’s position continuously. 
Future enhancements: If you have 3D coordinates (x, y, z), you would extend the function to solve for three dimensions (needs at least 4 anchors). Additionally, incorporating filtering (like a Kalman filter) on the output positions can smooth out noise in real-time.  
This module currently outputs raw computed positions for each time step, which can be directly used by the dashboard for visualization or further analysis."""
