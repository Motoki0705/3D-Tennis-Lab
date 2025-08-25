# Step 2.2: Core geometry functions
import numpy as np
import cv2


def estimate_homography(src_xy, dst_xy, ransac_thresh_px: float):
    """Estimates homography, returns H matrix and inlier mask."""
    src = np.asarray(src_xy, np.float32)
    dst = np.asarray(dst_xy, np.float32)
    # Use RANSAC only if we have more than the minimum required points (4)
    method = cv2.RANSAC if len(src) > 4 else 0
    H, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_thresh_px)
    return H, None if mask is None else mask.ravel().astype(bool)


def project_H(H, pts_xy):
    """Projects points using a homography matrix."""
    pts = np.asarray(pts_xy, np.float32)
    ones = np.ones((pts.shape[0], 1), np.float32)
    hom = np.hstack([pts, ones])
    prj = (H @ hom.T).T
    # Normalize by the 3rd coordinate
    prj = prj[:, :2] / prj[:, 2:3]
    return prj


def rmse_inliers(H, src_xy, dst_xy, inlier_mask):
    """Calculates the Root Mean Square Error for inliers."""
    if H is None or inlier_mask is None or not inlier_mask.any():
        return None
    src = np.asarray(src_xy, np.float32)[inlier_mask]
    dst = np.asarray(dst_xy, np.float32)[inlier_mask]
    pred = project_H(H, src)
    err = np.linalg.norm(pred - dst, axis=1)
    return float(np.sqrt(np.mean(err**2)))
