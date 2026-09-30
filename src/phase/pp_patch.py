"""Per-image principal point for the 3DGS/MCMC rasterizer, which otherwise assumes the principal point at the image
centre (symmetric frustum from FoVx/FoVy only). Reads cx, cy of every image's COLMAP camera and rebuilds each Camera's
projection matrix with the off-centre terms  P[0,2] = 2 cx/W - 1,  P[1,2] = 2 cy/H - 1  (W, H, cx, cy at COLMAP
resolution; resolution-independent in NDC). The covariance Jacobian depends only on focal, so it is unchanged.

    from pp_patch import patch_principal_points
    patch_principal_points(scene, dataset.source_path)          # after Scene(...) is built
"""
import os

import torch


def _pp_table(source_path):
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
    sp = os.path.join(source_path, "sparse", "0")
    ext = read_extrinsics_binary(os.path.join(sp, "images.bin")); intr = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
    tab = {}
    for im in ext.values():
        c = intr[im.camera_id]
        if c.model == "PINHOLE":
            cx, cy = c.params[2], c.params[3]
        elif c.model == "SIMPLE_PINHOLE":
            cx, cy = c.params[1], c.params[2]
        else:
            continue
        tab[os.path.splitext(im.name)[0]] = (2 * cx / c.width - 1, 2 * cy / c.height - 1)
    return tab


def patch_cameras(cams, tab):
    n, mx = 0, 0.0
    for cam in cams:
        off = tab.get(cam.image_name) or tab.get(os.path.splitext(cam.image_name)[0])
        if off is None or (abs(off[0]) < 1e-9 and abs(off[1]) < 1e-9):
            continue
        P = cam.projection_matrix.clone()                     # stored transposed: P_row_col = projection_matrix[col, row]
        P[2, 0] = off[0]; P[2, 1] = off[1]
        cam.projection_matrix = P
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(P.unsqueeze(0))).squeeze(0)
        n += 1; mx = max(mx, abs(off[0]) * cam.image_width / 2, abs(off[1]) * cam.image_height / 2)
    return n, mx


def patch_principal_points(scene, source_path, verbose=True):
    tab = _pp_table(source_path)
    tot = 0; mxall = 0.0
    for scale in scene.train_cameras:
        for getter in (scene.getTrainCameras, scene.getTestCameras):
            n, mx = patch_cameras(getter(scale), tab); tot += n; mxall = max(mxall, mx)
    if verbose:
        print(f"[pp] off-centre principal point applied to {tot} cameras (max |offset| {mxall:.2f} px at render res)", flush=True)
    return tot
