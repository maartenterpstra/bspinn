"""fastMRI multi-coil loader (streams only what calibration needs) and a writer for
fastMRI-format synthetic volumes with known Biot–Savart ground truth.

fastMRI files: dataset ``kspace`` (slices, coils, H, W) complex64, ``ismrmrd_header`` (XML),
optional ``mask`` (column mask, test/challenge sets), attribute ``acquisition``
(e.g. 'CORPD_FBK' knee, 'AXT2' brain).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .data import MultiCoilData
from .geometry import SliceGeometry, crop_center

_NS = {"m": "http://www.ismrm.org/ISMRMRD"}


def read_header(xml_bytes) -> dict:
    root = ET.fromstring(xml_bytes)

    def _get(path):
        el = root.find(path, _NS)
        return {k: float(el.find(f"m:{k}", _NS).text) for k in ("x", "y", "z")} if el is not None else None

    return {"enc_matrix": _get("m:encoding/m:encodedSpace/m:matrixSize"),
            "enc_fov": _get("m:encoding/m:encodedSpace/m:fieldOfView_mm"),
            "rec_matrix": _get("m:encoding/m:reconSpace/m:matrixSize"),
            "rec_fov": _get("m:encoding/m:reconSpace/m:fieldOfView_mm")}


def plane_from_acquisition(acq: str) -> str:
    acq = (acq or "").upper()
    if acq.startswith("COR"):
        return "coronal"
    if acq.startswith("SAG"):
        return "sagittal"
    return "axial"


def load_fastmri(path: str, calib: int = 48, slices=None, plane: str | None = None,
                 slice_spacing: float | None = None, full: bool = False) -> MultiCoilData:
    """Load a fastMRI volume. With ``full=False`` only the central ``calib`` x ``calib`` k-space of
    each slice is read (enough to fit the coil fields; maps are then evaluated at full resolution
    from the continuous model)."""
    import h5py
    with h5py.File(path, "r") as f:
        ksd = f["kspace"]
        S, N, H, W = ksd.shape
        idx = list(range(S)) if slices is None else [i for i in slices if 0 <= i < S]
        if full:
            ks = np.stack([ksd[i] for i in idx])
        else:
            c = min(calib, H), min(calib, W)
            r0, c0 = H // 2 - c[0] // 2, W // 2 - c[1] // 2
            ks = np.stack([ksd[i, :, r0:r0 + c[0], c0:c0 + c[1]] for i in idx])
        mask = None
        if "mask" in f:
            mk = np.asarray(f["mask"][()]).astype(bool)[None, None, None, :]
            mask = np.broadcast_to(mk, (len(idx), 1, H, W))
            if not full:
                mask = crop_center(mask, ks.shape[-2:])
        hdr = read_header(f["ismrmrd_header"][()]) if "ismrmrd_header" in f else None
        acq = f.attrs.get("acquisition", "")
    if hdr and hdr["enc_fov"] and hdr["enc_matrix"]:
        ps = (hdr["enc_fov"]["x"] / hdr["enc_matrix"]["x"], hdr["enc_fov"]["y"] / hdr["enc_matrix"]["y"])
        thick = hdr["enc_fov"]["z"]
    else:
        ps, thick = (1.0, 1.0), 5.0
    sp = slice_spacing or thick
    pos = (np.asarray(idx) - (S - 1) / 2) * sp
    geom = SliceGeometry((H, W), ps, pos, plane or plane_from_acquisition(acq))
    meta = {"source": path, "acquisition": acq, "calib_only": not full, "slices": idx}
    if not full:
        meta["full_shape"] = (H, W)
    return MultiCoilData(ks.astype(np.complex64), geom, mask=mask, meta=meta)


_HDR = """<?xml version="1.0"?><ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>
<encodedSpace><matrixSize><x>{H}</x><y>{W}</y><z>1</z></matrixSize>
<fieldOfView_mm><x>{fx}</x><y>{fy}</y><z>{fz}</z></fieldOfView_mm></encodedSpace>
<reconSpace><matrixSize><x>{H}</x><y>{W}</y><z>1</z></matrixSize>
<fieldOfView_mm><x>{fx}</x><y>{fy}</y><z>{fz}</z></fieldOfView_mm></reconSpace>
</encoding></ismrmrdHeader>"""


def write_fake_fastmri(path, n_coils=12, n_slices=6, shape=(96, 96), slice_spacing=6.0, seed=0,
                       acquisition="AXT2", noise=0.01):
    """fastMRI-format file built from the synthetic Biot–Savart phantom (array rotated randomly
    per subject); ground-truth maps are written to ``path + '.gt.npy'``."""
    import h5py
    from .data import make_synthetic
    rng = np.random.default_rng(seed)
    d, gt, _ = make_synthetic(n_coils, shape, n_slices, slice_spacing=slice_spacing, noise=noise, seed=seed,
                              rotation=float(rng.uniform(0, 2 * np.pi)))
    H, W = shape
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=d.kspace)
        f.create_dataset("ismrmrd_header", data=_HDR.format(H=H, W=W, fx=H * 1.0, fy=W * 1.0, fz=slice_spacing).encode())
        f.attrs["acquisition"] = acquisition
    np.save(path + ".gt.npy", gt)
    return gt
