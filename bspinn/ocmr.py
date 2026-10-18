"""OCMR (https://www.ocmr.info, Chen et al. 2020) reader: ISMRMRD .h5 raw data of 2-D cardiac cine, fully
sampled (fs_*) or prospectively undersampled (us_*). Follows the layout of the dataset's read_ocmr.m / .py:

    kData[kx, ky, kz, coil, phase(frame), set, slice, rep, avg]

Noise-only acquisitions (ACQ_IS_NOISE_MEASUREMENT) are returned separately, for pre-whitening. Asymmetric
echoes (center_sample) and partial-Fourier phase encoding (kspace_encoding_step_1.center) are placed on the
encoded matrix. ``cine_slices`` yields per slice the (coils, x, y, frames) k-space with the 2x readout
oversampling removed (image-domain crop to the recon matrix along x), averages summed.
Needs the ``ismrmrd`` package (pip install ismrmrd)."""
import numpy as np


def _ismrmrd():
    try:
        import ismrmrd
        import ismrmrd.xsd
    except ImportError as e:            # pragma: no cover
        raise ImportError("reading OCMR needs the ismrmrd package: pip install ismrmrd") from e
    return ismrmrd


def read_ocmr(path):
    """-> dict(kspace (kx, ky, kz, coil, phase, set, slice, rep, avg) complex64, noise (n, coil) or None,
    header fields: enc_matrix, recon_matrix, fov_mm (encoded), recon_fov_mm, field_T, n_coils)."""
    ismrmrd = _ismrmrd()
    dset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
    try:
        hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = hdr.encoding[0]
        eN = (enc.encodedSpace.matrixSize.x, enc.encodedSpace.matrixSize.y, enc.encodedSpace.matrixSize.z)
        rN = (enc.reconSpace.matrixSize.x, enc.reconSpace.matrixSize.y, enc.reconSpace.matrixSize.z)
        fov = (enc.encodedSpace.fieldOfView_mm.x, enc.encodedSpace.fieldOfView_mm.y, enc.encodedSpace.fieldOfView_mm.z)
        rfov = (enc.reconSpace.fieldOfView_mm.x, enc.reconSpace.fieldOfView_mm.y, enc.reconSpace.fieldOfView_mm.z)
        lim = enc.encodingLimits

        def nmax(field):
            f = getattr(lim, field, None)
            return (int(f.maximum) + 1) if f is not None and f.maximum is not None else 1

        ky_c = lim.kspace_encoding_step_1.center if lim.kspace_encoding_step_1 is not None else None
        kz_c = lim.kspace_encoding_step_2.center if lim.kspace_encoding_step_2 is not None else None
        n_coil = int(hdr.acquisitionSystemInformation.receiverChannels)
        dims = (eN[0], eN[1], max(1, eN[2]), n_coil, nmax("phase"), nmax("set"), nmax("slice"),
                nmax("repetition"), nmax("average"))
        k = np.zeros(dims, np.complex64)
        y_off = 0 if ky_c is None else eN[1] // 2 - int(ky_c)
        z_off = 0 if (kz_c is None or eN[2] <= 1) else eN[2] // 2 - int(kz_c)
        noise = []
        for i in range(dset.number_of_acquisitions()):
            acq = dset.read_acquisition(i)
            if acq.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                noise.append(np.asarray(acq.data).T)               # (samples, coils)
                continue
            d = np.asarray(acq.data)                               # (coils, samples)
            ix = acq.idx
            ns = d.shape[1]
            x0 = eN[0] // 2 - int(acq.center_sample)               # asymmetric echo: zeros before
            x0 = min(max(x0, 0), eN[0] - ns)
            ky, kz = int(ix.kspace_encode_step_1) + y_off, int(ix.kspace_encode_step_2) + z_off
            if not (0 <= ky < dims[1] and 0 <= kz < dims[2]):
                continue
            k[x0:x0 + ns, ky, kz, :, ix.phase, ix.set, ix.slice, ix.repetition, ix.average] += d.T
        info = {"enc_matrix": eN, "recon_matrix": rN, "fov_mm": fov, "recon_fov_mm": rfov, "n_coils": n_coil,
                "field_T": getattr(hdr.acquisitionSystemInformation, "systemFieldStrength_T", None)}
    finally:
        dset.close()
    return {"kspace": k, "noise": np.concatenate(noise, 0) if noise else None, **info}


def _remove_oversampling(k, n_recon):
    """k (..., kx, ky) -> readout cropped to n_recon in the image domain (removes the 2x oversampling)."""
    nx = k.shape[-2]
    if n_recon >= nx:
        return k
    img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=-2), axis=-2, norm="ortho"), axes=-2)
    c0 = nx // 2 - n_recon // 2
    img = img[..., c0:c0 + n_recon, :]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=-2), axis=-2, norm="ortho"), axes=-2)


def cine_slices(path, slices="all", set_=0, rep=0):
    """Yields (slice index, k (coils, x, y, frames) complex64, info) with info: noise (n, coils) or None,
    pixel_spacing_mm (x, y), field_T, file. x = readout (oversampling removed), y = phase encoding."""
    d = read_ocmr(path)
    k = d["kspace"]                                                # (kx, ky, kz, coil, phase, set, slice, rep, avg)
    ns = k.shape[6]
    sel = range(ns) if slices == "all" else ([ns // 2] if slices == "mid" else [int(s) for s in slices])
    rx = int(d["recon_matrix"][0]) if d["recon_matrix"][0] else k.shape[0]
    for s in sel:
        ks = k[:, :, 0, :, :, set_, s, rep, :].sum(-1)            # (kx, ky, coil, frames), averages summed
        ks = np.transpose(ks, (2, 0, 1, 3))                        # (coil, kx, ky, frames)
        ks = np.moveaxis(_remove_oversampling(np.moveaxis(ks, -1, 1), rx), 1, -1)
        fov_x = d["recon_fov_mm"][0] or d["fov_mm"][0] / 2
        sp = (fov_x / ks.shape[1], d["fov_mm"][1] / ks.shape[2])
        yield s, ks.astype(np.complex64), {"noise": d["noise"], "pixel_spacing_mm": sp, "field_T": d["field_T"],
                                           "file": str(path), "n_slices": ns}
