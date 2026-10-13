import math

import numpy as np
import torch

from bspinn.encoding import NUDFTOp
from bspinn.geometry import SliceGeometry, fft2c
from bspinn.models import HarmonicPINN, WireLoopCoils
from bspinn.physics import biot_savart_polyline, maxwell_static_residual

torch.manual_seed(0)


def test_biot_savart_on_axis_loop():
    a, M = 0.5, 600
    t = torch.linspace(0, 2 * math.pi, M + 1)[:-1]
    v = torch.stack([a * torch.cos(t), a * torch.sin(t), torch.zeros(M)], -1).double()
    z = torch.tensor([0.0, 0.3, 1.0]).double()
    B = biot_savart_polyline(v, torch.stack([0 * z, 0 * z, z], -1))
    ref = 2 * math.pi * a ** 2 / (a ** 2 + z ** 2) ** 1.5      # mu0 I / 4pi units
    assert torch.allclose(B[:, 2], ref, rtol=1e-4)
    assert B[:, :2].abs().max() < 1e-8


def test_wire_field_is_source_free():
    m = WireLoopCoils(3)
    with torch.no_grad():
        m.verts.normal_(); m.verts.mul_(0.3); m.verts[..., 0] += 2.0; m.amp.fill_(1.0)
    pts = torch.rand(400, 3) * 2 - 1
    div, curl = maxwell_static_residual(m.field, pts, 3)
    assert div.max() < 1e-4 and curl.max() < 1e-4


class _Fixed(HarmonicPINN):
    def __init__(self, f):
        super().__init__(2)
        self.f = f

    def potential(self, x):
        return self.f(x)[..., None]


def test_pinn_operators_on_known_potentials():
    pts = torch.rand(300, 3) * 2 - 1
    harmonic = _Fixed(lambda x: x[..., 0] ** 2 - x[..., 1] ** 2 + x[..., 2] * x[..., 0])
    assert harmonic.physics_loss(pts) < 1e-10
    nonharm = _Fixed(lambda x: x[..., 0] ** 2 + x[..., 1] ** 2)
    assert nonharm.physics_loss(pts) > 0.1
    div, curl = maxwell_static_residual(nonharm.field, pts, 2)
    assert curl.max() < 1e-5 and div.min() > 0.1        # B = -grad(phi): curl-free by construction


def test_pinn_variable_coils_and_correction_starts_at_wires():
    for n in (3, 11):
        w = WireLoopCoils(n)
        with torch.no_grad():
            w.verts.normal_(); w.verts[..., 0] += 2.0; w.amp.fill_(1.0); w.theta.uniform_(-3, 3)
        p = HarmonicPINN(n, base=w)
        pts = torch.rand(50, 3) - 0.5
        assert torch.allclose(p.sensitivity(pts), w.sensitivity(pts), atol=1e-6)
        assert p.physics_loss(pts) == 0


def test_nudft_matches_fft_on_grid_and_is_adjoint():
    h = w = 8
    kr, kc = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing="ij")
    traj = torch.stack([kr.flatten(), kc.flatten()], -1).float()
    op = NUDFTOp(traj, (h, w))
    x = torch.randn(2, h, w, dtype=torch.complex64)
    assert torch.allclose(op.forward(x), fft2c(x).reshape(2, -1), atol=1e-4)
    tr = torch.rand(37, 2) * 8 - 4
    op = NUDFTOp(tr, (h, w))
    y = torch.randn(2, 37, dtype=torch.complex64)
    lhs = (op.forward(x).conj() * y).sum()
    rhs = (x.conj() * op.adjoint(y)).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_lowres_grid_spans_same_fov():
    g = SliceGeometry((64, 80), (1.0, 1.0), np.array([0.0]))
    full, low = g.grid(), g.grid((16, 20))
    assert torch.allclose(full[0, 32, 40], low[0, 8, 10])          # centred-FFT origin pixel
    assert abs(float(full[..., 0].max() - low[..., 0].max())) < 0.1


def test_end_to_end_synthetic_beats_baseline():
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, fit
    from bspinn.report import evaluate
    data, gt, _ = make_synthetic(6, (48, 48), noise=0.005, seed=1)
    cfg = FitConfig(model="wires", calib=24, iters_wires=300, iters_auto=100, log_every=10 ** 9)
    model, prob, hist, extra = fit(data, cfg, log=lambda *a: None)
    rep, _, _ = evaluate(model, prob, extra, gt=gt, log=lambda *a: None)
    v = rep["vs_ground_truth"]
    assert v["biot_savart_model"]["absolute_err"] < v["lowres_division_baseline"]["absolute_err"]
    assert rep["physics_rel_rms"]["div"] < 1e-3


def test_batched_nudft_adjoint():
    from bspinn.encoding import NUDFTOpBatched
    t = torch.rand(3, 20, 2) * 8 - 4
    v = torch.rand(3, 20) > 0.2
    op = NUDFTOpBatched(t, v, (8, 8))
    x = torch.randn(3, 2, 8, 8, dtype=torch.complex64)
    y = torch.randn(3, 2, 20, dtype=torch.complex64)
    assert torch.allclose((op.forward(x).conj() * y).sum(), (x.conj() * op.adjoint(y)).sum(), rtol=1e-4)


def test_varpro_multiset_reduces_to_single():
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem
    data, gt, _ = make_synthetic(4, (32, 32), noise=0.0, seed=2)
    p = Problem(data, FitConfig(calib=16, calib_oversample=1))      # varpro path
    C = torch.randn(4, p.coords.shape[0], dtype=torch.complex64)
    one = p.data_loss(C)
    # a second set that is identically zero must not change the residual
    two = p.data_loss(torch.stack([C, torch.zeros_like(C)]))
    assert torch.allclose(one, two, rtol=1e-3)


def test_fold_sets_beat_single_set_on_folded_data():
    from bspinn.data import make_folded_synthetic
    from bspinn.fit import FitConfig, fit
    from bspinn.report import evaluate
    errs = {}
    for fs in (0, 1):
        data, gt, _ = make_folded_synthetic(6, (48, 48), noise=0.005, seed=1)
        cfg = FitConfig(model="wires", calib=24, iters_wires=250, iters_auto=60, fold_sets=fs, log_every=10 ** 9)
        model, prob, hist, extra = fit(data, cfg, log=lambda *a: None)
        rep, _, _ = evaluate(model, prob, extra, gt=gt, log=lambda *a: None)
        errs[fs] = rep["vs_ground_truth"]["biot_savart_model"]["absolute_err"]
    assert errs[1] < errs[0]


def test_fastmri_roundtrip(tmp_path):
    from bspinn.fastmri import load_fastmri, write_fake_fastmri
    p = str(tmp_path / "file_knee_CORPD_000.h5")
    write_fake_fastmri(p, n_coils=5, n_slices=3, shape=(40, 40), acquisition="CORPD_FBK")
    d = load_fastmri(p, calib=16)
    assert d.kspace.shape == (3, 5, 16, 16) and d.geom.shape == (40, 40)
    assert d.geom.plane == "coronal" and d.meta["calib_only"]


def test_dynamic_frames_are_biot_savart():
    from bspinn.dynamic import DynamicCoils
    from bspinn.physics import divergence_and_curl
    w = WireLoopCoils(3)
    with torch.no_grad():
        w.verts.normal_(); w.verts.mul_(0.2); w.verts[..., 0] += 2.0; w.amp.fill_(1.0)
    st = HarmonicPINN(3, base=w)
    dyn = DynamicCoils(st, T=5, K=2)
    with torch.no_grad():
        dyn.V.normal_(); dyn.V.mul_(0.05)
    # moved wires (frame 3) must still give a source-free field: point-wise autograd check
    xx = (torch.rand(60, 3) - 0.5).requires_grad_(True)
    F = dyn.frame_field(xx, [3])[0, 1]
    d, c = divergence_and_curl(F, xx)
    assert d.abs().max() < 1e-3 * F.norm(dim=-1).max() and c.norm(dim=-1).max() < 1e-3 * F.norm(dim=-1).max()


# ---------------------------------------------------------------- modularity & phase handling
import pytest  # noqa: E402


def test_phase_reference_keeps_relative_phase_and_signal():
    from bspinn.recon import phase_reference, virtual_body_coil
    g = torch.Generator().manual_seed(0)
    C = torch.randn(2, 5, 12, 12, dtype=torch.complex64, generator=g)
    m = torch.randn(2, 12, 12, dtype=torch.complex64, generator=g)
    for mode in ("inati", "coil", "model"):
        Cr, mr, info, Phi = phase_reference(C, mode, None, m)
        rel = lambda X: X[:, :, None] * X[:, None].conj()             # all C_j C_k^*
        assert torch.allclose(rel(Cr), rel(C), atol=1e-4)            # inter-coil phases unchanged
        assert torch.allclose(Cr * mr[:, None], C * m[:, None], atol=1e-4)   # signal model unchanged
    d = virtual_body_coil(C)
    Cr, _, _, _ = phase_reference(C, "inati", None, None, d)
    T = torch.einsum("n,snhw->shw", d.conj(), Cr)
    assert T.imag.abs().max() < 1e-4 and T.real.min() > 0           # virtual body coil is real positive
    # the robust weights never make the reference weaker than Inati's eigenvector weights
    s = lambda d: (torch.einsum("n,snhw->shw", d.conj(), C).abs() / C.abs().pow(2).sum(1).sqrt()).min()
    assert s(virtual_body_coil(C, robust=True)) >= s(virtual_body_coil(C, robust=False)) - 1e-4


def test_invalid_option_combinations_are_rejected():
    from bspinn.fit import FitConfig
    with pytest.raises(ValueError):
        FitConfig(model="free", fold_sets=1).validate()
    with pytest.raises(ValueError):
        FitConfig(model="wires", free_sets=1).validate()
    with pytest.raises(ValueError):
        FitConfig(fold_sets=1).validate("dynamic")
    with pytest.raises(ValueError):
        FitConfig(model="wires").validate("shared")
    FitConfig().validate("fit"); FitConfig().validate("dynamic"); FitConfig().validate("shared")


@pytest.mark.parametrize("opts", [
    dict(model="wires"),
    dict(model="pinn"),
    dict(model="pinn", field_mode="complex"),
    dict(model="free"),
    dict(model="wires", fold_sets=1),
    dict(model="pinn", shift_sets=1),
    dict(model="pinn", free_sets=1),
    dict(model="pinn", slice_batch=1, phase_ref="coil"),
    dict(model="pinn", phase_ref="model", data_term="kspace"),
    dict(model="pinn", phase_ref="inati-eig", init="wires", loops_per_coil=2),
])
def test_option_matrix(opts):
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, fit
    from bspinn.report import evaluate
    data, gt, _ = make_synthetic(4, (32, 32), n_slices=2, noise=0.005, seed=3)
    cfg = FitConfig(calib=16, iters_auto=10, iters_wires=20, iters_pinn=10, iters_free_init=10,
                    n_colloc=64, log_every=10 ** 9, **opts)
    model, prob, hist, extra = fit(data, cfg, log=lambda *a: None)
    rep, C, img = evaluate(model, prob, extra, gt=gt, log=lambda *a: None)
    assert torch.is_complex(C) and torch.isfinite(C).all() and C.shape == (2, 4, 32, 32)
    assert rep["phase_reference"]["phase_ref"] == cfg.phase_ref
    assert (rep["physics_rel_rms"] is None) == (cfg.model == "free")


def test_channel_mixing_is_exact():
    """Whitening + compression record a mixing matrix; a physical-coil model evaluated through it
    gives exactly the data loss of the mixed (virtual-channel) maps."""
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem
    data, gt, _ = make_synthetic(6, (32, 32), noise=0.01, noise_corr=1.0, seed=4)
    raw = data.kspace.copy()
    data.prewhiten(); data.compress(4, calib=16)
    M = data.meta["mix"]
    assert M.shape == (4, 6)
    assert np.allclose(data.kspace, np.einsum("vn,sn...->sv...", M, raw), atol=1e-4)
    p_phys = Problem(data, FitConfig(calib=16, model_channels="physical", calib_oversample=1))
    p_data = Problem(data, FitConfig(calib=16, model_channels="data", calib_oversample=1))
    assert p_phys.N_model == 6 and p_data.N_model == 4 and p_phys.N == 4
    C = torch.randn(6, p_phys.coords.shape[0], dtype=torch.complex64)
    Cv = torch.as_tensor(M) @ C
    assert torch.allclose(p_phys.data_loss(C), p_data.data_loss(Cv), rtol=1e-4)


def test_fit_with_whitening_and_compression_runs():
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, fit
    from bspinn.report import evaluate
    data, gt, _ = make_synthetic(6, (32, 32), noise=0.01, noise_corr=1.0, seed=5)
    cfg = FitConfig(calib=16, iters_auto=10, iters_wires=20, iters_pinn=10, n_colloc=64, compress=4, log_every=10 ** 9)
    model, prob, hist, extra = fit(data, cfg, log=lambda *a: None)
    rep, C, img = evaluate(model, prob, extra, gt=gt, log=lambda *a: None)
    assert C.shape[1] == 4 and model.n_coils == 6
    assert "biot_savart_model_physical_channels" in rep["vs_ground_truth"]


def test_phase_vortex_counter():
    from bspinn.recon import phase_vortices
    y, x = torch.meshgrid(torch.linspace(-1, 1, 41), torch.linspace(-1, 1, 41), indexing="ij")
    sup = torch.ones(41, 41, dtype=torch.bool)
    C = torch.stack([(x + 0.03) + 1j * (y + 0.02),            # one +1 vortex (field zero)
                     ((x + 0.03) - 1j * (y + 0.02)) * ((x - 0.51) + 1j * (y + 0.02)),  # -1 and +1
                     torch.exp(1j * 3 * x) * (2 + x)])[None]    # smooth phase ramp, no zero
    v = phase_vortices(C, sup)
    assert v["per_coil"] == [1, 2, 0] and v["net_winding"] == [1, 0, 0]


def test_init_z_offset_breaks_mirror_symmetry():
    """In-plane loops get zero gradient w.r.t. their out-of-plane position (z -> -z symmetry)."""
    w = WireLoopCoils(2, 1, 16, 1)
    P = torch.stack(torch.meshgrid(torch.linspace(-.5, .5, 9), torch.linspace(-.5, .5, 9), indexing="ij"), -1).reshape(-1, 2)
    pts = torch.cat([P, torch.zeros(len(P), 1)], 1)
    wts = torch.ones(2, len(P))
    for off, expect_zero in ((0.0, True), (0.15, False)):
        w.init_from_data(pts, wts, pts, normal_axis=2, z_offset=off)
        v = w.verts.detach().clone().requires_grad_(True)
        (w.field(pts, verts=v)[..., :2] ** 2).sum().backward()
        g = v.grad.sum((-2, -3))                                  # rigid translation gradient (N,3)
        gz = (g[:, 2].abs() / g[:, :2].norm(dim=-1)).max().item()
        assert (gz < 1e-3) == expect_zero, (off, gz)


def _small_case(noise=0.01):
    from bspinn.data import make_synthetic
    from bspinn.methods import EvalCase
    d, C, obj = make_synthetic(6, (64, 64), noise=noise, seed=0)
    return EvalCase("t", d.kspace[0], np.ones((64, 64), bool), (16, 16), sigma=noise, gt_maps=C[0], gt_object=obj[0])


def test_evaluation_oracle_sits_on_noise_floor_and_gfactor_is_one_at_R1():
    from bspinn import evaluation as ev
    from bspinn.methods import lowres
    case = _small_case()
    obj = ev.object_mask(case)
    k_or = ev.kspace_consistency(case.gt_maps, case)
    assert abs(k_or["outside_acs"]["residual_over_noise_floor"] - 1) < 0.05
    k_lr = ev.kspace_consistency(lowres(case), case)
    assert k_lr["outside_acs"]["residual_over_noise_floor"] > k_or["outside_acs"]["residual_over_noise_floor"]
    g, _ = ev.gfactor(ev.normalise(case.gt_maps), 1, 1, obj)
    assert abs(g["mean"] - 1) < 1e-4 and abs(g["max"] - 1) < 1e-4
    props = ev.map_properties(case.gt_maps, case, obj)
    assert props["gt_subspace_err"] < 1e-6 and props["gt_rel_phase_err_deg"] < 1e-3 and props["coverage"] == 1.0


def test_cg_sense_fully_sampled_equals_combination():
    from bspinn import evaluation as ev
    from bspinn.methods import ifft2c
    case = _small_case(0.0)
    Cn = ev.normalise(case.gt_maps)
    x = ev.cg_sense(case.ksp, np.ones(case.shape, bool), Cn, lam=0.0, iters=60)
    m = (Cn.conj() * ifft2c(case.ksp)).sum(0)
    assert np.linalg.norm(x - m) / np.linalg.norm(m) < 1e-4


def test_walsh_and_inati_recover_synthetic_subspace():
    from bspinn import evaluation as ev
    from bspinn.methods import inati, walsh
    case = _small_case(0.001)
    obj = ev.object_mask(case)
    for f in (walsh, inati):
        assert ev.map_properties(f(case), case, obj)["gt_subspace_err"] < 5e-3


def test_oversampled_calibration_grid_setup():
    """calib_oversample=2: maps/image live on a 2x grid over the same FOV; only the central
    calibration box of that grid's k-space is observed, so the k-space data term is used."""
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem
    data, gt, obj = make_synthetic(4, (32, 32), noise=0.0, seed=1)
    p = Problem(data, FitConfig(calib=16, calib_oversample=2))
    assert p.data_term == "kspace" and p.calib == (32, 32) and p.kcalib == (16, 16)
    assert float(p.mk.sum()) == 16 * 16 * p.mk.shape[0]


# ----------------------------------------------------------------------------- 3-D
def test_volume_grid_maps_array_axes_to_physical_axes():
    from bspinn.volume import VolumeGeometry
    g = VolumeGeometry((8, 6, 4), (2.0, 1.0, 1.0), "zyx").grid()
    assert g.shape == (8, 6, 4, 3)
    assert torch.allclose(g[:, 0, 0, 2], torch.linspace(float(g[0, 0, 0, 2]), float(g[-1, 0, 0, 2]), 8))   # z along axis 0
    assert float(g[..., 0].std()) > 0 and float(g[:, :, 0, 0].std()) == 0                                    # x along axis 2


def test_gfactor_3d_is_one_without_acceleration():
    from bspinn import evaluation as ev
    rng = np.random.default_rng(0)
    C = ev.normalise(rng.normal(size=(6, 8, 8, 8)) + 1j * rng.normal(size=(6, 8, 8, 8)))
    obj = np.ones((8, 8, 8), bool)
    g, _ = ev.gfactor(C, (1, 1), (1, 2), obj)
    assert abs(g["mean"] - 1) < 1e-5
    g2, _ = ev.gfactor(C, (2, 2), (1, 2), obj)
    assert g2["mean"] >= 1


def test_coarse_correction_matches_exact_network():
    from bspinn.fit import FitConfig, make_pinn
    from bspinn.models import WireLoopCoils
    from bspinn.volume import CoarseCorrection, Problem3D, make_synthetic3d
    d, gt, obj = make_synthetic3d(4, (16, 16, 16), noise=0.0)
    for cg, tol in ((8, 0.05), (16, 1e-5)):
        cfg = FitConfig(calib=8, calib_oversample=2, correction_grid=cg)
        p = Problem3D(d, cfg)
        w = WireLoopCoils(4, 1, 8, 1)
        with torch.no_grad():
            w.verts.normal_(0, 0.3); w.verts.add_(torch.tensor([1.6, 0, 0])); w.amp.fill_(1)
        pinn = make_pinn(p, cfg, w, 1)
        torch.nn.init.normal_(pinn.net[-1].weight, std=0.05)
        exact = pinn.correction_field(p.coords, create_graph=False)
        wrapped = CoarseCorrection(pinn, p).field(p.coords, create_graph=False) - pinn._base_field(p.coords)
        err = float((wrapped - exact).norm() / exact.norm())
        assert err < tol, (cg, err)


def test_fit3d_recovers_synthetic_volume_and_cli_runs(tmp_path):
    import json
    from bspinn.cli import main
    main(["fit3d", "--synthetic", "--size", "24", "--coils", "6", "--noise-level", "0.003", "--out", str(tmp_path),
          "--calib", "8", "--iters-auto", "40", "--iters-wires", "200", "--iters-pinn", "40", "--threads", "1",
          "--correction-grid", "8", "--model", "pinn"])
    rep = json.load(open(tmp_path / "report.json"))
    assert rep["vs_ground_truth"]["subspace_err"] < 2e-2
    assert np.load(tmp_path / "maps.npy").shape == (6, 24, 24, 24)


def test_orientation_auto_finds_b0_axis():
    from bspinn.fit import FitConfig
    from bspinn.volume import VolumeGeometry, make_synthetic3d, select_orientation
    d, gt, obj = make_synthetic3d(6, (20, 20, 20), noise=0.003)
    d.geom = VolumeGeometry(d.geom.shape, d.geom.voxel, "yxz")              # wrong guess on purpose
    best, scores = select_orientation(d, FitConfig(calib=8, iters_auto=80, threads=1), log=lambda *a: None)
    assert best == "zyx", scores


# ----------------------------------------------------------------------------- Helmholtz (wave) model
def test_wave_wires_reduce_to_biot_savart_and_match_full_wave_reference():
    from bspinn.data import synthetic_array
    from bspinn.models import WaveWireLoopCoils, WireLoopCoils, tissue_k2
    from bspinn.volume import wave_loop_fields
    verts, _ = synthetic_array(3, n_rings=2)
    pts = torch.rand(200, 3) * 1.2 - 0.6
    static = WireLoopCoils(3, 1, 48, 1)
    w0 = WaveWireLoopCoils(3, 1, 48, 1, k2_init=0)
    k2 = tissue_k2(3.0, 50, 0.6, 0.12)
    wk = WaveWireLoopCoils(3, 1, 48, 1, k2_init=k2, n_quad=4)
    for m in (static, w0, wk):
        with torch.no_grad():
            m.verts.copy_(torch.from_numpy(verts)); m.amp.fill_(1)
    B0 = w0.field(pts).detach()
    assert torch.allclose(B0[..., 0], static.field(pts)[..., 0].detach(), atol=1e-5) and B0[..., 1].abs().max() == 0
    Bk = wk.field(pts).detach()
    G = wave_loop_fields(verts, pts, k2, n_r=16, n_phi=64)
    err = np.linalg.norm((Bk[..., 0] + 1j * Bk[..., 1]).numpy() - G) / np.linalg.norm(G)
    assert err < 1e-2, err


def test_full_wave_reference_solves_helmholtz_and_is_divergence_free():
    from bspinn.data import synthetic_array
    from bspinn.models import tissue_k2
    from bspinn.volume import wave_loop_fields
    verts, _ = synthetic_array(2, n_rings=2)
    k2 = tissue_k2(3.0, 50, 0.6, 0.12)
    p0, h = torch.tensor([[0.1, 0.2, -0.1]], dtype=torch.float64), 1e-3
    f = lambda p: wave_loop_fields(verts, p, k2)[:, 0, :]
    F0, lap, div = f(p0), 0, 0
    for d in range(3):
        e = torch.zeros(1, 3, dtype=torch.float64); e[0, d] = h
        Fp, Fm = f(p0 + e), f(p0 - e)
        lap = lap + (Fp - 2 * F0 + Fm) / h ** 2
        div = div + (Fp[:, d] - Fm[:, d]) / (2 * h)
    assert np.abs(lap + k2 * F0).max() < 1e-4 * np.abs(k2 * F0).max()
    assert np.abs(div).max() < 1e-3 * np.abs(F0).max()


def test_helmholtz_needs_3d_and_wave_maps_match_training():
    from bspinn.fit import FitConfig, make_pinn
    from bspinn.models import WaveWireLoopCoils, tissue_k2
    from bspinn.volume import CoarseCorrection, Problem3D, make_synthetic3d, volume_maps
    with pytest.raises(ValueError):
        FitConfig(physics="helmholtz").validate("fit")
    k2 = tissue_k2(3.0, 50, 0.6, 0.12)
    d, gt, obj = make_synthetic3d(4, (16, 16, 16), noise=0.0, wave_k2=k2, voxel=(15, 15, 15))
    cfg = FitConfig(calib=8, physics="helmholtz", correction_grid=8)
    p = Problem3D(d, cfg)
    w = WaveWireLoopCoils(4, 1, 12, 1, k2_init=k2)
    with torch.no_grad():
        w.verts.normal_(0, 0.3); w.verts.add_(torch.tensor([1.6, 0, 0])); w.amp.fill_(1)
    w.coarse = (p.coords, p.coarse_coords, p.calib, p.coarse_shape)
    pinn = make_pinn(p, cfg, w, 1)
    assert pinn.wave_base and float(pinn.physics_loss(p.support_pts[:64])) < 1e-10      # zero correction: exact
    torch.nn.init.normal_(pinn.net[-1].weight, std=0.05)
    train = CoarseCorrection(pinn, p).sensitivity(p.coords, create_graph=False).detach()
    ev_ = volume_maps(pinn, p, shape=p.calib).reshape(4, -1)
    # the two paths broadcast the points differently (per-coil vs shared), so float32 rounding differs;
    # the closed-form kernels themselves are accurate to ~6e-6 against float64
    assert float((train - ev_).norm() / train.norm()) < 2e-5


@pytest.mark.parametrize("layout,shape", [("2d", (4, 24, 20)), ("3d", (4, 12, 12, 10)), ("dynamic", (4, 20, 20, 3))])
def test_api_shapes_and_coil_axis(layout, shape):
    from bspinn.api import coil_combine, estimate_maps
    from bspinn.volume import make_synthetic3d
    if layout == "3d":
        d, _, _ = make_synthetic3d(4, shape[1:], noise=0.0)
        k = d.kspace
    else:
        from bspinn.data import make_synthetic
        d, _, _ = make_synthetic(4, shape[1:3], noise=0.0)
        k = d.kspace[0]
        if layout == "dynamic":
            k = np.stack([k] * shape[3], -1)
    few = dict(iters_auto=2, iters_wires=3, iters_pinn=3, iters_dynamic=2, rank=1) if layout == "dynamic" else \
        dict(iters_auto=2, iters_wires=3, iters_pinn=3)
    kw = dict(calib=8, verbose=False, axes="xyz") if layout == "3d" else dict(calib=8, verbose=False)
    C, img, info = estimate_maps(np.moveaxis(k, 0, -1), layout, coil_axis=-1, return_image=True, **kw, **few)
    assert C.shape == k.shape and C.dtype == np.complex64 and np.isfinite(C).all()
    assert img.shape == k.shape[1:] and info["image_joint"].shape == k.shape[1:]
    assert np.allclose((np.abs(C) ** 2).sum(0), 1, atol=1e-4)                     # rss normalisation
    assert info["support"].shape == k.shape[1:] if layout != "dynamic" else info["support"].shape == k.shape[1:3]
    assert coil_combine(k, C, layout).shape == k.shape[1:]
    with pytest.raises(ValueError):
        estimate_maps(k[0], layout, verbose=False)


def test_api_image_conventions():
    """Returned image(s) pair with the returned maps: fully sampled SENSE image reproduces the data,
    the joint image reproduces the calibration data up to the (short) fit's residual, and noise
    pre-whitening returns maps for the original channels."""
    from bspinn.api import _cfft, estimate_maps, sense_image
    from bspinn.data import make_synthetic
    d, G, obj = make_synthetic(4, (32, 32), noise=0.002, noise_corr=0.3)
    k = d.kspace[0]
    C, img, info = estimate_maps(k, "2d", noise=d.noise, calib=12, return_image=True, verbose=False,
                                 iters_auto=20, iters_wires=60, iters_pinn=60)
    x = _cfft(k, (1, 2), True)
    assert img.shape == (32, 32) and np.isfinite(img).all()
    assert np.abs(sense_image(k, C, "2d") - (C.conj() * x).sum(0) / (np.abs(C) ** 2).sum(0)).max() < 1e-3 * np.abs(img).max()
    acs = np.zeros((32, 32), bool); acs[10:22, 10:22] = True
    n = d.noise.reshape(-1, 4)
    L = np.linalg.inv(np.linalg.cholesky(n.conj().T @ n / n.shape[0]))
    pred = np.einsum("ij,j...->i...", L, _cfft(C * info["image_joint"][None], (1, 2)))
    kw = np.einsum("ij,j...->i...", L, k)
    r = np.linalg.norm((pred - kw)[:, acs]) / np.linalg.norm(kw[:, acs])
    assert abs(r / np.sqrt(info["data_loss"]) - 1) < 0.1, (r, info["data_loss"])     # same residual as the fit
    with pytest.raises(ValueError):
        estimate_maps(k, "2d", compress=2, verbose=False)


def test_api_mask_maps():
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    d, _, _ = make_synthetic(4, (32, 32), noise=0.01)
    k = d.kspace[0]
    few = dict(calib=12, verbose=False, iters_auto=2, iters_wires=3, iters_pinn=3)
    C, img, info = estimate_maps(k, "2d", return_image=True, mask_maps=True, **few)
    sup = info["support"]
    assert info["maps_masked"] and 0 < sup.mean() < 1
    assert np.all(C[:, ~sup] == 0) and np.all(img[~sup] == 0) and np.all(info["image_joint"][~sup] == 0)
    assert np.allclose((np.abs(C[:, sup]) ** 2).sum(0), 1, atol=1e-4)
    own = np.zeros((32, 32), bool); own[8:24, 8:24] = True
    C2, info2 = estimate_maps(k, "2d", mask_maps=own, **few)
    assert np.array_equal(info2["support"], own) and np.all(C2[:, ~own] == 0)
    loose = estimate_maps(k, "2d", mask_maps=True, support_threshold=0.02, support_dilate=4, **few)[1]["support"]
    assert loose.sum() >= sup.sum()
    kd = np.stack([k] * 3, -1)
    Cd, infod = estimate_maps(kd, "dynamic", mask_maps=True, rank=0, **few)
    assert Cd.shape == kd.shape and np.all(Cd[:, ~infod["support"]] == 0)


def test_keepout_distance_and_wire_core():
    from bspinn.geometry import SliceGeometry
    from bspinn.keepout import KeepOut
    from bspinn.physics import biot_savart_polyline
    g = SliceGeometry((40, 40), (1, 1), np.zeros(1), "axial")
    co = g.grid()
    sup = (co[..., :2].norm(dim=-1) < 0.5).float()
    k = KeepOut(sup, co, False, slab=0.15)
    d = k.distance(torch.tensor([[0.8, 0, 0], [0.2, 0, 0], [0, 0, 0.5], [0.0, 0.0, 0.1]]))
    assert abs(d[0] - 0.3) < 0.03 and d[1] < 0 and abs(d[2] - 0.35) < 0.03 and d[3] < 0     # prism +/- slab
    # Rankine core: exact beyond the core radius, bounded inside it
    V = torch.tensor([[[0.0, 0, -1], [0.0, 0, 1], [1.0, 0, 1], [1.0, 0, -1]]])
    far = torch.tensor([[0.3, 0.2, 0.0], [0.5, 0.5, 0.1]])
    assert torch.allclose(biot_savart_polyline(V, far), biot_savart_polyline(V, far, core=0.05))
    near = torch.tensor([[0.001, 0.0, 0.0]])
    assert biot_savart_polyline(V, near, core=0.05).norm() < 0.05 * biot_savart_polyline(V, near).norm()


def test_wires_stay_outside_object_in_every_stage():
    """Regression: the PINN stage trained the wires without the keep-out penalty, so conductors could
    drift into the object (singular, outlier-valued maps)."""
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, fit, wire_modules
    from bspinn.keepout import segment_samples
    d, _, _ = make_synthetic(4, (32, 32), noise=0.01)
    cfg = FitConfig(calib=12, iters_auto=5, iters_wires=40, iters_pinn=40, lr_pinn=3e-2, lr_wires=5e-2,
                    log_every=1000)
    model, prob, hist, extra = fit(d, cfg, log=lambda *a: None)
    for w in wire_modules(model):
        dist = prob.keep.distance(segment_samples(w.verts.detach(), 8))
        assert float(dist.min()) > 0.5 * cfg.outside_margin
        assert w.core == cfg.wire_radius


def test_support_cleaning_and_fov_truncation():
    from bspinn.geometry import SliceGeometry
    from bspinn.keepout import KeepOut, clean_support
    m = np.zeros((1, 40, 40), bool); m[0, 25:, 10:30] = True; m[0, :2, 15:25] = True; m[0, 30:33, 18:20] = False
    c = clean_support(m, True)[0]
    assert not c[:2].any() and c[30:33, 18:20].all()           # wrap-around blob dropped, hole filled
    g = SliceGeometry((40, 40), (1, 1), np.zeros(1), "axial"); co = g.grid()
    k = KeepOut(torch.from_numpy(c[None]).float(), co, False)
    below = torch.tensor([[0.0, 1.25, 0.0]])                    # beyond the FOV edge the object touches
    assert 0.2 < float(k.distance(below)) < 0.35
    k2 = KeepOut(torch.from_numpy(c[None]).float(), co, False, extend=0.1)
    assert float(k2.distance(below)) < float(k.distance(below)) - 0.05


def test_plane_auto_finds_inplane_b0():
    from bspinn.api import _select_plane
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig
    d, _, _ = make_synthetic(8, (48, 48), plane="coronal", noise=0.005)
    best, sc, _ = _select_plane(d.kspace, None, (1.0, 1.0), None, FitConfig(calib=16, iters_auto=60), lambda *a: None)
    assert best in ("coronal", "coronal_t") and sc["axial"] > min(sc.values()), sc


def test_sense_image_does_not_extrapolate_beyond_acquired_kspace():
    """Regression: zero-padded / partial-Fourier k-space edges were treated as undersampling, and
    CG-SENSE filled them with amplified noise (high-frequency stripes)."""
    from bspinn.api import _cfft, acquired_extent, coil_combine, sense_image
    from bspinn.data import make_synthetic
    d, G, obj = make_synthetic(6, (48, 48), noise=0.01)
    k = d.kspace[0].copy(); k[:, :, :4] = 0; k[:, :, -3:] = 0          # zero-padded PE edges
    C = G.reshape(k.shape)
    assert np.allclose(sense_image(k, C), coil_combine(k, C), atol=1e-5)  # fully sampled inside the extent
    ku = k.copy(); ku[:, :, 5:43:2] = 0; ku[:, :, 20:28] = k[:, :, 20:28]  # R=2 inside the extent (+ ACS)
    x = sense_image(ku, C)
    box = acquired_extent(np.abs(ku).sum(0) > 0)
    K = np.abs(_cfft(x, (0, 1))) ** 2
    assert K[~box].sum() < 1e-10 * K.sum()
    Ko = np.abs(_cfft(sense_image(ku, C, extrapolate=True), (0, 1))) ** 2
    assert Ko[~box].sum() > 1e-4 * Ko.sum()                                  # the old behaviour did fill them


@pytest.mark.parametrize("pattern", ["uniform", "random", "vd"])
def test_cartesian_masks_2d_and_dynamic(pattern):
    from bspinn.masks import cartesian_mask, effective_acceleration
    m = cartesian_mask((64, 80), R=4, acs=12, pattern=pattern, seed=3)
    assert m.shape == (64, 80) and m.dtype == bool
    assert m[:, 34:46].all()                                      # centred ACS block (80//2 - 6 .. +6)
    assert (m == m[:1]).all()                                     # readout (axis 0) fully sampled: lines
    if pattern != "uniform":
        assert abs(effective_acceleration(m) - 4) < 0.3          # net acceleration incl. ACS
    pf = cartesian_mask((64, 80), R=4, acs=12, pattern=pattern, partial_fourier=0.75)
    assert not pf[:, 60:].any()
    ext = np.zeros((64, 80), bool); ext[:, 5:75] = True           # zero-padded edges: never sampled
    assert not cartesian_mask((64, 80), 4, 12, pattern, extent=ext)[:, :5].any()
    d = cartesian_mask((32, 40, 6), R=4, acs=8, pattern=pattern, layout="dynamic")
    assert d.shape == (32, 40, 6) and d[:, 16:24].all()
    assert not all((d[..., t] == d[..., 0]).all() for t in range(1, 6))   # varies over time


@pytest.mark.parametrize("pattern", ["uniform", "caipi", "vd", "poisson"])
def test_cartesian_masks_3d(pattern):
    from bspinn.masks import cartesian_mask, effective_acceleration
    R = (2, 2) if pattern in ("uniform", "caipi") else 6
    m = cartesian_mask((8, 40, 48), R=R, acs=(8, 10), pattern=pattern, seed=1)
    assert (m == m[:1]).all()                                     # readout axis 0 fully sampled
    assert m[0, 16:24, 19:29].all()                               # ACS block in the PE plane
    if pattern in ("vd", "poisson"):
        assert abs(effective_acceleration(m) - 6) < 0.4


def test_estimate_maps_undersample_uses_calibration_block():
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    d, _, _ = make_synthetic(4, (32, 40), noise=0.01)
    k = d.kspace[0]
    C, img, info = estimate_maps(k, "2d", plane="axial", calib=10, return_image=True, verbose=False,
                                 undersample=dict(R=3, pattern="uniform"), iters_auto=2, iters_wires=3, iters_pinn=3)
    m = info["mask"]
    assert m[:, 15:25].all() and info["undersample"]["acs"] == 10 and 1.8 < info["R_effective"] < 2.6
    assert img.shape == (32, 40) and np.isfinite(img).all()


def test_estimate_maps_with_own_mask_and_calibration_warning():
    import warnings
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    from bspinn.masks import cartesian_mask
    d, _, _ = make_synthetic(4, (32, 40), noise=0.01)
    k = d.kspace[0]
    few = dict(plane="axial", iters_auto=2, iters_wires=3, iters_pinn=3, verbose=False)
    m = cartesian_mask(k.shape[1:], R=3, acs=10, pattern="random", seed=4)
    _, info = estimate_maps(k, "2d", undersample=m, calib=10, **few)
    assert np.array_equal(info["mask"], m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimate_maps(k, "2d", undersample=cartesian_mask(k.shape[1:], R=3, acs=4), calib=12, **few)
    assert any("calibration block" in str(x.message) for x in w)


def test_solve_image_joint_full_resolution():
    from bspinn.api import _cfft, estimate_maps
    from bspinn.data import make_synthetic
    from bspinn.fit import upsample_maps
    d, _, _ = make_synthetic(4, (32, 40), noise=0.005)
    k = d.kspace[0].copy(); k[:, :, :3] = 0; k[:, :, -3:] = 0          # zero-padded PE edges
    few = dict(plane="axial", calib=12, iters_auto=5, iters_wires=20, iters_pinn=20, iters_joint=10, verbose=False)
    C, x, info = estimate_maps(k, "2d", solve_image=True, sense_lambda=1e-2, undersample=dict(R=2, pattern="uniform"), **few)
    assert info["image_solved_jointly"] and x.shape == (32, 40) and np.isfinite(x).all()
    m = info["mask"]; ku = k * m[None]
    r = np.linalg.norm((_cfft(C * x[None], (1, 2)) - ku)[:, m]) / np.linalg.norm(ku[:, m])
    assert r < 0.5                                   # the joint image explains the acquired samples
    K = np.abs(_cfft(x, (0, 1))) ** 2
    # nothing extrapolated outside the acquired extent (only the smooth gauge of the returned maps spills a little;
    # extrapolating CG-SENSE put 20-25 % there on real data)
    assert K[:, :3].sum() + K[:, -3:].sum() < 0.02 * K.sum()
    with pytest.raises(ValueError):
        estimate_maps(np.stack([k] * 2, -1), "dynamic", solve_image=True, sense_lambda=1e-2, rank=1, **few)
    # band-limited map interpolation keeps point values on the coarse grid
    import torch
    g = torch.randn(1, 2, 8, 10, dtype=torch.complex64)
    up = upsample_maps(g, (16, 20))
    assert torch.allclose(up[..., ::2, ::2], g, atol=1e-5)


def test_taylor_mode_matches_autograd():
    """trunk_taylor (value, gradient, Laplacian in one forward pass) == nested autograd; and the
    Laplace / Helmholtz physics losses built on it equal the autograd reference implementations."""
    from bspinn.models import HarmonicPINN, HelmholtzPINN, WireLoopCoils, make_trunk, trunk_taylor
    from bspinn.physics import grad
    torch.manual_seed(0)
    net = make_trunk(3 + 5, 32, 3, 2).double()
    x = torch.randn(4, 50, 3, dtype=torch.float64, requires_grad=True)
    z = torch.randn(4, 1, 5, dtype=torch.float64)
    h, T, L = trunk_taylor(net, x, z)
    ref = net(torch.cat([x, z.expand(-1, 50, -1)], -1))
    assert torch.allclose(h, ref, atol=1e-12)
    for k in range(2):
        g = grad(ref[..., k], x)
        lap = sum(grad(g[..., e], x)[..., e] for e in range(3))
        assert torch.allclose(T[..., k], g, atol=1e-10) and torch.allclose(L[..., k], lap, atol=1e-9)
    # physics losses vs autograd reference
    w = WireLoopCoils(3, 1, 8, 1)
    with torch.no_grad():
        w.verts.normal_(0, 0.3); w.verts.add_(torch.tensor([1.5, 0, 0])); w.amp.fill_(1)
    p = HarmonicPINN(3, "complex", 6, 32, 3, 1, base=w)
    torch.nn.init.normal_(p.net[-1].weight, std=0.1)
    pts = torch.randn(40, 3) * 0.4
    xx = pts.unsqueeze(0).repeat(3, 1, 1).requires_grad_(True)
    phi = p.potential(xx); Bw = p._base_field(xx); num = 0; den = 0
    for k in range(2):
        g = grad(phi[..., k], xx); lap = sum(grad(g[..., d], xx)[..., d] for d in range(3))
        num = num + lap.pow(2).mean(1); Bt = Bw[..., k] - g
        den = den + (Bt[..., 0].pow(2) + Bt[..., 1].pow(2)).mean(1)
    assert torch.allclose(p.physics_loss(pts), (num / den).mean(), rtol=1e-4)
    hp = HelmholtzPINN(3, 6, 32, 3, 1, base=w, k2_init=0.5 - 0.2j)
    torch.nn.init.normal_(hp.net[-1].weight, std=0.1)
    xx = pts.unsqueeze(0).repeat(3, 1, 1).requires_grad_(True)
    Bc = hp._net(xx); Bt = Bc + hp._base_field(xx); a, b = hp.k2_re, hp.k2_im; res = 0; div = [0, 0]
    for d in range(3):
        lp = []
        for q in range(2):
            g = grad(Bc[..., d, q], xx); div[q] = div[q] + g[..., d]; lp.append(sum(grad(g[..., e], xx)[..., e] for e in range(3)))
        fr, fi = Bt[..., d, 0], Bt[..., d, 1]
        res = res + (lp[0] + a * fr - b * fi).pow(2) + (lp[1] + a * fi + b * fr).pow(2)
    den = Bt.pow(2).sum((-1, -2)).mean(1)
    want = (res.mean(1) / (den * max(abs(hp.k2) ** 2, 1.0))).mean() + hp.lambda_div * ((div[0] ** 2 + div[1] ** 2).mean(1) / den).mean()
    assert torch.allclose(hp.physics_loss(pts), want, rtol=1e-4)


def test_compiled_step_matches_eager():
    """cfg.compile: the whole iteration (forward + backward) compiled once per stage gives the same
    optimisation trajectory as eager execution (wires and PINN stage, multi-slice batches included)."""
    import copy
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem, fit_pinn, fit_wires, make_pinn
    d, _, _ = make_synthetic(4, (24, 24), n_slices=3, noise=0.01)
    d.prewhiten()
    out = {}
    for comp in (False, True):
        cfg = FitConfig(calib=12, log_every=1, compile=comp, compile_min_iters=0, slice_batch=2, n_colloc=64)
        prob = Problem(copy.deepcopy(d), cfg)
        try:
            w, m, hw, _ = fit_wires(prob, cfg, 1, 4, log=lambda *a: None)
            p = make_pinn(prob, cfg, w, 1)
            torch.nn.init.normal_(p.net[-1].weight, std=0.05)
            m, hp = fit_pinn(prob, cfg, p, m, log=lambda *a: None, iters=4)
        except Exception as e:                               # no C++ toolchain / unsupported platform
            if comp:
                pytest.skip(f"torch.compile unavailable: {e}")
            raise
        out[comp] = ([h["data"] for h in hw], [h["data"] for h in hp], p.base.verts.detach().clone())
    for a, b in zip(out[False][:2], out[True][:2]):
        assert np.allclose(a, b, rtol=1e-3), (a, b)
    assert torch.allclose(out[False][2], out[True][2], atol=1e-4)


def test_physics_loss_pair_matches_two_calls():
    """physics_loss_pair (object + air collocation in one network pass) == two physics_loss calls."""
    from bspinn.models import HarmonicPINN, HelmholtzPINN, WaveWireLoopCoils, WireLoopCoils
    torch.manual_seed(0)
    w = WireLoopCoils(3, 1, 8, 1)
    with torch.no_grad():
        w.verts.normal_(0, 0.3); w.verts.add_(torch.tensor([1.5, 0, 0])); w.amp.fill_(1)
    ww = WaveWireLoopCoils(3, 1, 8, 1, k2_init=2 - 1j)
    with torch.no_grad():
        ww.verts.copy_(w.verts); ww.amp.fill_(1)
    for m in (HarmonicPINN(3, "real", 6, 32, 3, 1, base=w), HarmonicPINN(3, "complex", 6, 32, 3, 1, base=w),
              HelmholtzPINN(3, 6, 32, 3, 1, base=w, k2_init=.5 - .2j), HelmholtzPINN(3, 6, 32, 3, 1, base=ww)):
        torch.nn.init.normal_(m.net[-1].weight, std=0.1)
        a, b = torch.randn(40, 3) * .4, torch.randn(13, 3) * .8
        ps = [q for q in m.parameters() if q.requires_grad]
        l1 = m.physics_loss(a) + 0.25 * m.physics_loss(b)
        l2 = m.physics_loss_pair(a, b, 0.25)
        g1 = torch.autograd.grad(l1, ps, allow_unused=True); g2 = torch.autograd.grad(l2, ps, allow_unused=True)
        assert torch.allclose(l1, l2, rtol=1e-5)
        assert all((x is None and y is None) or torch.allclose(x, y, rtol=1e-4, atol=1e-6) for x, y in zip(g1, g2))


def test_cuda_graph_stage_plumbing_matches_eager():
    """cfg.cuda_graph: capturable Adam with tensor learning rates, closed-form cosine schedule, slice batches
    through static index tensors and random draws inside the step give the eager trajectory. Runs the
    graph code path without capture on CPU (the capture itself needs CUDA: scripts/bench_gpu.py checks it)."""
    import copy
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem, fit_pinn, fit_wires, make_pinn
    d, _, _ = make_synthetic(4, (24, 24), n_slices=3, noise=0.01)
    d.prewhiten()
    out = {}
    for graph in (False, True):
        cfg = FitConfig(calib=12, log_every=1, cuda_graph=graph, slice_batch=2, n_colloc=64)
        cfg._graph_emulate = True
        prob = Problem(copy.deepcopy(d), cfg)
        w, m, hw, _ = fit_wires(prob, cfg, 1, 8, log=lambda *a: None)
        p = make_pinn(prob, cfg, w, 1)
        torch.nn.init.normal_(p.net[-1].weight, std=0.05)
        m, hp = fit_pinn(prob, cfg, p, m, log=lambda *a: None, iters=8)
        out[graph] = ([h["data"] for h in hw], [h["data"] for h in hp], p.base.verts.detach().clone())
    for a, b in zip(out[False][:2], out[True][:2]):
        assert np.allclose(a, b, rtol=1e-4), (a, b)
    assert torch.allclose(out[False][2], out[True][2], atol=1e-5)


def test_fused_trunk_paths_match():
    """GPU code paths, checked on CPU: the stacked Taylor-mode trunk (one matmul per layer) equals
    trunk_taylor, and sensitivity_and_physics (data grid + collocation in one pass) equals the
    separate sensitivity / physics_loss_pair calls, values and parameter gradients."""
    from bspinn.models import HarmonicPINN, WireLoopCoils, make_trunk, trunk_taylor, trunk_taylor_stacked
    torch.manual_seed(0)
    net = make_trunk(3 + 5, 32, 3, 2).double()
    x = torch.randn(4, 50, 3, dtype=torch.float64)
    z = torch.randn(4, 1, 5, dtype=torch.float64)
    for lap in (True, False):
        a, b = trunk_taylor(net, x, z, lap), trunk_taylor_stacked(net, x, z, lap)
        assert all((u is None and v is None) or torch.allclose(u, v, atol=1e-12) for u, v in zip(a, b))
    w = WireLoopCoils(3, 1, 8, 1)
    with torch.no_grad():
        w.verts.normal_(0, 0.3); w.verts.add_(torch.tensor([1.5, 0, 0])); w.amp.fill_(1)
    for mode in ("real", "complex"):
        p = HarmonicPINN(3, mode, 6, 32, 3, 1, base=w)
        torch.nn.init.normal_(p.net[-1].weight, std=0.1)
        pts, c1, c2 = torch.rand(30, 3) - .5, torch.randn(20, 3) * .4, torch.randn(7, 3) * .8
        ps = [q for q in p.parameters() if q.requires_grad]
        C1, ph1 = p.sensitivity(pts), p.physics_loss_pair(c1, c2, .25)
        C2, ph2 = p.sensitivity_and_physics(pts, c1, c2, .25)
        assert torch.allclose(C1, C2, atol=1e-6) and torch.allclose(ph1, ph2, rtol=1e-5)
        g1 = torch.autograd.grad((C1.abs() ** 2).sum() + ph1, ps, allow_unused=True)
        g2 = torch.autograd.grad((C2.abs() ** 2).sum() + ph2, ps, allow_unused=True)
        assert all((u is None and v is None) or torch.allclose(u, v, rtol=1e-4, atol=1e-5) for u, v in zip(g1, g2))


def test_fold_over_detected_and_corrected():
    """An object wider than the FOV folds over: tissue beyond one edge appears at the other, seen by the
    far coils. fold='auto' must detect it along the phase-encode axis and give lower SENSE error than
    the plain (non-periodic) physical maps."""
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    from bspinn.masks import cartesian_mask
    ax = (1, 2)
    F = lambda x: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
    Fi = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
    d, _, _ = make_synthetic(8, (64, 96), noise=0.003, seed=3)
    I = Fi(d.kspace[0])
    Iw = I[:, :, 16:80].copy(); Iw[:, :, :16] += I[:, :, 80:]; Iw[:, :, -16:] += I[:, :, :16]   # FOV 64: wrap
    k = F(Iw).astype(np.complex64)
    m = cartesian_mask((64, 64), R=3, acs=16, pattern="vd", seed=1)
    ref = np.sqrt((np.abs(Iw) ** 2).sum(0))
    err = {}
    for fold in ("auto", False):
        C, img, info = estimate_maps(k * m[None], "2d", calib=16, mask=m, verbose=False, return_image=True,
                                     fold=fold, sense_lambda=1e-2, iters_auto=40, iters_wires=150, iters_pinn=150)
        a = np.abs(img); s = (a * ref).sum() / (a ** 2).sum()
        err[fold] = np.linalg.norm(s * a - ref) / np.linalg.norm(ref)
        if fold == "auto":
            assert info["fold"]["corrected"] and all(r.startswith("axis1") for r in info["fold"]["regions"])
    assert err["auto"] < 0.8 * err[False], err


def _rf_case(joint=False):
    from bspinn.data import make_synthetic
    from bspinn.methods import EvalCase
    d, C, obj = make_synthetic(6, (64, 64), noise=0.01, seed=0)
    return EvalCase("t", d.kspace[0], np.ones((64, 64), bool), (16, 16), sigma=0.01, gt_maps=C[0],
                    gt_object=obj[0], joint=joint), C[0]


def test_reference_free_metrics_separate_true_perturbed_and_noise_fitted_maps():
    from bspinn import evaluation as ev
    from bspinn import refree
    from bspinn.methods import ESTIMATORS, ifft2c
    case, C = _rf_case()
    obj = ev.object_mask(case)
    yy, xx = np.mgrid[:64, :64] / 64
    bad = C * (1 + 0.3 * np.cos(2 * np.pi * (xx[None] + yy[None] + np.arange(6)[:, None, None] / 6)))   # per coil
    good = refree.evaluate(C, case, obj, name="oracle", resample=("r2r", "sure"))["summary"]
    worse = refree.evaluate(bad, case, obj)["summary"]
    assert 0.9 < good["lco_outside_acs_over_floor"] < 1.1 and 0.9 < good["lco_object_over_floor"] < 1.1
    assert abs(good["residual_signal_slope"]) < 0.05 and 0.8 < good["residual_whiteness"] < 1.2
    assert good["sure_map_dof"] == 0.0 and 0.85 < good["sure_over_oracle"] < 1.15
    assert worse["lco_object_over_floor"] > 2 and worse["residual_whiteness"] > 2 and worse["residual_signal_slope"] > 0.2
    # an estimator that copies the data: zero residual on its own data, caught by r2r / SURE / bandwidth
    ESTIMATORS["_copy"] = lambda c, **k: ev.normalise(ifft2c(c.ksp * c.sampled[None]))
    try:
        cj, _ = _rf_case(joint=True)
        cp = refree.evaluate(ESTIMATORS["_copy"](cj), cj, obj, name="_copy", resample=("r2r", "sure"), sure_probes=2)["summary"]
    finally:
        del ESTIMATORS["_copy"]
    assert cp["lco_object_over_floor"] < 1e-6                     # fooled: it saw the noise...
    assert cp["r2r_optimism"] > 20 * abs(good["r2r_optimism"])   # ...but not on an independent copy
    assert cp["sure_over_oracle"] > 4                             # exact: N = 6
    assert cp["projector_hf_fraction"] > 100 * good["projector_hf_fraction"]


def test_reference_free_prospective_and_benchmark_plumbing(tmp_path):
    import importlib.util
    import os
    from bspinn import evaluation as ev
    from bspinn import refree
    case, C = _rf_case()
    m = np.zeros((64, 64), bool); m[:, ::2] = True; m[:, 24:40] = True
    case.ksp = case.ksp * m[None]; case.sampled = m; case.fully_sampled = False
    out = refree.evaluate(C, case, ev.object_mask(case), lco_coils=[0, 1])
    assert 0.85 < out["summary"]["lco_outside_calib_over_floor"] < 1.15
    assert out["summary"]["lco_outside_calib_over_sample_noise"] > 1
    spec = importlib.util.spec_from_file_location("evm", os.path.join(os.path.dirname(__file__), "..", "scripts", "evaluate_methods.py"))
    evm = importlib.util.module_from_spec(spec); spec.loader.exec_module(evm)
    case2, _ = _rf_case()
    res = evm.run_case(case2, ["lowres", "oracle"], str(tmp_path), Rs=(2,), log=lambda *a: None, resample=("r2r",))
    assert set(res["methods"]) == {"lowres", "oracle"}
    tsv = open(tmp_path / "t" / "reference_free.tsv").read()
    assert "r2r_optimism" in tsv and "lco_outside_acs_over_floor" in tsv


def test_evaluate_methods_dataset_items(tmp_path, monkeypatch):
    import importlib.util
    import os
    import sys
    from bspinn.data import make_synthetic
    mod = tmp_path / "fake_ds_mod.py"
    mod.write_text(
        "import numpy as np\n"
        "from bspinn.data import make_synthetic\n"
        "class DS:\n"
        "    def __len__(self): return 3\n"
        "    def __getitem__(self, i):\n"
        "        d, C, o = make_synthetic(6, (80, 72), noise=0.0, seed=i)\n"
        "        k = d.kspace[0]; r = np.random.default_rng(i); A = np.eye(6) + 0.3 * r.normal(size=(6, 6))\n"
        "        n = (r.normal(size=k.shape) + 1j * r.normal(size=k.shape)) * 0.01\n"
        "        k = k + np.einsum('ij,jxy->ixy', A, n); k[:, :, :6] = 0\n"
        "        return {'kspace': k, 'fname': f'fake{i}'}\n"
        "def get(base=None): return DS()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location("evm2", os.path.join(os.path.dirname(__file__), "..", "scripts", "evaluate_methods.py"))
    evm = importlib.util.module_from_spec(spec); spec.loader.exec_module(evm)
    # noise estimate (correlated channels, signal up to the k-space edge) and whitening
    import fake_ds_mod
    k = fake_ds_mod.DS()[1]["kspace"]
    r = np.random.default_rng(1); A = np.eye(6) + 0.3 * r.normal(size=(6, 6))
    psi, info = evm.estimate_noise(k, np.abs(k).sum(0) > 0)
    assert abs(np.trace(psi).real / np.trace(2e-4 * A @ A.T) - 1) < 0.15
    cases = list(evm.loader_cases("fake_ds_mod:get", None, [0, 2], 0, 0, "kspace", 0, 24, None, None))
    c = cases[0]()
    assert c.pe_axis == 1 and c.sigma == 1.0 and not c.sampled[:, :6].any()
    monkeypatch.setattr(sys, "argv", ["x", "--loader", "fake_ds_mod:get", "--indices", "0", "2", "--methods", "lowres", "walsh",
                                      "--Rs", "4", "--out", str(tmp_path / "ev")])
    evm.main()
    rows = open(tmp_path / "ev" / "summary.tsv").read().splitlines()
    assert len(rows) == 5 and "ssim_leave_coil_out" in rows[0] and "ce_acs_over_floor" in rows[0]
    assert "real\twalsh\t2" in open(tmp_path / "ev" / "ranks.tsv").read()


def _fold_problem(X=64, Y=64, N=6, band=12, seed=0):
    """Smooth maps T on an extended grid; tissue beyond the left edge folds into the right `band` columns."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, X), np.linspace(-1, 1 + 2 * Y / Y, 2 * Y), indexing="ij")
    T = np.zeros((N, X, 2 * Y), complex)
    for j in range(N):
        cx, cy = 2.5 * np.cos(2 * np.pi * j / N) + 1, 2.5 * np.sin(2 * np.pi * j / N)
        T[j] = 1 / ((xx - cx) ** 2 + (yy - cy) ** 2 + 0.5) * np.exp(1j * (0.7 * xx * np.cos(j) + 0.5 * yy * np.sin(j)))
    C, Cs = T[:, :, :Y], T[:, :, Y:]                       # Cs = maps one FOV to the right (+ direction)
    img = np.ones((X, Y))
    fold = np.zeros((X, Y)); fold[:, :band] = 1            # tissue one FOV away appears in the first columns
    I = C * img + Cs * fold
    ax = (1, 2)
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(I, axes=ax), axes=ax, norm="ortho"), axes=ax)
    return C.astype(np.complex64), Cs.astype(np.complex64), k.astype(np.complex64)


def test_fold_correct_readout_axis_and_near_field_guard():
    from bspinn.api import _fold_correct, _proj_rough
    C, Cs, k = _fold_problem()
    log = lambda *a: None
    out, rep = _fold_correct(C, {(1, 1): Cs}, k, None, 24, log)
    assert rep["corrected"] and "axis1+FOV" in rep["regions"]
    _, rep_ro = _fold_correct(C, {(1, 1): Cs}, k, None, 24, log, readout_axis=1)
    assert not rep_ro["corrected"] and rep_ro["readout_axis_excluded"] == [1]
    # a conductor next to the shifted points: a near-field blob in the shifted maps inside the fold band
    X, Y = C.shape[1:]
    yy, xx = np.mgrid[:X, :Y]
    r2 = (yy - 32.0) ** 2 + (xx - 5.0) ** 2
    blob = (np.exp(1j * 0.9 * (yy + 2 * xx)) * np.exp(-r2 / 18))[None] * np.abs(Cs).max() * np.linspace(1, -1, C.shape[0])[:, None, None]
    Csb = (Cs + blob).astype(np.complex64)
    assert _proj_rough(Csb)[32, 5] > 3 * np.percentile(_proj_rough(C), 99)
    out_b, rep_b = _fold_correct(C, {(1, 1): Csb}, k, None, 24, log)
    assert rep_b["corrected"]                                           # the rest of the band is still corrected
    assert np.allclose(out_b[:, 32, 5], C[:, 32, 5])                     # ...but not next to the conductor
    assert not np.allclose(out_b[:, 10, 5], C[:, 10, 5])


def test_heldout_score_prefers_true_maps():
    from bspinn.api import _heldout_score, _n_heldout
    from bspinn.data import make_synthetic
    from bspinn.masks import cartesian_mask
    d, C, _ = make_synthetic(6, (64, 64), noise=0.003, seed=2)
    k = d.kspace[0]
    m = cartesian_mask((64, 64), R=3, acs=16, pattern="vd", seed=1)
    yy, xx = np.mgrid[:64, :64] / 64
    bad = C[0] * (1 + 0.3 * np.cos(2 * np.pi * (xx[None] + np.arange(6)[:, None, None] / 6)))
    assert _n_heldout(k * m, m, 16) > 0 and _n_heldout(k * m, np.pad(np.ones((16, 16), bool), 24), 16) == 0
    good_s, bad_s = _heldout_score(C[0], k * m, m, 16, "cpu"), _heldout_score(bad, k * m, m, 16, "cpu")
    assert good_s < 0.8 * bad_s, (good_s, bad_s)


def test_mixing_and_smooth_factor_modules():
    from bspinn.models import ChannelMixer, SmoothFactor
    sf = SmoothFactor(4, (2, 0), order=2)
    pts = torch.rand(50, 3) * 2 - 1
    assert torch.allclose(sf(pts), torch.ones(4, 50, dtype=torch.complex64))      # starts at exp(0) = 1
    assert float(sf.penalty()) == 0.0
    far = pts.clone(); far[:, 2] += 2.0
    edge = pts.clone(); edge[:, 2] = 1.0
    with torch.no_grad():
        sf.a.normal_()
    assert torch.allclose(sf(far), sf(edge))                                     # constant beyond the FOV
    mx = ChannelMixer(4)
    C = torch.randn(4, 10, dtype=torch.complex64)
    assert torch.allclose(mx(C), C)
    with torch.no_grad():
        mx.e.normal_()
    assert not torch.allclose(mx(C), C) and float(mx.penalty()) > 0
    assert torch.allclose(torch.diagonal(mx.off), torch.zeros(4))                # no diagonal (= gains)


def test_learned_mixing_recovers_coupled_channels():
    """Channels that are complex mixtures of Biot–Savart coils (coupling): the learnable mixing lowers
    the map error substantially compared with the plain physics model."""
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    ax = (1, 2)
    F = lambda x: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
    Fi = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
    d, C, obj = make_synthetic(6, (48, 48), noise=0.001, seed=4)
    rng = np.random.default_rng(0)
    E = 0.1 * (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))); np.fill_diagonal(E, 0)
    M = np.eye(6) + E
    k = np.einsum("ij,jxy->ixy", M, d.kspace[0])
    Ct = np.einsum("ij,jxy->ixy", M, C[0])
    Gn = Ct / np.sqrt((np.abs(Ct) ** 2).sum(0, keepdims=True))
    err = {}
    for mix in (False, True):
        Cm, _ = estimate_maps(k, "2d", calib=16, plane="axial", verbose=False, iters_auto=40, iters_wires=200,
                              iters_pinn=300, learn_mixing=mix, resid_order=0, fold=False, threads=1)
        Cn = Cm / np.sqrt((np.abs(Cm) ** 2).sum(0, keepdims=True))
        err[mix] = float((1 - np.abs((Cn.conj() * Gn).sum(0)) ** 2)[obj[0] > 0].mean())
    assert err[True] < 0.5 * err[False], err


def test_joint_stage_keeps_factor_and_coupling():
    """solve_image=True refines the same model the API returns (smooth factor and channel coupling applied
    in the joint stage too): the returned maps and image explain the acquired data, and the maps stay
    accurate (v33 before the fix: the joint stage optimised maps without them, the API applied them)."""
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    from bspinn.masks import cartesian_mask
    ax = (1, 2)
    F = lambda x: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
    d, C, obj = make_synthetic(6, (48, 48), noise=0.002, seed=5)
    m = cartesian_mask((48, 48), R=3, acs=16, pattern="vd", seed=1)
    k = d.kspace[0] * m[None]
    Gn = C[0] / np.sqrt((np.abs(C[0]) ** 2).sum(0, keepdims=True))
    Cm, img, info = estimate_maps(k, "2d", calib=16, mask=m, plane="axial", verbose=False, iters_auto=30,
                                  iters_wires=150, iters_pinn=200, iters_joint=60, solve_image=True, fold=False,
                                  threads=1)
    Cn = Cm / np.sqrt((np.abs(Cm) ** 2).sum(0, keepdims=True))
    err = float((1 - np.abs((Cn.conj() * Gn).sum(0)) ** 2)[obj[0] > 0].mean())
    res = np.linalg.norm(m[None] * F(Cm * img[None]) - k) / np.linalg.norm(k)
    assert err < 1e-3 and res < 0.1, (err, res)              # v32 joint: res 0.081 on the same data


def test_lbfgs_optimizer_option():
    """optimizer='lbfgs' (every stage, joint included) and 'adam+lbfgs' run and give accurate maps."""
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    from bspinn.masks import cartesian_mask
    d, C, obj = make_synthetic(6, (48, 48), noise=0.002, seed=5)
    m = cartesian_mask((48, 48), R=3, acs=16, pattern="vd", seed=1)
    k = d.kspace[0] * m[None]
    Gn = C[0] / np.sqrt((np.abs(C[0]) ** 2).sum(0, keepdims=True))
    for opt, extra in (("lbfgs", {}), ("adam+lbfgs", {"iters_polish": 40}), ("lbfgs", {"solve_image": True, "iters_joint": 10})):
        out = estimate_maps(k, "2d", calib=16, mask=m, plane="axial", verbose=False, iters_auto=20, iters_wires=100,
                            iters_pinn=100, fold=False, threads=1, optimizer=opt, **extra)
        Cn = out[0] / np.sqrt((np.abs(out[0]) ** 2).sum(0, keepdims=True))
        err = float((1 - np.abs((Cn.conj() * Gn).sum(0)) ** 2)[obj[0] > 0].mean())
        assert np.isfinite(out[0]).all() and err < 2e-3, (opt, extra, err)
    from bspinn.fit import FitConfig
    import pytest
    with pytest.raises(ValueError):
        FitConfig(optimizer="sgd").validate()


def test_auto_sense_lambda_grows_with_noise():
    """The automatic SENSE weight (every rule) is larger for noisier data, and sense_image / estimate_maps
    accept 'auto' while a number still works as before."""
    from bspinn.api import sense_image
    from bspinn.data import make_synthetic
    from bspinn.masks import cartesian_mask
    from bspinn.senselam import choose_sense_lambda
    d, C, obj = make_synthetic(6, (64, 64), noise=0.0, seed=1)
    k0 = d.kspace[0]
    m = cartesian_mask((64, 64), R=3, acs=16, pattern="vd", seed=1)
    Cn = C[0] / np.sqrt((np.abs(C[0]) ** 2).sum(0, keepdims=True))
    rng = np.random.default_rng(0)
    lam = {}
    for s in (0.002, 0.05):
        k = (k0 + s * (rng.normal(size=k0.shape) + 1j * rng.normal(size=k0.shape)) / np.sqrt(2)) * m[None]
        _, info = choose_sense_lambda(k, Cn, m, method="gcv", iters=25, probes=2)
        lam[s] = info
        rep = {}
        x = sense_image(k, Cn, "2d", mask=m, lam="auto-map", iters=25, report=rep)
        assert np.isfinite(x).all() and rep["method"] == "map" and rep["value"] > 0
    assert lam[0.05]["lam_map"] > 2 * lam[0.002]["lam_map"]      # (sharp phantom edges leak into the noise estimate)
    assert lam[0.05]["lam_gcv"] > lam[0.002]["lam_gcv"]
    lcv, icv = choose_sense_lambda(k, Cn, m, method="cv", calib=16, iters=25)     # default rule of 'auto'
    assert 1e-5 <= lcv <= 1.0 and len(icv["cv_error"]) == 16 and lcv == icv["lam_cv"]
    la, ia = choose_sense_lambda(k, Cn, m, method="cv+map", calib=16, iters=25)    # = 'auto'
    assert la == max(ia["lam_cv"], ia["lam_map"])
    rep = {}
    sense_image(k, Cn, "2d", mask=m, lam=3e-2, iters=10, report=rep)
    assert rep == {"method": "manual", "value": 3e-2}


def test_true_joint_fit_options():
    """joint_start='wires' (maps and image jointly from the first PINN step) and joint_image='gradient'
    (simultaneous steps on maps and image) run, return the joint image, and choose the automatic image
    weight inside the joint stage."""
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    from bspinn.masks import cartesian_mask
    d, C, obj = make_synthetic(6, (64, 64), noise=0.003, seed=5)
    m = cartesian_mask((64, 64), R=3, acs=16, pattern="vd", seed=1)
    k = d.kspace[0] * m[None]
    Gn = C[0] / np.sqrt((np.abs(C[0]) ** 2).sum(0, keepdims=True))
    for ov in (dict(joint_start="wires"), dict(joint_image="gradient"), dict(joint_start="wires", joint_image="gradient")):
        Cm, img, info = estimate_maps(k, "2d", calib=16, mask=m, plane="axial", verbose=False, iters_auto=20,
                                      iters_wires=100, iters_pinn=100, iters_joint=20, solve_image=True, fold=False,
                                      threads=1, **ov)
        Cn = Cm / np.sqrt((np.abs(Cm) ** 2).sum(0, keepdims=True))
        err = float((1 - np.abs((Cn.conj() * Gn).sum(0)) ** 2)[obj[0] > 0].mean())
        assert info["image_solved_jointly"] and info["sense_lambda"]["method"] == "cv+map", ov
        assert np.isfinite(img).all() and err < 2e-3, (ov, err)
