#!/usr/bin/env python3
"""iMED PE submission — tri3d cross-camera pose + grid BA (default).

Task (survey/competition/pe_task_io.md): per frame, the same-time relative pose
endoscope2/L <- endoscope1/L (T_{e2<-e1}), frame 0 = identity. Scored by
Sim(3)-aligned mean ATE [mm], lowest wins. Local test (19 seq, corrected K):

  official baseline (ALIKED+LightGlue essential, t/||t||)   2.54 mm
  tri3d (stereo 3D-3D Umeyama, consistent scale)            1.034
  + IRLS robust rigid fit                                    0.998
  + grid BA (temporal egomotion, k=1, lamrel=0.3)  <-DEFAULT 0.953  (-62.5% vs baseline)

PIPELINE (default = tri3d + gridba=1):
  1. Fixed stereo extrinsic per scope (e1 L-R, e2 L-R), aggregated over frames.
  2. Per frame: triangulate e1 and e2 stereo clouds, match e1L<->e2L, Umeyama
     (IRLS) -> same-time T_{e2<-e1}(t).  Rigid model: outliers = mismatches /
     non-rigid / sync-jitter points, down-weighted by residual (velocity-based
     weighting was tried and HURT -- residual is the right signal).
  3. GRID BA: add temporal egomotion edges. The exact rigid relation
     T(t+1) = M_e2(t) T(t) M_e1(t)^{-1} is linear in the translations with the
     rotations fixed, so we fuse same-time (absolute) + egomotion (relative,
     weak weight -- deformation makes it approximate) + smoothness by one linear
     least squares.  Only translation matters for ATE.
  4. Re-anchor frame 0 to exact identity (contract).

KEY FIXES baked in:
  - load_K: iMED_pe intrinsics were shipped unscaled (1280x1024 calib on 600x480
    frames); auto-detect and rescale by W/1280 (wrong K cost ~40% ATE). Handles
    both raw and corrected /input, no double-scaling.
  - fp16 matcher is OFF: it degraded ATE 0.953 -> 1.05 on the full set.

SPEED: feature cache shared across the stereo aggregation + temporal edges, and
per-frame stereo clouds reused by the egomotion edges -- 18.5 -> 11 min / 19 seq
on RTX8000 (exact, ATE bit-identical). Eval host is RTX4090 (faster). Budget =
10 min total for all sequences.

Docker: /input(ro) -> /output, GPU, --network=none (weights baked in).
"""
import os, sys, glob, re, argparse
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

from lightglue import ALIKED, SuperPoint, DISK, DoGHardNet, LightGlue
from lightglue.utils import load_image, rbd

_EXTRACTORS = {"aliked": ALIKED, "superpoint": SuperPoint, "disk": DISK, "doghardnet": DoGHardNet}
_EXTRACTOR = "aliked"

CAM_REF, CAM_SRC = ("endoscope1", "L"), ("endoscope2", "L")   # 3D from e1L stereo, PnP into e2L
STEREO_REF = ("endoscope1", "R")                              # e1 stereo partner
MAX_KPTS = 2048
MIN_PNP = 12            # min 3D-2D correspondences for PnP
MIN_MATCHES = 8
_NAGG = 15
_ROBUST = "mad2"   # rigid-fit robustness: none|mad2(legacy)|irls
_RANSAC = 0        # RANSAC iters on the rigid model
_VELW = 0.0        # velocity down-weight strength (0 = off)
_VELMODE = "abs"   # abs | residual (deviation from global affine flow)
INPUT_DIR, OUTPUT_DIR = "/input", "/output"


# ----------------------------- matching -------------------------------------
class Matcher:
    def __init__(self, device="cuda", extractor=None, fp16=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ex = extractor or _EXTRACTOR
        self.ext = _EXTRACTORS[ex](max_num_keypoints=MAX_KPTS).eval().to(self.device)
        self.lg = LightGlue(features=ex).eval().to(self.device)
        # fp16 autocast on CUDA: the matcher is robust to half precision and it is
        # ~2x on Ada (the RTX4090 eval host) tensor cores, the main runtime lever.
        self.amp = fp16 and self.device.type == "cuda"

    @torch.no_grad()
    def feats(self, path):
        with torch.autocast("cuda", enabled=self.amp):
            return self.ext.extract(load_image(path).to(self.device))

    @torch.no_grad()
    def match(self, f0, f1):
        with torch.autocast("cuda", enabled=self.amp):
            m = self.lg({"image0": f0, "image1": f1})
        f0, f1, m = [rbd(x) for x in (f0, f1, m)]
        idx = m["matches"]                       # (M,2) indices into f0,f1 keypoints
        k0 = f0["keypoints"].cpu().numpy()
        k1 = f1["keypoints"].cpu().numpy()
        return idx.cpu().numpy(), k0, k1


# ----------------------------- geometry --------------------------------------
NATIVE_CALIB_W = 1280   # iMED_pe intrinsics were calibrated at 1280x1024


def parse_K(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    mats, i = {}, 0
    while i < len(lines):
        if lines[i].startswith("#") and lines[i][1:].strip().startswith("K"):
            key = lines[i][1:].strip().split()[0]
            mats[key] = np.array([[float(x) for x in lines[i + j].split()] for j in (1, 2, 3)])
            i += 4
        else:
            i += 1
    return mats


def _image_wh(seq):
    for cam, side in (("endoscope1", "L"), ("endoscope2", "L")):
        ps = glob.glob(os.path.join(seq, cam, side, "*.png"))
        if ps:
            h, w = cv2.imread(ps[0]).shape[:2]
            return w, h
    return None, None


def load_K(seq):
    """K.txt with the iMED_pe intrinsics-scale bug handled robustly.

    Early iMED_pe releases shipped 600x480 frames but left K.txt at the native
    1280x1024 calibration (principal point lands near / past the image edge, e.g.
    cx=595, cy=498 on a 600x480 frame). The organizers' fix is a uniform 15/32 =
    W/1280 rescale of fx,fy,cx,cy. Using the raw K on a 600x480 frame costs ~40%
    ATE (3-seq: 1.19 -> 0.68 mm), so we must scale it.

    We do NOT know which K version the hidden test /input ships, so we DETECT:
    if the principal point is implausibly off-centre for THIS image, the K is the
    unscaled native calibration and we rescale by W/1280. An already-corrected K
    (cx ~= W/2) is left untouched -- so the container is right either way, with
    no risk of double-scaling.
    """
    K = parse_K(os.path.join(seq, "K.txt"))
    w, h = _image_wh(seq)
    if not w:
        return K
    unscaled = any(M[0, 2] > 0.65 * w or M[1, 2] > 0.65 * h for M in K.values())
    if unscaled:
        s = w / float(NATIVE_CALIB_W)
        for M in K.values():
            M[0, :] *= s
            M[1, :] *= s
    return K


def frame_list(seq, cam, side):
    ps = sorted(glob.glob(os.path.join(seq, cam, side, "*.png")))
    return [(int(re.search(r"frame_(\d+)\.png$", os.path.basename(p)).group(1)), p) for p in ps]


class FlowVelocity:
    """Per-pixel image velocity in endoscope1/L, from dense Farneback flow between
    consecutive frames (CPU). Used to down-weight fast-moving points, whose
    cross-camera correspondences are the ones sync-jitter / motion blur corrupt
    (error ~ image velocity). Lazy + cached; disabled by default."""

    def __init__(self, e1L):
        self.paths = {f: p for f, p in e1L}
        self.order = [f for f, _ in e1L]
        self.pos = {f: i for i, f in enumerate(self.order)}
        self._cache = {}

    def _gray(self, fid):
        g = cv2.imread(self.paths[fid], cv2.IMREAD_GRAYSCALE)
        return g

    def mag(self, fid):
        """HxW velocity magnitude (px/frame) at fid, averaged over available neighbours."""
        if fid in self._cache:
            return self._cache[fid]
        i = self.pos[fid]
        g0 = self._gray(fid)
        mags = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(self.order):
                g1 = self._gray(self.order[j])
                fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
                mags.append(np.linalg.norm(fl, axis=2))
        m = np.mean(mags, 0) if mags else np.zeros_like(g0, np.float32)
        self._cache[fid] = m
        return m

    def flow(self, fid):
        """HxWx2 flow to the next available neighbour (for residual mode)."""
        if ("f", fid) in self._cache:
            return self._cache[("f", fid)]
        i = self.pos[fid]; g0 = self._gray(fid)
        fl = None
        for j in (i + 1, i - 1):
            if 0 <= j < len(self.order):
                fl = cv2.calcOpticalFlowFarneback(g0, self._gray(self.order[j]), None,
                                                  0.5, 3, 21, 3, 5, 1.2, 0)
                break
        if fl is None:
            fl = np.zeros(g0.shape + (2,), np.float32)
        self._cache[("f", fid)] = fl
        return fl

    def weights_at(self, fid, pts_xy, alpha=1.0, mode="abs"):
        """w = 1/(1 + alpha * v), where v is either absolute image velocity ('abs')
        or the flow's deviation from a global affine model ('residual' = locally
        anomalous motion, i.e. the part not explained by the camera's own motion)."""
        H, W = self.mag(fid).shape if mode == "abs" else self.flow(fid).shape[:2]
        x = np.clip(pts_xy[:, 0].round().astype(int), 0, W - 1)
        y = np.clip(pts_xy[:, 1].round().astype(int), 0, H - 1)
        if mode == "abs":
            v = self.mag(fid)[y, x]
        else:
            fl = self.flow(fid)
            ys, xs = np.mgrid[0:H:16, 0:W:16]                # subsample to fit global affine
            X = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1)
            Fu = fl[ys.ravel(), xs.ravel()]
            au, *_ = np.linalg.lstsq(X, Fu[:, 0], rcond=None)
            av, *_ = np.linalg.lstsq(X, Fu[:, 1], rcond=None)
            P = np.stack([x, y, np.ones_like(x)], 1).astype(float)
            pred = np.stack([P @ au, P @ av], 1)
            v = np.linalg.norm(fl[y, x] - pred, axis=1)      # residual from global motion
        return 1.0 / (1.0 + alpha * v)


def essential_pose(k0, k1, K0, K1):
    """Unit-translation relative pose T_{1<-0} from 2D-2D (cross intrinsics)."""
    n0 = cv2.undistortPoints(k0.reshape(-1, 1, 2), K0, None).reshape(-1, 2)
    n1 = cv2.undistortPoints(k1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(n0, n1, method=cv2.RANSAC, prob=0.999, threshold=1e-3)
    if E is None or E.shape != (3, 3):
        return None
    _, Rr, t, _ = cv2.recoverPose(E, n0, n1, np.eye(3), mask=mask)
    t = t.reshape(3); t = t / (np.linalg.norm(t) + 1e-12)
    return Rr, t


def estimate_stereo(matcher, ref_frames, stereo_frames, K_L, K_R, n_agg=15, feats=None):
    """Estimate the FIXED e1 L-R extrinsic [R|t] (unit t) once, aggregated over
    a few frames (median rotation, mean unit translation). Used for all frames so
    triangulated depth has a constant scale. `feats`: pass a cached extractor
    (FeatCache.feats) so the aggregation frames are not re-extracted."""
    feats = feats or matcher.feats
    Rs, ts = [], []
    idxs = np.linspace(0, len(ref_frames) - 1, min(n_agg, len(ref_frames))).round().astype(int)
    for i in idxs:
        fL = feats(ref_frames[i][1]); fR = feats(stereo_frames[i][1])
        idx, kL, kR = matcher.match(fL, fR)
        if idx.shape[0] < MIN_MATCHES:
            continue
        out = essential_pose(kL[idx[:, 0]], kR[idx[:, 1]], K_L, K_R)
        if out is not None:
            Rs.append(out[0]); ts.append(out[1])
    if not Rs:
        return None
    R_lr = R.from_matrix(np.stack(Rs)).mean().as_matrix()
    t_lr = np.mean(np.stack(ts), 0); t_lr /= (np.linalg.norm(t_lr) + 1e-12)
    return R_lr, t_lr


def triangulate(kL, kR, K_L, K_R, R_lr, t_lr):
    P1 = K_L @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_R @ np.hstack([R_lr, t_lr.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, kL.T, kR.T)
    X = (X[:3] / X[3]).T                            # (N,3) in e1L frame
    return X


def _umeyama_weighted(src, dst, w):
    """Weighted Sim(3) src->dst closed form (w >= 0 per point). Returns (s,R,t)."""
    w = np.asarray(w, float); wsum = w.sum()
    if wsum < 1e-9:
        w = np.ones(len(src)); wsum = float(len(src))
    ms = (w[:, None] * src).sum(0) / wsum
    md = (w[:, None] * dst).sum(0) / wsum
    sc, dc = src - ms, dst - md
    Sigma = (dc * w[:, None]).T @ sc / wsum
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Rr = U @ S @ Vt
    var = (w * (sc ** 2).sum(1)).sum() / wsum
    s = np.trace(np.diag(D) @ S) / var if var > 1e-12 else 1.0
    t = md - s * Rr @ ms
    return s, Rr, t


def umeyama_3d(src, dst, w=None, robust="mad2", n_iter=5, ransac_iters=0, ransac_thr=None):
    """Robust Sim(3) src->dst.

    ORGAN/LOCAL-MOTION handling: for a same-time cross-camera rigid pose, any
    correct correspondence gives the exact transform, so points that disagree
    with the dominant rigid model are either mismatches or corrupted by
    inter-camera sync jitter / motion blur (error ~ image velocity). Both are
    handled here as outliers / low-weight points.

      w        : optional prior weights (e.g. 1/velocity, approach (2)).
      robust   : 'mad2' = legacy single median+2sigma reject; 'irls' = iteratively
                 reweighted (Tukey) least squares (approach (1)); 'none'.
      ransac_iters>0 : minimal-sample RANSAC on the rigid model first.
    """
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    n = len(src)
    if n < 4:
        return None
    w0 = np.ones(n) if w is None else np.clip(np.asarray(w, float), 0, None)

    inl = np.ones(n, bool)
    if ransac_iters and n >= 6:
        rng = np.random.default_rng(0)
        best = None
        thr = ransac_thr
        for _ in range(ransac_iters):
            idx = rng.choice(n, 4, replace=False)
            try:
                s, Rr, t = _umeyama_weighted(src[idx], dst[idx], w0[idx])
            except np.linalg.LinAlgError:
                continue
            res = np.linalg.norm((s * (Rr @ src.T).T + t) - dst, axis=1)
            if thr is None:
                thr = np.median(res) + 1e-6
            cnt = int((res < thr).sum())
            if best is None or cnt > best[0]:
                best = (cnt, res < thr)
        if best is not None and best[1].sum() >= 4:
            inl = best[1]

    s, Rr, t = _umeyama_weighted(src[inl], dst[inl], w0[inl])

    if robust == "mad2":
        res = np.linalg.norm((s * (Rr @ src.T).T + t) - dst, axis=1)
        keep = inl & (res < (np.median(res) + 2 * (np.std(res) + 1e-9)))
        if keep.sum() >= 4:
            s, Rr, t = _umeyama_weighted(src[keep], dst[keep], w0[keep])
    elif robust == "irls":
        for _ in range(n_iter):
            res = np.linalg.norm((s * (Rr @ src.T).T + t) - dst, axis=1)
            sig = np.median(res) + 1e-9
            u = res / (4.685 * sig)                      # Tukey biweight
            tw = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
            wt = w0 * tw * inl
            if wt.sum() < 1e-9 or (wt > 0).sum() < 4:
                break
            s, Rr, t = _umeyama_weighted(src, dst, wt)
    return s, Rr, t


def _T(R_, t_):
    T = np.eye(4); T[:3, :3] = R_; T[:3, 3] = np.asarray(t_).reshape(3); return T


def frame_to_initial(raw):
    """raw: list of (fid, 4x4 T(t) or None). Return rows = T(0)^{-1} T(t), frame0 identity."""
    T0 = next((T for _, T in raw if T is not None), np.eye(4))
    T0inv = np.linalg.inv(T0)
    rows = []
    for fid, T in raw:
        if T is None:
            rows.append((fid, np.full(3, np.nan), np.full(4, np.nan)))
        else:
            rel = T0inv @ T
            rows.append((fid, rel[:3, 3], R.from_matrix(rel[:3, :3]).as_quat()))
    return rows


def temporal_ba(rows, lam=10.0, weights=None):
    """Temporal smoothing of the translation track with a DATA term + 2nd-difference
    (acceleration) penalty: solve (W + lam*D2^T D2) c = W c_meas per coordinate.
    Unlike median smoothing the data term preserves genuine motion (e.g. zoom_in)
    while removing per-frame jitter. ATE depends only on translation, so rotation is
    kept as measured. NaN frames stay NaN (interpolated only to anchor the smoother)."""
    fids = [r[0] for r in rows]; Q = [r[2] for r in rows]
    C = np.array([np.asarray(r[1], float) for r in rows])
    n = len(C); valid = np.isfinite(C).all(1)
    if valid.sum() < 3:
        return rows
    idx = np.arange(n); Cf = C.copy()
    for d in range(3):                                    # interp invalid to anchor smoother
        Cf[~valid, d] = np.interp(idx[~valid], idx[valid], C[valid, d])
    w = (np.ones(n) if weights is None else np.asarray(weights, float)).copy()
    w[~valid] = 0.0                                       # invalid: no data term (pure smoothness)
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    A = np.diag(w) + lam * (D2.T @ D2)
    out = Cf.copy()
    for d in range(3):
        out[:, d] = np.linalg.solve(A, w * Cf[:, d])
    return [(fids[i], out[i] if valid[i] else np.full(3, np.nan), Q[i]) for i in range(n)]


def smooth_centers(rows, win=5):
    """Median-filter the translation track over time (trajectory is smooth)."""
    fids = [r[0] for r in rows]
    C = np.array([r[1] for r in rows])
    valid = np.isfinite(C).all(1)
    out = C.copy()
    half = win // 2
    for i in range(len(C)):
        lo, hi = max(0, i - half), min(len(C), i + half + 1)
        seg = C[lo:hi][valid[lo:hi]]
        if valid[i] and len(seg):
            out[i] = np.median(seg, axis=0)
    return [(fids[i], out[i], rows[i][2]) for i in range(len(rows))]


# ----------------------------- per-sequence ----------------------------------
def predict_tri3d(seq, matcher, smooth=False, stats=None):
    """3D-3D cross-camera: triangulate e1 (L-R) and e2 (L-R) stereo clouds, pair via
    e1L<->e2L matches, Umeyama -> T_{e2<-e1} with consistent scale; frame-to-initial.
    If stats is a list, append per-frame {fid,n_pairs,scale,method}."""
    K = load_K(seq)
    K1L, K1R, K2L, K2R = K["K1_L"], K["K1_R"], K["K2_L"], K["K2_R"]
    e1L = frame_list(seq, "endoscope1", "L"); e1R = frame_list(seq, "endoscope1", "R")
    e2L = frame_list(seq, "endoscope2", "L"); e2R = frame_list(seq, "endoscope2", "R")
    e1R_m = {f: p for f, p in e1R}; e2L_m = {f: p for f, p in e2L}; e2R_m = {f: p for f, p in e2R}

    st1 = estimate_stereo(matcher, e1L, e1R, K1L, K1R, n_agg=_NAGG)
    st2 = estimate_stereo(matcher, e2L, e2R, K2L, K2R, n_agg=_NAGG)
    flow = FlowVelocity(e1L) if _VELW > 0 else None
    raw = []
    conf = {}          # fid -> per-frame pose confidence (support x geometric spread)
    for fid, pL in e1L:
        if st1 is None or st2 is None or fid not in e2L_m or fid not in e1R_m or fid not in e2R_m:
            raw.append((fid, None)); continue
        f1L = matcher.feats(pL); f1R = matcher.feats(e1R_m[fid])
        f2L = matcher.feats(e2L_m[fid]); f2R = matcher.feats(e2R_m[fid])
        i1, k1L, k1R = matcher.match(f1L, f1R)          # e1 stereo
        i2, k2L, k2R = matcher.match(f2L, f2R)          # e2 stereo
        ic, kc1, kc2 = matcher.match(f1L, f2L)          # cross e1L<->e2L
        if min(i1.shape[0], i2.shape[0], ic.shape[0]) < MIN_MATCHES:
            raw.append((fid, None))
            if stats is not None: stats.append({"fid": fid, "n_pairs": 0, "scale": float("nan"), "method": "no_match"})
            continue
        X1 = triangulate(k1L[i1[:, 0]], k1R[i1[:, 1]], K1L, K1R, *st1)
        X2 = triangulate(k2L[i2[:, 0]], k2R[i2[:, 1]], K2L, K2R, *st2)
        d1 = {int(a): X1[r] for r, a in enumerate(i1[:, 0])}
        d2 = {int(a): X2[r] for r, a in enumerate(i2[:, 0])}
        src, dst, pxy = [], [], []
        for r in range(ic.shape[0]):
            a, d = int(ic[r, 0]), int(ic[r, 1])
            if a in d1 and d in d2:
                x1, x2 = d1[a], d2[d]
                if np.isfinite(x1).all() and np.isfinite(x2).all() and x1[2] > 0 and x2[2] > 0:
                    src.append(x1); dst.append(x2); pxy.append(kc1[a])   # e1L pixel of this match
        if len(src) >= MIN_PNP:
            S, D = np.array(src), np.array(dst)
            w = flow.weights_at(fid, np.array(pxy), alpha=_VELW, mode=_VELMODE) if flow is not None else None
            out = umeyama_3d(S, D, w=w, robust=_ROBUST,
                             ransac_iters=_RANSAC)          # e1 coords -> e2 coords
            if out is not None:
                s, Rr, t = out
                resid = float(np.median(np.linalg.norm((s * (Rr @ S.T).T + t) - D, axis=1)))
                # confidence for grid-BA: support (inliers) x geometric spread of the
                # 3D correspondences, divided by fit residual. A clustered / near-
                # degenerate point set (the scene_5-type ill-conditioning) yields low
                # spread -> low confidence -> the temporal smoothness prior pulls that
                # frame toward its better-conditioned neighbours.
                spread = float(np.median(np.linalg.norm(S - np.median(S, 0), axis=1)))
                conf[fid] = len(src) * spread / (1.0 + 5.0 * resid)
                if stats is not None:
                    stats.append({"fid": fid, "n_pairs": len(src), "scale": float(s), "method": "umeyama",
                                  "residual": resid, "n_cross": int(ic.shape[0]), "spread": spread,
                                  "conf": conf[fid],
                                  "depth1": float(np.median(S[:, 2])), "depth2": float(np.median(D[:, 2]))})
                raw.append((fid, _T(Rr, t))); continue
        if stats is not None: stats.append({"fid": fid, "n_pairs": len(src), "scale": float("nan"), "method": "fallback",
                                            "residual": float("nan"), "n_cross": int(ic.shape[0]), "depth1": float("nan"), "depth2": float("nan")})
        raw.append((fid, None))
    rows = frame_to_initial(raw)
    if smooth:
        rows = smooth_centers(rows)
    return rows, conf


# ------------------------------------------------------------ grid BA (temporal)

class FeatCache:
    """Extract each frame's features once. The plain tri3d path re-extracts inside
    estimate_stereo (nagg=9999 -> every frame twice); the grid path also needs
    features at t and t+k, so caching is both a speedup and a prerequisite."""

    def __init__(self, matcher):
        self.m = matcher
        self._c = {}

    def feats(self, path):
        f = self._c.get(path)
        if f is None:
            f = self.m.feats(path)
            self._c[path] = f
        return f

    def match(self, p0, p1):
        return self.m.match(self.feats(p0), self.feats(p1))


def _egomotion(fc, pL_t, pL_u, dt, du):
    """Rigid motion M = T_{cam(u)<-cam(t)} of one scope from frame t to frame u.
    Reuses the same-time stereo clouds dt/du (kp_index -> 3D) computed once per
    frame; only the temporal L-L match is new. IRLS robust. Deformation makes
    this approximate (hence the weak weight in the grid solve).
    Returns (M 4x4, n_inliers) or (None, 0)."""
    iL, _, _ = fc.match(pL_t, pL_u)                          # e?L(t) <-> e?L(u)
    if iL.shape[0] < MIN_MATCHES:
        return None, 0
    src, dst = [], []
    for r in range(iL.shape[0]):
        a, b = int(iL[r, 0]), int(iL[r, 1])
        if a in dt and b in du:
            x, y = dt[a], du[b]
            if x[2] > 0 and y[2] > 0:
                src.append(x); dst.append(y)
    if len(src) < MIN_PNP:
        return None, 0
    out = umeyama_3d(np.array(src), np.array(dst), robust="irls")   # X(u) = s R X(t) + t
    if out is None:
        return None, 0
    s, Rr, t = out
    return _T(Rr, t), len(src)


def predict_tri3d_grid(seq, matcher, k_temporal=1, lam=2.0, lam_rel=1.0):
    """Grid BA: same-time cross-camera poses (absolute) + temporal egomotion edges
    (relative), fused into a smooth translation track by linear least squares.

    Exact rigid relation T(t+1) = M_e2(t) T(t) M_e1(t)^{-1}; with rotations fixed
    to the per-frame estimates it is LINEAR in the translations:
        c(t+1) = B(t) c(t) + d(t),   B = M_e2.R,  d = M_e2.t - B R(t) M_e1.R^T M_e1.t
    We minimise  w_abs|c(t)-c_meas(t)|^2 + lam_rel|c(t+1)-B c(t)-d|^2
                 + lam|c(t+1)-2c(t)+c(t-1)|^2.  ATE needs only translation.
    """
    K = load_K(seq)
    K1L, K1R, K2L, K2R = K["K1_L"], K["K1_R"], K["K2_L"], K["K2_R"]
    e1L = frame_list(seq, "endoscope1", "L"); e1R = frame_list(seq, "endoscope1", "R")
    e2L = frame_list(seq, "endoscope2", "L"); e2R = frame_list(seq, "endoscope2", "R")
    p1R = {f: p for f, p in e1R}; p2L = {f: p for f, p in e2L}; p2R = {f: p for f, p in e2R}
    fc = FeatCache(matcher)

    st1 = estimate_stereo(matcher, e1L, e1R, K1L, K1R, n_agg=_NAGG, feats=fc.feats)
    st2 = estimate_stereo(matcher, e2L, e2R, K2L, K2R, n_agg=_NAGG, feats=fc.feats)
    fids = [f for f, _ in e1L]
    n = len(fids)
    pos = {f: i for i, f in enumerate(fids)}
    cloud = {}      # fid -> (scope) dict of kp_index -> 3D, reused by temporal egomotion

    # -- same-time absolute poses (reuse the tri3d per-frame estimate) --
    R_abs = [None] * n; c_meas = np.full((n, 3), np.nan); w_abs = np.zeros(n)
    for i, (fid, pL) in enumerate(e1L):
        if st1 is None or st2 is None or fid not in p2L or fid not in p1R or fid not in p2R:
            continue
        i1, k1L, k1R = fc.match(pL, p1R[fid])
        i2, k2L, k2R = fc.match(p2L[fid], p2R[fid])
        ic, kc1, kc2 = fc.match(pL, p2L[fid])
        if min(i1.shape[0], i2.shape[0], ic.shape[0]) < MIN_MATCHES:
            continue
        X1 = triangulate(k1L[i1[:, 0]], k1R[i1[:, 1]], K1L, K1R, *st1)
        X2 = triangulate(k2L[i2[:, 0]], k2R[i2[:, 1]], K2L, K2R, *st2)
        d1 = {int(a): X1[r] for r, a in enumerate(i1[:, 0])}
        d2 = {int(a): X2[r] for r, a in enumerate(i2[:, 0])}
        cloud[fid] = (d1, d2)                             # reused by temporal egomotion
        src, dst = [], []
        for r in range(ic.shape[0]):
            a, d = int(ic[r, 0]), int(ic[r, 1])
            if a in d1 and d in d2 and d1[a][2] > 0 and d2[d][2] > 0:
                src.append(d1[a]); dst.append(d2[d])
        if len(src) < MIN_PNP:
            continue
        out = umeyama_3d(np.array(src), np.array(dst), robust="irls")
        if out is None:
            continue
        s, Rr, t = out
        S = np.array(src)
        resid = float(np.median(np.linalg.norm((s * (Rr @ S.T).T + t) - np.array(dst), axis=1)))
        spread = float(np.median(np.linalg.norm(S - np.median(S, 0), axis=1)))
        R_abs[i] = Rr; c_meas[i] = t; w_abs[i] = len(src) * spread / (1.0 + 5.0 * resid)

    valid = np.array([R is not None for R in R_abs])
    if valid.sum() < 3:
        rows, conf = predict_tri3d(seq, matcher)         # fall back to plain path
        return rows, conf
    w_abs = w_abs / (np.median(w_abs[valid]) + 1e-9)     # normalise: median-valid weight -> 1

    # -- temporal egomotion edges: c(t+k) = B c(t) + d --
    edges = []                                            # (i, j, B(3x3), d(3), w)
    for i in range(n):
        for k in range(1, k_temporal + 1):
            j = i + k
            if j >= n or not valid[i] or not valid[j]:
                continue
            fi, fj = fids[i], fids[j]
            if fi not in cloud or fj not in cloud:
                continue
            (d1i, d2i), (d1j, d2j) = cloud[fi], cloud[fj]
            M1, nn1 = _egomotion(fc, e1L[i][1], e1L[j][1], d1i, d1j)   # e1 clouds reused
            M2, nn2 = _egomotion(fc, p2L[fi], p2L[fj], d2i, d2j)      # e2 clouds reused
            if M1 is None or M2 is None:
                continue
            B = M2[:3, :3]
            d = M2[:3, 3] - B @ R_abs[i] @ M1[:3, :3].T @ M1[:3, 3]
            we = lam_rel * min(nn1, nn2) / 50.0            # weight ~ temporal match support
            edges.append((i, j, B, d, we))

    # -- assemble linear least squares for c in R^{3n} --
    A = np.zeros((3 * n, 3 * n)); b = np.zeros(3 * n)
    def blk(i): return slice(3 * i, 3 * i + 3)
    for i in range(n):                                    # absolute data term
        if valid[i]:
            A[blk(i), blk(i)] += w_abs[i] * np.eye(3)
            b[blk(i)] += w_abs[i] * c_meas[i]
    for (i, j, B, d, we) in edges:                       # relative: |c_j - B c_i - d|^2
        A[blk(j), blk(j)] += we * np.eye(3)
        A[blk(j), blk(i)] += -we * B
        A[blk(i), blk(j)] += -we * B.T
        A[blk(i), blk(i)] += we * (B.T @ B)
        b[blk(j)] += we * d
        b[blk(i)] += -we * (B.T @ d)
    for i in range(1, n - 1):                            # 2nd-difference smoothness
        for (col, sgn) in ((i - 1, 1.0), (i, -2.0), (i + 1, 1.0)):
            for (col2, sgn2) in ((i - 1, 1.0), (i, -2.0), (i + 1, 1.0)):
                A[blk(col), blk(col2)] += lam * sgn * sgn2 * np.eye(3)
    A += 1e-6 * np.eye(3 * n)                            # regularise gauge / gaps
    c = np.linalg.solve(A, b).reshape(n, 3)

    raw = [(fids[i], _T(R_abs[i], c[i]) if valid[i] else None) for i in range(n)]
    return frame_to_initial(raw), {fids[i]: w_abs[i] for i in range(n) if valid[i]}


def predict_sequence(seq, matcher, method="stereo_pnp"):
    K = load_K(seq)
    K1L, K1R, K2L = K["K1_L"], K["K1_R"], K["K2_L"]
    e1L = frame_list(seq, *CAM_REF)
    e2L = frame_list(seq, *CAM_SRC)
    e1R = frame_list(seq, *STEREO_REF)
    ids = [fid for fid, _ in e1L]
    e2L_map = {fid: p for fid, p in e2L}
    e1R_map = {fid: p for fid, p in e1R}

    R_lr = t_lr = None
    if method == "stereo_pnp":
        st = estimate_stereo(matcher, e1L, e1R, K1L, K1R)
        if st is not None:
            R_lr, t_lr = st

    rows = []
    for i, (fid, pL) in enumerate(e1L):
        if i == 0:
            rows.append((fid, np.zeros(3), np.array([0, 0, 0, 1.0]))); continue
        if fid not in e2L_map:
            rows.append((fid, np.full(3, np.nan), np.full(4, np.nan))); continue
        fe1L = matcher.feats(pL)
        fe2L = matcher.feats(e2L_map[fid])
        idx_c, kc0, kc1 = matcher.match(fe1L, fe2L)          # e1L <-> e2L (cross)
        if idx_c.shape[0] < MIN_MATCHES:
            rows.append((fid, np.full(3, np.nan), np.full(4, np.nan))); continue

        pose = None
        if method == "stereo_pnp" and R_lr is not None and fid in e1R_map:
            fe1R = matcher.feats(e1R_map[fid])
            idx_s, ks0, ks1 = matcher.match(fe1L, fe1R)      # e1L <-> e1R (stereo), shared e1L feats
            if idx_s.shape[0] >= MIN_MATCHES:
                X = triangulate(ks0[idx_s[:, 0]], ks1[idx_s[:, 1]], K1L, K1R, R_lr, t_lr)
                d3 = {int(a): X[r] for r, a in enumerate(idx_s[:, 0])}   # e1L kpt idx -> 3D
                obj, img = [], []
                for r, a in enumerate(idx_c[:, 0]):
                    if int(a) in d3:
                        x = d3[int(a)]
                        if np.isfinite(x).all() and x[2] > 0:
                            obj.append(x); img.append(kc1[idx_c[r, 1]])
                if len(obj) >= MIN_PNP:
                    obj = np.array(obj, np.float64); img = np.array(img, np.float64)
                    ok, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj, img, K2L, None, reprojectionError=3.0,
                        iterationsCount=200, flags=cv2.SOLVEPNP_EPNP)
                    if ok and inl is not None and len(inl) >= MIN_PNP:
                        Rr, _ = cv2.Rodrigues(rvec)
                        pose = (Rr, tvec.reshape(3))          # T_{e2<-e1}, metric-scale t
        if pose is None:                                       # fallback: essential, unit t
            out = essential_pose(kc0[idx_c[:, 0]], kc1[idx_c[:, 1]], K1L, K2L)
            if out is None:
                rows.append((fid, np.full(3, np.nan), np.full(4, np.nan))); continue
            pose = out
        Rr, t = pose
        rows.append((fid, t, R.from_matrix(Rr).as_quat()))
    return rows


def anchor_frame0(rows):
    """Contract: frame 0 must be exactly (t=0, q=identity). temporal_ba smooths the
    whole translation track and can nudge frame 0 off the origin, so re-anchor by
    subtracting frame 0's translation from every frame (a global translation, which
    the Sim(3) ATE alignment absorbs and which preserves inter-frame structure)."""
    rows = [list(r) for r in rows]
    t0 = np.asarray(rows[0][1], float)
    if not np.isfinite(t0).all():
        t0 = next((np.asarray(r[1], float) for r in rows if np.isfinite(r[1]).all()), np.zeros(3))
    for i in range(len(rows)):
        if np.isfinite(rows[i][1]).all():
            rows[i][1] = np.asarray(rows[i][1], float) - t0
    rows[0][1] = np.zeros(3)
    rows[0][2] = np.array([0.0, 0.0, 0.0, 1.0])
    return [tuple(r) for r in rows]


def write_rows(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for fid, t, q in rows:
            f.write("%d %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n" %
                    (fid, t[0], t[1], t[2], q[0], q[1], q[2], q[3]))


def is_seq(d):
    return os.path.isfile(os.path.join(d, "K.txt")) and os.path.isdir(os.path.join(d, *CAM_REF))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_DIR); ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--method", default="tri3d", choices=["tri3d", "stereo_pnp", "baseline"])
    ap.add_argument("--smooth", action="store_true", help="median-filter trajectory (legacy; prefer temporal-BA)")
    ap.add_argument("--nagg", type=int, default=9999, help="frames to aggregate for stereo extrinsic (default: all)")
    ap.add_argument("--lam", type=float, default=2.0, help="temporal-BA smoothness weight (0 disables)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", type=int, default=0, help="fp16 matcher (default off; hurt ATE 0.953->1.05 on full-19)")
    ap.add_argument("--extractor", default="aliked", choices=list(_EXTRACTORS))
    ap.add_argument("--name", default="pose_predictions.txt")
    ap.add_argument("--robust", default="mad2", choices=["none", "mad2", "irls"],
                    help="(1) robust rigid fit: irls = Tukey IRLS")
    ap.add_argument("--ransac", type=int, default=0, help="(1) RANSAC iters on rigid model (0=off)")
    ap.add_argument("--velw", type=float, default=0.0,
                    help="(2) velocity down-weighting strength alpha (w=1/(1+alpha*vel), 0=off)")
    ap.add_argument("--velmode", default="abs", choices=["abs", "residual"],
                    help="(2) abs velocity or residual-from-global-affine-flow")
    ap.add_argument("--confw", default="none", choices=["none", "conf"],
                    help="(grid-BA) confidence-weight the temporal-BA data term by per-frame pose confidence")
    ap.add_argument("--confpow", type=float, default=1.0, help="exponent on the per-frame confidence weight")
    ap.add_argument("--gridba", type=int, default=1, help="grid-BA temporal window K (default 1). 0 disables. Adds temporal egomotion edges up to t+-K.")
    ap.add_argument("--lamrel", type=float, default=0.3, help="grid-BA relative (temporal edge) weight (default 0.3)")
    a = ap.parse_args()
    globals()["_NAGG"] = a.nagg
    globals()["_ROBUST"] = a.robust
    globals()["_RANSAC"] = a.ransac
    globals()["_VELW"] = a.velw
    globals()["_VELMODE"] = a.velmode
    seqs = [a.input] if is_seq(a.input) else [os.path.join(a.input, n) for n in sorted(os.listdir(a.input))
                                              if is_seq(os.path.join(a.input, n))]
    matcher = Matcher(a.device, a.extractor, fp16=bool(a.fp16))
    for seq in seqs:
        name = os.path.basename(os.path.normpath(seq))
        out = os.path.join(a.output, name, a.name)     # contract: /output/<seq>/pose_predictions.txt
        if a.method == "tri3d" and a.gridba > 0:
            rows, conf = predict_tri3d_grid(seq, matcher, k_temporal=a.gridba,
                                            lam=a.lam, lam_rel=a.lamrel)
        elif a.method == "tri3d":
            rows, conf = predict_tri3d(seq, matcher, a.smooth)
        else:
            rows, conf = predict_sequence(seq, matcher, a.method), {}
        if a.method == "tri3d" and a.gridba == 0 and a.lam > 0:  # temporal-BA (grid-BA has its own smoothness)
            weights = None
            if a.confw == "conf" and conf:
                c = np.array([conf.get(f, 0.0) for f, _, _ in rows], float)
                med = np.median(c[c > 0]) if (c > 0).any() else 1.0
                weights = (np.clip(c / (med + 1e-9), 0.0, None)) ** a.confpow  # normalised, >0 only where a pose exists
            rows = temporal_ba(rows, lam=a.lam, weights=weights)
        rows = anchor_frame0(rows)                             # enforce frame0 = (t=0, q=identity)
        reg = 100.0 * np.mean([np.isfinite(t).all() for _, t, _ in rows])
        write_rows(rows, out)
        print(f"[{a.method}] {name}: {len(rows)} poses ({reg:.0f}% reg) -> {out}", flush=True)


if __name__ == "__main__":
    main()
