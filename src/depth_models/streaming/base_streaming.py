#!/usr/bin/env python3
"""
Shared multi-camera streaming pipeline for all DA3 backends.

Holds everything that is backend-agnostic: loading synchronized image
folders, batching them into chunks of ``chunk_size`` images, aligning
consecutive chunks full-vs-full via SIM3, accumulating the transform
chain, and saving per-view depth + warped extrinsics + intrinsics.

Optional ``mask_dirs`` (one binary motion-mask folder per input folder)
zero the confidence of moving pixels during chunk alignment only — depth
outputs are unaffected.

A short final chunk is padded at its start — with copies of the previous
chunk's tail, or with duplicated images when the whole sequence fits in a
single chunk — so fixed-N exports always see exactly ``chunk_size`` images
in time order.  Padding is kept for the alignment (which uses all frames
including padding) and discarded when saving outputs.

Concrete backends (PyTorch, torch.export/torch.compile, TorchScript)
inherit :class:`BaseStreaming` and implement exactly two hooks:

- ``_load_model()`` — load the backend model and set
  ``self.model_num_views`` (``None`` for dynamic-N exports).
- ``_process_chunk(start, end) -> dict`` — run one chunk and return
  ``{"depth", "conf", "extrinsics", "intrinsics"}`` numpy arrays.

Run any backend through ``da3_streaming/run_stream.py --backend ...``.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_models.streaming.loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    weighted_align_point_maps,
)
from utils.depth_utils import (
    decode_packed_rgb_to_log_depth,
    encode_log_depth_to_packed_rgb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics):
    """depth [N,H,W], intrinsics [N,3,3], extrinsics [N,3,4] w2c → [N,H,W,3] world."""
    depth_t = torch.as_tensor(depth, dtype=torch.float32)
    intrs_t = torch.as_tensor(intrinsics, dtype=torch.float32)
    extrs_t = torch.as_tensor(extrinsics, dtype=torch.float32)
    N, H, W = depth_t.shape
    dev = depth_t.device
    u = torch.arange(W, device=dev).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=dev).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=dev)
    pixel_coords = torch.cat([u, v, ones], dim=-1)
    cam = torch.einsum("nij,nhwj->nhwi", torch.inverse(intrs_t), pixel_coords)
    cam = cam * depth_t.unsqueeze(-1)
    cam_h = torch.cat([cam, ones], dim=-1)
    ext4 = torch.zeros(N, 4, 4, device=dev)
    ext4[:, :3, :4] = extrs_t
    ext4[:, 3, 3] = 1.0
    world = torch.einsum("nij,nhwj->nhwi", torch.inverse(ext4), cam_h)
    return world[..., :3].cpu().numpy()


def _warp_extrinsics(
    extrinsics_3x4: np.ndarray, s: float, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Apply SIM3 to world-to-camera extrinsics: w2c' = w2c @ inv(S)."""
    N = extrinsics_3x4.shape[0]
    S = np.eye(4, dtype=np.float32)
    S[:3, :3] = s * R
    S[:3, 3] = t
    S_inv = np.linalg.inv(S)
    w2c_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
    w2c_4x4[:, :3, :] = extrinsics_3x4
    w2c_4x4[:, 3, 3] = 1.0
    warped = w2c_4x4 @ S_inv
    return warped[:, :3, :]



# ===========================================================================
# Base class
# ===========================================================================
class BaseStreaming:
    """Shared chunked SIM3-alignment streaming pipeline.

    Subclasses implement the backend-specific hooks ``_load_model`` and
    ``_process_chunk`` (see the module docstring).
    """

    # num_views of a fixed-N export, or None for dynamic-N models.  Set by
    # subclasses in ``_load_model``; must equal chunk_size (see
    # _check_fixed_num_views).  Every chunk is padded to exactly chunk_size
    # images regardless of export type, so any total image count works.
    model_num_views: int | None = None

    def __init__(
        self,
        input_dirs: list[str],
        save_dir: str,
        config: dict,
        chunk_size: int = 24,
        start_frame: int = 0,
        max_frames: int | None = None,
        interval: int = 1,
        device: str | None = None,
        mask_dirs: list[str] | None = None,
    ):
        self.config = config
        self.input_dirs = [os.path.normpath(d) for d in input_dirs]
        self.num_cams = len(self.input_dirs)

        # Optional per-camera motion-mask folders (one per input_dir, same
        # order and image count); used only to zero moving-pixel confidence
        # during chunk alignment.
        if mask_dirs is not None and len(mask_dirs) != self.num_cams:
            raise ValueError(
                f"mask_dirs ({len(mask_dirs)} folders) must match the number "
                f"of input_dirs ({self.num_cams})"
            )
        self.mask_dirs = [os.path.normpath(d) for d in mask_dirs] if mask_dirs else None

        # Output folders are named after the input folders — basenames must differ.
        basenames = [Path(d).name for d in self.input_dirs]
        if len(set(basenames)) != len(basenames):
            raise ValueError(
                f"Duplicate input folder basenames ({basenames}) would collide "
                "in the output directory."
            )

        if chunk_size % self.num_cams != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be divisible by the number of "
                f"input folders ({self.num_cams})"
            )
        self.chunk_size = chunk_size
        self.frames_per_chunk = chunk_size // self.num_cams

        self.start_frame = start_frame
        self.max_frames = max_frames
        self.interval = interval

        self.device = device

        self.output_dir = save_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.img_list: list[str] = []
        self.mask_paths: list[str] | None = None  # parallel to img_list
        self.chunk_indices: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Backend-specific hooks
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the backend model and set ``self.model_num_views``."""
        raise NotImplementedError

    def _process_chunk(self, start: int, end: int) -> dict:
        """Run the model on ``self.img_list[start:end]``.

        Returns ``{"depth", "conf", "extrinsics", "intrinsics"}`` numpy
        arrays for the chunk.
        """
        raise NotImplementedError

    def _check_fixed_num_views(self, reexport_hint: str) -> None:
        """A fixed-N export must match chunk_size exactly (dynamic-N has None)."""
        if self.model_num_views is not None and self.model_num_views != self.chunk_size:
            raise ValueError(
                f"Model was exported with fixed num_views={self.model_num_views}, "
                f"but chunk_size={self.chunk_size}. {reexport_hint}"
            )

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_image_list(self) -> int:
        """Load synchronized image lists from all cameras.

        All folders must contain the same sorted frame stems.  Returns the
        number of loaded time steps.
        """
        print(f"Loading images from {self.num_cams} folders:")
        exts = ("*.jpg", "*.jpeg", "*.png")
        per_cam: list[list[Path]] = []
        for d in self.input_dirs:
            files = sorted(p for ext in exts for p in Path(d).glob(ext))
            if not files:
                raise RuntimeError(f"No images found in {d}.")
            per_cam.append(files)
            print(f"  {d}: {len(files)} images")

        # All cameras must see the same synchronized stems.
        reference = [p.stem for p in per_cam[0]]
        for cam_i, files in enumerate(per_cam[1:], start=1):
            stems = [p.stem for p in files]
            if stems != reference:
                raise RuntimeError(
                    f"Stem mismatch between {self.input_dirs[0]} and "
                    f"{self.input_dirs[cam_i]}: the two folders do not contain "
                    "the same synchronized frame stems."
                )

        # start_frame matches the image stem number (e.g. 210 → frame_000210.jpg)
        target_stem = f"frame_{self.start_frame:06d}"
        _start_idx = None
        for _i, _s in enumerate(reference):
            if _s == target_stem:
                _start_idx = _i
                break
        if _start_idx is None:
            raise RuntimeError(
                f"start_frame={self.start_frame} (looking for {target_stem}) not found"
            )
        per_cam = [files[_start_idx:] for files in per_cam]

        if self.interval > 1:
            per_cam = [files[:: self.interval] for files in per_cam]
            print(f"  Interval {self.interval}: {len(per_cam[0])} time steps available")

        available = len(per_cam[0])
        if self.max_frames is None:
            self.max_frames = available
        n = min(self.max_frames, available)

        # Interleave per time step: [t0_c0, t0_c1, ..., t0_cK, t1_c0, ...]
        img_list: list[str] = []
        for t in range(n):
            for cam_files in per_cam:
                img_list.append(str(cam_files[t]))
        self.img_list = img_list

        print(
            f"  Loaded {n} time steps x {self.num_cams} cameras "
            f"({len(img_list)} images interleaved)"
        )
        return n

    def _load_mask_paths(self) -> None:
        """Build the per-image mask path list parallel to ``self.img_list``.

        Mask folders are matched to input folders by position, and mask
        files to frames by number (``frame_000210.jpg`` → ``mask_000210``
        in the same mask folder).  Must be called after
        ``_get_chunk_indices`` — padding copies spliced into
        ``self.img_list`` duplicate their mask automatically.  Every
        selected image must have a mask (the folders hold the same number
        of images in the same order as the input folders).
        """
        self.mask_paths = None
        if self.mask_dirs is None:
            return

        exts = ("*.jpg", "*.jpeg", "*.png")
        per_cam: list[dict[int, str]] = []
        for d in self.mask_dirs:
            by_frame: dict[int, str] = {}
            for ext in exts:
                for p in Path(d).glob(ext):
                    try:
                        n = int(p.stem.split("_")[-1])
                    except ValueError:
                        continue
                    by_frame[n] = str(p)
            if not by_frame:
                raise RuntimeError(f"No mask images found in {d}.")
            per_cam.append(by_frame)

        mask_paths: list[str] = []
        for i, img_path in enumerate(self.img_list):
            try:
                n = int(Path(img_path).stem.split("_")[-1])
            except ValueError as e:
                raise RuntimeError(
                    f"Cannot parse frame number from image path {img_path}"
                ) from e
            mask = per_cam[i % self.num_cams].get(n)
            if mask is None:
                raise RuntimeError(
                    f"No mask for frame {n} (from {img_path}) in "
                    f"{self.mask_dirs[i % self.num_cams]}"
                )
            mask_paths.append(mask)
        self.mask_paths = mask_paths

    @staticmethod
    def _stack_chunk_masks(mask_paths: list[str], h: int, w: int) -> np.ndarray:
        """Load the binary masks of one chunk; 1 = static pixel, 0 = moving.

        Mask images are thresholded at 127 (JPEG-compressed binary) and
        inverted so static pixels become 1, then resized with
        nearest-neighbour interpolation to the conf grid ``(h, w)``.
        Returns ``(N, h, w)`` float32.
        """
        masks = []
        for p in mask_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Could not read mask {p}")
            m = (img < 127).astype(np.float32)
            if (m.shape[0], m.shape[1]) != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(m)
        return np.stack(masks)

    # ------------------------------------------------------------------
    # Chunk indices
    # ------------------------------------------------------------------
    def _get_chunk_indices(self):
        """Build full-size chunk windows, padding the final chunk at its start.

        Fixed-N exports need exactly ``chunk_size`` images per forward, so a
        short final chunk is padded with leading images: copies of the
        previous chunk's tail when there is one, or duplicated copies of the
        first images when the whole sequence fits in a single chunk.  The
        padding keeps frame time order inside the chunk, and real images
        always occupy the tail of each chunk.  Padding copies are spliced
        into ``self.img_list`` so every chunk stays a contiguous window —
        capture ``len(self.img_list)`` beforehand if the real count is
        needed.  Returns ``(chunks, num_chunks, pads)`` where ``pads[i]`` is
        the number of leading padding images in chunk ``i``.
        """
        L = len(self.img_list)
        nc = (L + self.chunk_size - 1) // self.chunk_size
        chunks = [(i * self.chunk_size, min((i + 1) * self.chunk_size, L)) for i in range(nc)]
        pads = [0] * nc

        start, end = chunks[-1]
        pad = self.chunk_size - (end - start)
        if pad > 0:
            if nc == 1:
                copies = (self.img_list * ((pad + L - 1) // L))[:pad]
                self.img_list = copies + self.img_list
                chunks[-1] = (0, self.chunk_size)
            else:
                self.img_list = (
                    self.img_list[:start]
                    + self.img_list[start - pad : start]
                    + self.img_list[start:]
                )
                chunks[-1] = (start, start + self.chunk_size)
            pads[-1] = pad

        return chunks, nc, pads

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def _align_pair(self, data_a: dict, data_b: dict) -> tuple[float, np.ndarray, np.ndarray]:
        pcd_a = depth_to_point_cloud_vectorized(
            data_a["depth"],
            data_a["intrinsics"],
            data_a["extrinsics"],
        )
        pcd_b = depth_to_point_cloud_vectorized(
            data_b["depth"],
            data_b["intrinsics"],
            data_b["extrinsics"],
        )
        thresh = min(np.median(data_a["conf"]), np.median(data_b["conf"])) * 0.1

        # Alignment-only motion masks zero the conf of moving pixels.
        # Multiplication creates fresh arrays — data["conf"] is not
        # modified in place.
        conf_a = data_a["conf"]
        conf_b = data_b["conf"]
        if data_a.get("align_mask") is not None:
            conf_a = conf_a * data_a["align_mask"]
        if data_b.get("align_mask") is not None:
            conf_b = conf_b * data_b["align_mask"]

        s, R, t = weighted_align_point_maps(
            pcd_a,
            conf_a,
            pcd_b,
            conf_b,
            conf_threshold=thresh,
            config=self.config,
            precompute_scale=None,
        )
        return s, R, t

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        t_run0 = time.perf_counter()
        num_steps = self._load_image_list()
        if num_steps == 0:
            raise RuntimeError("No frames found.")

        # Real image count, captured before _get_chunk_indices splices padding
        # copies into self.img_list.
        total_imgs = len(self.img_list)
        self.chunk_indices, num_chunks, self.chunk_pads = self._get_chunk_indices()
        print(
            f"Processing {total_imgs} images in {num_chunks} chunks "
            f"(size={self.chunk_size}, {self.frames_per_chunk} steps/chunk)"
        )
        if self.chunk_pads[-1]:
            pad = self.chunk_pads[-1]
            src = "duplicated" if num_chunks == 1 else "copied from the previous chunk"
            print(f"  Last chunk padded at its start with {pad} {src} images")

        # Built after the padding splice so duplicates map to the same mask.
        self._load_mask_paths()

        # Per-time-step metadata: (stem, chunk_idx, intr_Kx3x3)
        frame_meta: list[tuple[str, int, np.ndarray]] = []
        all_extrinsics: list[np.ndarray] = []
        sim3_chain: list[tuple[float, np.ndarray, np.ndarray]] = []
        prev_overlap: dict | None = None

        out_dirs = [
            os.path.join(self.output_dir, f"depth_{Path(d).name}") for d in self.input_dirs
        ]
        for d in out_dirs:
            os.makedirs(d, exist_ok=True)

        # Per-chunk inference (model forward) timings; the total covers the
        # whole run() including alignment and file I/O.
        self.chunk_times: list[float] = []
        for ci, (ch_start, ch_end) in enumerate(self.chunk_indices):
            print(f"[Chunk {ci}/{num_chunks}]")
            t_chunk0 = time.perf_counter()
            data = self._process_chunk(ch_start, ch_end)
            chunk_s = time.perf_counter() - t_chunk0
            self.chunk_times.append(chunk_s)
            print(f"  infer {chunk_s:.3f}s")
            n_imgs = data["depth"].shape[0]  # multiple of num_cams
            pad_imgs = self.chunk_pads[ci]

            # Alignment-only mask stack for this chunk's images, on the conf
            # grid (1 = static, 0 = moving).  Carried by prev_overlap into
            # the next _align_pair call.
            if self.mask_paths is not None:
                conf_h, conf_w = data["conf"].shape[1], data["conf"].shape[2]
                data["align_mask"] = self._stack_chunk_masks(
                    self.mask_paths[ch_start:ch_end], conf_h, conf_w
                )

            # Save depth per view using the source image stem; leading padding
            # images are discarded here (but were used for alignment below).
            # Depth is stored packed-RGB under the same "depth" key (encoded
            # log depth, see utils.depth_utils) — ~2x smaller in the
            # compressed npz than float32 — and decoded back on load
            # (load_npz_data checks the channel count).
            for local_t in range(pad_imgs, n_imgs, self.num_cams):
                stem = Path(self.img_list[ch_start + local_t]).stem
                for v in range(self.num_cams):
                    depth_path = os.path.join(out_dirs[v], f"{stem}.npz")
                    np.savez_compressed(
                        depth_path,
                        depth=encode_log_depth_to_packed_rgb(
                            data["depth"][local_t + v]
                        ),
                    )

                all_extrinsics.append(data["extrinsics"][local_t : local_t + self.num_cams].copy())
                frame_meta.append(
                    (stem, ci, data["intrinsics"][local_t : local_t + self.num_cams].copy())
                )

            # Align with previous chunk (full chunk vs full chunk)
            if ci > 0:
                s, R, t = self._align_pair(prev_overlap, data)
                sim3_chain.append((s, R, t))
                print(f"  {ci - 1}→{ci}: s={s:.4f}  |t|={np.linalg.norm(t):.4f}")

            prev_overlap = data  # the whole chunk IS the overlap

        # Accumulate transforms (chunk k → chunk 0)
        if num_chunks > 1:
            sim3_cum = accumulate_sim3_transforms(sim3_chain)
        else:
            sim3_cum = []

        # Apply alignment and save final outputs
        print("\nApplying alignment and saving final outputs ...")
        for (stem, ci, intr), ext in zip(frame_meta, all_extrinsics):
            if ci == 0 or ci > len(sim3_cum):
                s, R, t = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            else:
                s, R, t = sim3_cum[ci - 1]

            ext_warped = _warp_extrinsics(ext, s, R, t)

            for v in range(self.num_cams):
                depth_path = os.path.join(out_dirs[v], f"{stem}.npz")
                depth_raw = decode_packed_rgb_to_log_depth(
                    np.load(depth_path)["depth"]
                )
                np.savez_compressed(
                    depth_path,
                    depth=encode_log_depth_to_packed_rgb(depth_raw * s),
                    extrinsics=ext_warped[v],
                    intrinsics=intr[v],
                )

        total_s = time.perf_counter() - t_run0
        print(f"Done. {len(frame_meta)} time steps saved to {self.output_dir}/")
        return {"total_s": total_s, "chunk_times": self.chunk_times}
