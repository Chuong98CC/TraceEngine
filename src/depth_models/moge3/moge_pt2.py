"""MoGev3_PT2: minimal inference wrapper around the exported MoGe v3 pt2 checkpoint.

This is the trimmed, standalone core of `infer_pt2.moge_pt2`: load the
exported dense graph + the eager sparse refiner, preprocess an input image,
run inference, and post-process to camera-space outputs. No export code,
no visualization, no mesh/ply saving -- preprocessing, inference and
post-processing only.

Image entry follows the repo-wide contract (``utils.image_io``): ``infer``
accepts any ``ImageInput`` (path / PIL image / RGB HWC numpy / CHW tensor),
decoded to a CHW uint8 RGB tensor via :func:`to_image_tensor`, then stretched
to the fixed graph size and fed to the fp16 graph in [0, 1] (divide by 255,
no ImageNet normalization).

The exported graph (`weights/moge3/moge3_l.pt2`) contains the dense network
(DINOv2-L encoder + neck + heads + scale head) compiled for a fixed input
size (default 640x480) in fp16 with a dynamic batch dimension. The sparse
3D refiner cannot be exported (custom flex_gemm kernels), so its weights
live in a small companion checkpoint (`moge3_l_refiner.pt`) and run eagerly
on the graph's intermediate outputs. Post-processing (focal/shift recovery,
intrinsics, depth, metric scale, masking) is done in fp32 exactly like
`moge/model/v3.py` `MoGeModel.infer()`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .refiner import Sparse3DUNet

from .utils.postprocess import refine_logz, remap_points_exp, resize_channel_last, affine_to_camera
from utils.image_io import ImageInput, to_image_tensor, to_pixel_uint8

# Order of the exported graph's outputs; keep in sync with `ExportWrapper.forward`
# in the (repo-side) export script. Only `moge_pt2` consumes them here.
OUTPUT_NAMES = ('raw_coord', 'refiner_feature', 'normal', 'mask', 'metric_scale')


def load_pt2(path) -> nn.Module:
    """Load an exported program and return its runnable GraphModule."""
    exported = torch.export.load(path)
    return exported.module()


class MoGev3_PT2:
    """MoGe v3 dense network (exported graph) + eager sparse refiner.

    Usage:
        model = MoGev3_PT2('weights/moge3/moge3_l.pt2')
        out = model.infer('frame.jpg')  # any ImageInput
        # -> {points, depth, mask, intrinsics, normal}
    """

    def __init__(
        self,
        pt2_path,
        refiner_path=None,
        device='cuda',
        input_size=(640, 480),   # (width, height) the graph was compiled for
        refine_steps=3,
    ):
        device = torch.device(device)
        if device.type != 'cuda':
            raise RuntimeError('MoGev3_PT2 requires CUDA: the exported graph is fp16 and the refiner uses Triton kernels.')
        self.device = device
        self.width, self.height = input_size
        self.refine_steps = refine_steps

        # --- Load the exported dense graph ---
        # Note: torch 2.11's ExportedProgram.module() raises on .eval()/train();
        # the unlifted graph module is already inference-only.
        self.graph = load_pt2(pt2_path).to(self.device)

        # --- Load the eager sparse refiner ---
        if refiner_path is None:
            refiner_path = str(pt2_path).replace('.pt2', '_refiner.pt')
        refiner_ckpt = torch.load(refiner_path, map_location='cpu', weights_only=True)
        self.depth_resolution = refiner_ckpt['refiner_depth_resolution']
        self.refiner = Sparse3DUNet(**refiner_ckpt['model_config'])
        missing, unexpected = self.refiner.load_state_dict(refiner_ckpt['model'], strict=False)
        if missing or unexpected:
            raise RuntimeError(f'Refiner state dict mismatch: missing={missing}, unexpected={unexpected}')
        self.refiner = self.refiner.half().to(self.device).eval()

        # Grid geometry derived from the fixed input size and num_tokens=3600.
        aspect_ratio = self.width / self.height
        num_tokens = 3600
        self.base_h = round((num_tokens / aspect_ratio) ** 0.5)
        self.base_w = round((num_tokens * aspect_ratio) ** 0.5)
        self.refine_h, self.refine_w = self.base_h * 16, self.base_w * 16
        self.aspect_ratio = aspect_ratio

        # Sanity-check the graph's expected input shape.
        example = self.graph(torch.zeros(1, 3, self.height, self.width, device=self.device, dtype=torch.float16))
        expected = [None, None, (self.height, self.width, 3), (self.height, self.width), ()]
        self._check_shapes(example, expected)

    @staticmethod
    def _check_shapes(outputs, expected):
        for i, (name, out, exp) in enumerate(zip(OUTPUT_NAMES, outputs, expected)):
            if exp is None:
                continue
            got = tuple(out.shape[1:])
            if got != exp:
                raise RuntimeError(f'Graph output {name} has shape {got}, expected {exp}')

    # ------------------------------------------------------------------
    # Image entry (shared repo-wide decode + fixed-size stretch)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU)."""
        return to_pixel_uint8(to_image_tensor(image))

    @staticmethod
    def _resize_to_fixed(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Stretch a CHW uint8 tensor to exactly (h, w); no-op at that size.

        The graph is exported for a fixed size, so the stretch is
        non-preserving.  Bilinear (``align_corners=False``) is the closest
        tensor-first surrogate of the legacy cv2 ``INTER_AREA`` stretch
        (<=1 LSB at integer decimation; the same convention as Any2Full's
        fixed-size stretch and the eager v3 model's internal bilinear
        resizes).  Kept tensor-first: no numpy bounce at the feed boundary.
        """
        if (x.shape[1], x.shape[2]) == (h, w):
            return x
        return F.interpolate(
            x.float().unsqueeze(0), (h, w), mode="bilinear", align_corners=False
        )[0].round().byte()

    def preprocess(self, image: ImageInput) -> torch.Tensor:
        """ImageInput -> fp16 CUDA tensor (1, 3, H, W) at the fixed size.

        Pixel space is RGB (repo-wide contract) and the feed is in [0, 1]
        (divide by 255, no ImageNet normalization), matching the eager
        MoGe scripts' convention.
        """
        x = self._resize_to_fixed(self._load_image(image), self.height, self.width)
        return x.float().div(255.0).unsqueeze(0).to(device=self.device, dtype=torch.float16)

    @torch.inference_mode()
    def infer(
        self,
        image: ImageInput,
        fov_x=None,
        force_projection: bool = True,
        apply_mask: bool = True,
        refine_steps: int = None,
    ) -> dict:
        """Run inference on a single image (any ``ImageInput``).

        Returns {points, depth, mask, intrinsics, normal} as batched fp32
        CUDA tensors: points/depth/normal (1, H, W, 3|1), mask (1, H, W),
        intrinsics (1, 3, 3). Mirrors `moge/model/v3.py` `infer()`.
        """
        if refine_steps is None:
            refine_steps = self.refine_steps
        x = self.preprocess(image)
        return self.infer_tensor(x, fov_x=fov_x, force_projection=force_projection,
                                 apply_mask=apply_mask, refine_steps=refine_steps)

    @torch.inference_mode()
    def infer_tensor(
        self,
        x: torch.Tensor,   # (1, 3, H, W) fp16 CUDA, already preprocessed
        fov_x=None,
        force_projection: bool = True,
        apply_mask: bool = True,
        refine_steps: int = None,
    ) -> dict:
        """Run inference on a preprocessed input tensor (see `infer`)."""
        if refine_steps is None:
            refine_steps = self.refine_steps
        x = x.to(device=self.device, dtype=torch.float16)
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError(f'Expected input size {(self.width, self.height)}, got {tuple(x.shape[-2:])}')

        raw_coord, refiner_feature, normal, mask, metric_scale = self.graph(x)
        batch_size = x.shape[0]

        # --- Sparse refinement (eager; the refiner is not in the graph) ---
        if refine_steps > 0:
            current_coord = refine_logz(self.refiner, raw_coord, refiner_feature,
                                        self.depth_resolution, refine_steps, self.device)
        else:
            current_coord = raw_coord

        # --- Resize + remap to the affine point map (mirror v3 forward) ---
        current_coord = resize_channel_last(current_coord, self.height, self.width)
        affine_points = remap_points_exp(current_coord).float()

        # --- fp32 post-processing (mirror v3 infer) ---
        mask = mask.float()
        metric_scale = metric_scale.float().reshape(-1)
        normal = normal.float()
        result = affine_to_camera(
            affine_points,
            mask,
            metric_scale,
            aspect_ratio=self.aspect_ratio,
            fov_x=fov_x,
            force_projection=force_projection,
            apply_mask=apply_mask,
        )
        if apply_mask:
            result['normal'] = torch.where(result['mask'][..., None], normal, torch.zeros_like(normal))
        else:
            result['normal'] = normal
        return result

    def infer_file(self, image_path, **kwargs) -> dict:
        """Run inference on an image file path (``infer`` accepts paths too)."""
        return self.infer(image_path, **kwargs)
