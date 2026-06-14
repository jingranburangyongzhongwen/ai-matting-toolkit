"""
ViTMatte 引擎：基于 Trimap 的 alpha matte 精细化
配合 RMBG-2.0 使用，从软 mask 自动生成 trimap，无需手动调参
"""
import os
import time
import threading

import cv2
import numpy as np
import torch
from PIL import Image

from log import get_logger
logger = get_logger(__name__)


class ViTMatteRefiner:
    HF_REPO = "hustvl/vitmatte-small-distinctions-646"

    def __init__(self, model_path: str, device: str = "cpu", hf_repo: str = None,
                 is_matany: bool = False):
        self.device = device
        self.model_path = model_path
        self.hf_repo = hf_repo or self.HF_REPO
        self.model = None
        self.processor = None
        self.is_matany = is_matany
        self._trimap_erode = 1
        self._trimap_dilate = 3
        self._trimap_kernel = 7
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load_model(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            local_path = self.model_path
            safetensors = os.path.join(local_path, "model.safetensors")
            d2_ckpt = os.path.join(local_path, "ViTMatte_B_DIS.pth")

            if os.path.isfile(safetensors):
                self.processor = VitMatteImageProcessor.from_pretrained(local_path)
                self.model = VitMatteForImageMatting.from_pretrained(local_path)
            elif os.path.isfile(d2_ckpt):
                # MatAny: detectron2 权重加载到 transformers 模型骨架
                logger.info("从 detectron2 加载: %s", d2_ckpt)
                base_repo = "hustvl/vitmatte-base-distinctions-646"
                self.model = VitMatteForImageMatting.from_pretrained(base_repo)
                self._load_detectron2_weights(d2_ckpt)
                self.model.save_pretrained(local_path)
                # processor 配置跟 Base 一样，复制过来（save_pretrained 不保存 processor）
                import shutil
                base_local = os.path.join(os.path.dirname(local_path), "vitmatte-base")
                proc_src = os.path.join(base_local, "preprocessor_config.json")
                if os.path.isfile(proc_src):
                    shutil.copy2(proc_src, os.path.join(local_path, "preprocessor_config.json"))
                self.processor = VitMatteImageProcessor.from_pretrained(local_path)
                logger.info("已缓存到: %s", local_path)
            else:
                local_path = self._ensure_local(local_path)
                self.processor = VitMatteImageProcessor.from_pretrained(local_path)
                self.model = VitMatteForImageMatting.from_pretrained(local_path)

            self._patch_attention()
            self.model.float().to(self.device).eval()
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info("VRAM ViTMatte loaded — allocated: %.2fGB, reserved: %.2fGB", allocated, reserved)
            logger.info("模型加载完成")


    def _load_detectron2_weights(self, ckpt_path: str):
        """加载 detectron2 backbone 权重到 transformers 模型"""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_sd = ckpt["model"] if "model" in ckpt else ckpt
        dst_sd = self.model.state_dict()

        suffix_map = {
            "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
            "attn.qkv.weight": "attention.qkv.weight", "attn.qkv.bias": "attention.qkv.bias",
            "attn.proj.weight": "attention.proj.weight", "attn.proj.bias": "attention.proj.bias",
            "attn.rel_pos_h": "attention.rel_pos_h", "attn.rel_pos_w": "attention.rel_pos_w",
            "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
            "mlp.fc1.weight": "mlp.fc1.weight", "mlp.fc1.bias": "mlp.fc1.bias",
            "mlp.fc2.weight": "mlp.fc2.weight", "mlp.fc2.bias": "mlp.fc2.bias",
        }
        top_map = {
            "patch_embed.proj.weight": "backbone.embeddings.projection.weight",
            "patch_embed.proj.bias": "backbone.embeddings.projection.bias",
            "pos_embed": "backbone.embeddings.position_embeddings",
        }

        converted = 0
        for src_key, src_val in src_sd.items():
            # 去掉 backbone. 前缀
            key = src_key
            if key.startswith("backbone."):
                key = key[9:]

            if key in top_map:
                dst_key = top_map[key]
            elif key.startswith("blocks."):
                rest = key[len("blocks."):]
                idx, attr = rest.split(".", 1)
                new_attr = suffix_map.get(attr)
                if not new_attr:
                    continue
                dst_key = f"backbone.encoder.layer.{idx}.{new_attr}"
            else:
                continue

            if dst_key not in dst_sd:
                continue
            if src_val.shape == dst_sd[dst_key].shape:
                dst_sd[dst_key] = src_val
                converted += 1
            elif "position_embeddings" in dst_key and src_val.shape[1] > dst_sd[dst_key].shape[1]:
                dst_sd[dst_key] = src_val[:, 1:1 + dst_sd[dst_key].shape[1]]
                converted += 1

        self.model.load_state_dict(dst_sd)
        logger.info("从 detectron2 加载了 %d 个 backbone 权重", converted)

    def _patch_attention(self):
        """为 ViTMatte 注册 attention 优化。
        Small: 纯 strided（降显存）。
        MatAny: window + strided（对齐训练时的混合 attention）。
        Base: 不做优化（全 attention，质量最好但大图可能 OOM）。"""
        from transformers.models.vitdet.modeling_vitdet import add_decomposed_relative_positions

        WINDOW_BLOCKS = {0, 1, 3, 4, 6, 7, 9, 10}

        hidden_dim = 0
        for module in self.model.modules():
            if hasattr(module, 'qkv') and hasattr(module, 'scale'):
                hidden_dim = module.qkv.in_features
                break
        is_small = hidden_dim < 768

        # Small: 纯 strided
        # MatAny (= Base + window+strided): 对齐 Matte-Anything 的混合 attention
        # Base (非 MatAny): 全 attention，不 patch
        do_strided = is_small or self.is_matany
        do_window = self.is_matany

        def strided_forward(self, hidden_state, output_attentions=False):
            batch_size, height, width, _ = hidden_state.shape
            qkv = self.qkv(hidden_state).reshape(
                batch_size, height * width, 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
            queries, keys, values = qkv.reshape(
                3, batch_size * self.num_heads, height * width, -1
            ).unbind(0)
            dim = queries.shape[-1]
            queries = queries.view(batch_size * self.num_heads, height, width, dim)
            keys = keys.view(batch_size * self.num_heads, height, width, dim)
            values = values.view(batch_size * self.num_heads, height, width, dim)
            out_full = torch.zeros(batch_size * self.num_heads, height, width, dim,
                                   device=queries.device, dtype=queries.dtype)
            for sh in range(2):
                for sw in range(2):
                    q_sub = queries[:, sh::2, sw::2].reshape(batch_size * self.num_heads, -1, dim)
                    k_sub = keys[:, sh::2, sw::2].reshape(batch_size * self.num_heads, -1, dim)
                    v_sub = values[:, sh::2, sw::2].reshape(batch_size * self.num_heads, -1, dim)
                    h_sub, w_sub = (height + 1 - sh) // 2, (width + 1 - sw) // 2
                    attn = (q_sub * self.scale) @ k_sub.transpose(-2, -1)
                    if self.use_relative_position_embeddings:
                        attn = add_decomposed_relative_positions(
                            attn, q_sub, self.rel_pos_h, self.rel_pos_w, (h_sub, w_sub), (h_sub, w_sub))
                    attn = attn.softmax(dim=-1)
                    out_full[:, sh::2, sw::2] = (attn @ v_sub).reshape(
                        batch_size * self.num_heads, h_sub, w_sub, dim)
                    del attn, q_sub, k_sub, v_sub
                    torch.cuda.empty_cache()
            hidden_state = out_full.view(batch_size, self.num_heads, height, width, dim)
            hidden_state = hidden_state.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
            return (self.proj(hidden_state),)

        # ---- 注册 patch ----
        encoder = None
        for module in self.model.modules():
            if hasattr(module, 'layer') and isinstance(module.layer, torch.nn.ModuleList):
                encoder = module
                break

        strided_count, window_count = 0, 0
        if encoder is not None and (do_strided or do_window):
            for idx, layer in enumerate(encoder.layer):
                attn = layer.attention if hasattr(layer, 'attention') else None
                if attn is None or not hasattr(attn, 'qkv'):
                    continue
                if do_strided:
                    attn._original_forward = attn.forward
                    attn._strided_forward = lambda hs, output_attentions=False, _m=attn: strided_forward(_m, hs, output_attentions)
                if do_window and idx in WINDOW_BLOCKS:
                    layer._original_window_size = layer.window_size
                    layer.window_size = 14
                    window_count += 1
                else:
                    strided_count += 1

        logger.info("attention=native(window+global)")

    def _set_strided(self, enabled: bool):
        """切换优化 attention / 全 attention。

        当前所有调用点均传入 enabled=False（始终使用原生 window+global attention），
        以保证最高质量。strided forward 已在 _patch_attention 中注册，可作为
        大分辨率场景下的显存优化后备方案启用。
        """
        count = 0
        # strided attention blocks
        for module in self.model.modules():
            if hasattr(module, '_strided_forward') and hasattr(module, '_original_forward'):
                module.forward = module._strided_forward if enabled else module._original_forward
                count += 1
        # window attention blocks (MatAny only)
        encoder = None
        for module in self.model.modules():
            if hasattr(module, 'layer') and isinstance(module.layer, torch.nn.ModuleList):
                encoder = module
                break
        if encoder is not None:
            for layer in encoder.layer:
                if hasattr(layer, '_original_window_size'):
                    if enabled:
                        layer.window_size = 14
                    else:
                        layer.window_size = layer._original_window_size
                    count += 1

    def _ensure_local(self, path: str) -> str:
        """本地路径有完整模型就直接用，否则从 HuggingFace 下载到该路径"""
        weights_exist = (
            os.path.isfile(os.path.join(path, "model.safetensors")) or
            os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        )
        if weights_exist:
            return path
        if not self.hf_repo:
            raise FileNotFoundError(f"本地模型不存在: {path}，且无 HuggingFace 源可下载")
        logger.info("本地模型不完整，下载 %s 到 %s ...", self.hf_repo, path)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=self.hf_repo,
            local_dir=path,
        )
        logger.info("下载完成")
        return path

    def refine(self, image, alpha: np.ndarray, transparent_detector=None,
               transparent_boxes=None, soft: bool = False, mode: str = "auto",
               _debug_dir: str = None) -> np.ndarray:
        """
        用 ViTMatte 精细化 alpha matte
        Args:
            image: 原图，PIL.Image 或 HxWx3 RGB numpy array
            alpha: 粗 alpha，HxW uint8 [0,255]，软 / 硬 mask 都接受
            transparent_detector: 可选 GroundingDinoDetector，内部检测透明物体并修正 trimap
            transparent_boxes: 可选已检测好的透明物体框 [[x1,y1,x2,y2], ...]，
                  传了就直接用、跳过内部检测（便于调用方把检测单独作为一个阶段展示）
            soft: alpha 是否为软概率图（RMBG-2.0）。True 时额外把模型自己的过渡区
                  并入 unknown；False（SAM 二值 mask）只用形态学窄带
            mode: "strip"（条带）/ "subject"（主体crop）/ "full"（边缘crop）
        Returns:
            精细化后的 HxW uint8 alpha [0,255]
        """
        with self._infer_lock:
            return self._refine_locked(
                image,
                alpha,
                transparent_detector=transparent_detector,
                transparent_boxes=transparent_boxes,
                soft=soft,
                mode=mode,
                _debug_dir=_debug_dir,
            )

    def _refine_locked(self, image, alpha: np.ndarray, transparent_detector=None,
                       transparent_boxes=None, soft: bool = False,
                       mode: str = "auto", _debug_dir: str = None) -> np.ndarray:
        self._load_model()
        # 始终跑模型原生注意力（window+global），不用降质的 strided 省显存近似。
        self._set_strided(enabled=False)

        # RMBG already provides a soft alpha. Keep the unknown band narrow so
        # ViTMatte refines the edge instead of hallucinating a broad matte.
        # erode=3, dilate=5, kernel=9 → unknown band ~72px，足以处理发丝。
        # conservative merge + _guard_against_overcut 保护确定前景区域。
        if soft:
            trimap_erode, trimap_dilate, trimap_kernel = 3, 5, 9
        else:
            trimap_erode, trimap_dilate, trimap_kernel = 2, 4, 7

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img_np = np.asarray(image.convert("RGB"))
        full_h, full_w = img_np.shape[:2]

        trimap = self._make_trimap(alpha, soft=soft,
                                    erode=trimap_erode,
                                    dilate=trimap_dilate,
                                    kernel_size=trimap_kernel)
        boxes = transparent_boxes
        if boxes is None and transparent_detector is not None:
            boxes = transparent_detector.detect(image)
        if boxes:
            logger.info("检测到 %d 个透明物体，修正 trimap", len(boxes))
            trimap = self._mark_transparent(trimap, boxes)

        if _debug_dir:
            os.makedirs(_debug_dir, exist_ok=True)
            self._dump_trimap(trimap)
            Image.fromarray(trimap, "L").save(os.path.join(_debug_dir, "20_vitmatte_trimap.png"))

        roi_ys, roi_xs = np.where(trimap > 0)
        if roi_ys.size == 0:
            return np.zeros((full_h, full_w), dtype=np.uint8)

        unk_ys, unk_xs = np.where(trimap == 127)
        if unk_ys.size == 0:
            return (alpha > 127).astype(np.uint8) * 255

        if mode == "full":
            result = self._refine_crop(img_np, trimap, alpha, full_h, full_w,
                                       roi_ys, roi_xs, margin_div=50, tag_prefix="边缘crop",
                                       conservative=soft)
        elif mode == "subject":
            fg_ys, fg_xs = np.where(alpha > 127)
            if fg_ys.size == 0:
                return np.zeros((full_h, full_w), dtype=np.uint8)
            result = self._refine_crop(img_np, trimap, alpha, full_h, full_w,
                                       fg_ys, fg_xs, margin_div=25, tag_prefix="主体crop",
                                       conservative=soft)
        else:
            result = self._refine_strip(img_np, trimap, alpha, full_h, full_w,
                                        unk_ys, unk_xs, conservative=soft)

        if _debug_dir:
            h, w = result.shape
            total = h * w
            bg = np.sum(result < 10) / total * 100
            fg = np.sum(result > 245) / total * 100
            edge = np.sum((result >= 10) & (result <= 245)) / total * 100
            edge_std = result[(result >= 10) & (result <= 245)].std() if np.any((result >= 10) & (result <= 245)) else 0
            logger.debug("ViTMatte输出: bg=%.1f%% fg=%.1f%% edge=%.1f%% edge_std=%.1f", bg, fg, edge, edge_std)
            Image.fromarray(result, "L").save(os.path.join(_debug_dir, "21_vitmatte_alpha.png"))

        return result

    @staticmethod
    def _merge_refined_unknown(alpha_crop: np.ndarray, trimap_crop: np.ndarray,
                               refined: np.ndarray, weight: np.ndarray = None,
                               conservative: bool = False) -> np.ndarray:
        """
        Only replace trimap unknown pixels with ViTMatte output.
        Known foreground/background must stay constrained, otherwise full-frame
        predictions can leak faint alpha into sure-background areas.
        """
        merged = alpha_crop.astype(np.float32).copy()
        refined_u8 = np.clip(refined * 255.0, 0, 255).astype(np.float32)
        unknown = trimap_crop == 127
        if not np.any(unknown):
            return alpha_crop.copy()

        orig = alpha_crop.astype(np.float32)
        if weight is None:
            merge_weight = np.ones_like(orig, dtype=np.float32)
        else:
            merge_weight = weight.astype(np.float32)

        if conservative:
            not_solid = np.clip((252.0 - orig) / 52.0, 0.0, 1.0)
            merge_weight *= not_solid

        blended = orig * (1.0 - merge_weight) + refined_u8 * merge_weight
        merged[unknown] = blended[unknown]

        merged[trimap_crop == 0] = 0.0
        fg = trimap_crop == 255
        merged[fg] = np.maximum(merged[fg], alpha_crop[fg].astype(np.float32))
        return np.clip(merged, 0, 255).astype(np.uint8)

    @staticmethod
    def _merge_accept_region(
        alpha_crop: np.ndarray,
        trimap_crop: np.ndarray,
        refined: np.ndarray,
        accept: np.ndarray,
    ) -> np.ndarray:
        """Merge ViTMatte output in user-accepted region without foreground clamping."""
        refined_u8 = np.clip(refined * 255.0, 0, 255).astype(np.float32)
        orig = alpha_crop.astype(np.float32)
        result = orig.copy()
        # Manual repair marks accept as editable; do not max-clamp known FG here.
        result[accept] = refined_u8[accept]
        return np.clip(result, 0, 255).astype(np.uint8)

    def _refine_crop(self, img_np, trimap, alpha, full_h, full_w,
                     roi_ys, roi_xs, margin_div=50, tag_prefix="crop",
                     conservative: bool = False):
        """矩形 crop 推理：crop 到指定区域 + 边距，然后全图推理"""
        margin = max(32, min(full_h, full_w) // margin_div)
        x1 = max(0, int(roi_xs.min()) - margin)
        y1 = max(0, int(roi_ys.min()) - margin)
        x2 = min(full_w, int(roi_xs.max()) + 1 + margin)
        y2 = min(full_h, int(roi_ys.max()) + 1 + margin)
        crop_img = img_np[y1:y2, x1:x2]
        crop_tri = trimap[y1:y2, x1:x2]
        crop_h, crop_w = crop_tri.shape

        refined = self._run_vitmatte(crop_img, crop_tri, crop_h, crop_w,
                                     f"{tag_prefix} {crop_w}x{crop_h}", full_w, full_h)

        full_alpha = alpha.copy()
        full_alpha[y1:y2, x1:x2] = self._merge_refined_unknown(
            alpha[y1:y2, x1:x2],
            crop_tri,
            refined,
            conservative=conservative,
        )
        return full_alpha

    def _refine_strip(self, img_np, trimap, alpha, full_h, full_w,
                      unk_ys, unk_xs, conservative: bool = False):
        """条带推理：只处理 unknown 区域 + 上下文，省显存。边界渐变混合消除接缝。"""
        STRIP_HALF = 384
        sx1 = max(0, int(unk_xs.min()) - STRIP_HALF)
        sy1 = max(0, int(unk_ys.min()) - STRIP_HALF)
        sx2 = min(full_w, int(unk_xs.max()) + 1 + STRIP_HALF)
        sy2 = min(full_h, int(unk_ys.max()) + 1 + STRIP_HALF)
        strip_img = img_np[sy1:sy2, sx1:sx2]
        strip_tri = trimap[sy1:sy2, sx1:sx2]
        strip_h, strip_w = strip_img.shape[:2]

        refined = self._run_vitmatte(strip_img, strip_tri, strip_h, strip_w,
                                     f"条带 {strip_w}x{strip_h}", full_w, full_h)

        # 边界渐变混合：在条带边缘用 Gaussian 权重与 RMBG alpha 混合，
        # 中心区域完全信任 ViTMatte，边界处快速过渡，消除接缝。
        # 参考 MONAI/nnU-Net 行业标准的 Gaussian 加权滑窗方案。
        BLEND = 48
        _t = np.linspace(-3.0, 3.0, BLEND)
        _sigma = BLEND / 6.0
        _gaussian = np.exp(-0.5 * (_t / _sigma) ** 2)
        _gaussian /= _gaussian.max()
        weight = np.ones((strip_h, strip_w), dtype=np.float32)
        if sy1 > 0:
            weight[:BLEND] *= _gaussian[:, np.newaxis]
        if sy2 < full_h:
            weight[-BLEND:] *= _gaussian[::-1][:, np.newaxis]
        if sx1 > 0:
            weight[:, :BLEND] *= _gaussian[np.newaxis, :]
        if sx2 < full_w:
            weight[:, -BLEND:] *= _gaussian[::-1][np.newaxis, :]

        full_alpha = alpha.copy()
        full_alpha[sy1:sy2, sx1:sx2] = self._merge_refined_unknown(
            alpha[sy1:sy2, sx1:sx2],
            strip_tri,
            refined,
            weight=weight,
            conservative=conservative,
        )
        return full_alpha

    def _run_vitmatte(self, infer_img, infer_tri, infer_h, infer_w,
                      tag, full_w, full_h):
        """执行 ViTMatte 推理并返回 refined alpha"""
        logger.debug("tile=%dx%d", infer_w, infer_h)
        t0 = time.perf_counter()
        inputs = self.processor(
            images=Image.fromarray(infer_img),
            trimaps=Image.fromarray(infer_tri, mode="L"),
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logger.info("预处理耗时 %.2fs, 全图 %dx%d → %s", time.perf_counter() - t0, full_w, full_h, tag)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        logger.info("推理耗时 %.2fs", time.perf_counter() - t1)

        refined = outputs.alphas[0, 0].float().cpu().numpy()[:infer_h, :infer_w]
        return np.clip(refined, 0, 1)

    def refine_with_trimap(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        trimap: np.ndarray,
        accept_mask: np.ndarray,
        mode: str = "full",
        _debug_dir: str = None,
    ) -> np.ndarray:
        """Run ViTMatte on an externally provided trimap, merge only in accept_mask."""
        with self._infer_lock:
            return self._refine_with_trimap_locked(
                image, alpha, trimap, accept_mask, mode=mode, _debug_dir=_debug_dir,
            )

    def _refine_with_trimap_locked(
        self, image: np.ndarray, alpha: np.ndarray, trimap: np.ndarray,
        accept_mask: np.ndarray, mode: str = "full", _debug_dir: str = None,
    ) -> np.ndarray:
        self._load_model()
        self._set_strided(enabled=False)

        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            img_np = image
        else:
            img_np = np.asarray(Image.fromarray(image).convert("RGB"))
        full_h, full_w = img_np.shape[:2]

        # ROI crop from accept_mask with 128px padding
        ys, xs = np.where(accept_mask)
        if ys.size == 0:
            return alpha.copy()
        pad = 128
        y_start = max(0, int(ys.min()) - pad)
        y_end = min(full_h, int(ys.max()) + 1 + pad)
        x_start = max(0, int(xs.min()) - pad)
        x_end = min(full_w, int(xs.max()) + 1 + pad)

        roi_img = img_np[y_start:y_end, x_start:x_end]
        roi_tri = trimap[y_start:y_end, x_start:x_end]
        roi_h, roi_w = roi_img.shape[:2]

        unk_ys, unk_xs = np.where(roi_tri == 127)
        if unk_ys.size == 0:
            return alpha.copy()

        refined = self._run_vitmatte(
            roi_img, roi_tri, roi_h, roi_w,
            f"manual-refine {roi_w}x{roi_h}", full_w, full_h,
        )

        # Merge only in accept_mask region, conservative=False (user confirmed area)
        roi_alpha = alpha[y_start:y_end, x_start:x_end]
        roi_trimap = trimap[y_start:y_end, x_start:x_end]
        roi_accept = accept_mask[y_start:y_end, x_start:x_end]

        # Merge ViTMatte output across entire accept_mask (not just unknown band)
        roi_merged = self._merge_accept_region(
            roi_alpha, roi_trimap, refined, roi_accept,
        )

        # Only write back where accept_mask is True
        result = alpha.copy()
        result[y_start:y_end, x_start:x_end] = np.where(
            roi_accept, roi_merged, roi_alpha,
        )

        if _debug_dir:
            os.makedirs(_debug_dir, exist_ok=True)
            Image.fromarray(result, "L").save(
                os.path.join(_debug_dir, "manual_vitmatte_alpha.png"),
            )

        return result

    @staticmethod
    def _make_trimap(alpha: np.ndarray, soft: bool = False,
                     erode: int = 1, dilate: int = 2,
                     kernel_size: int = 5) -> np.ndarray:
        """
        造 trimap：RMBG soft alpha 用窄 unknown band，避免 ViTMatte 生成宽鬼影。
        soft=True 时 RMBG 过渡区 (0.05<a<0.95) 补充细结构。
        """
        bin_mask = (alpha > 127).astype(np.uint8) * 255
        kernel_size = max(3, int(kernel_size))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(bin_mask, kernel, iterations=erode)
        dilated = cv2.dilate(bin_mask, kernel, iterations=dilate)

        trimap = np.zeros_like(bin_mask)
        trimap[dilated == 255] = 127
        trimap[eroded == 255] = 255

        if soft:
            a = alpha.astype(np.float32) / 255.0
            transition = (a > 0.05) & (a < 0.95)
            trimap[transition] = 127

        # Trimap 质量检查：unknown band 过窄或过宽时记录警告
        total = trimap.size
        unknown_ratio = np.sum(trimap == 127) / total
        fg_ratio = np.sum(trimap == 255) / total
        if unknown_ratio < 0.005:
            logger.warning("Trimap unknown band < 0.5%% (%.2f%%)，ViTMatte 可能缺乏预测空间",
                           unknown_ratio * 100)
        elif unknown_ratio > 0.50:
            logger.warning("Trimap unknown band > 50%% (%.2f%%)，ViTMatte 可能产生幻觉",
                           unknown_ratio * 100)
        if fg_ratio < 0.001:
            logger.warning("Trimap 确定前景 < 0.1%% (%.3f%%)，可能无法约束预测", fg_ratio * 100)

        return trimap

    @staticmethod
    def _mark_transparent(trimap: np.ndarray, boxes) -> np.ndarray:
        """对检测到的透明物体框，把框内 foreground 像素改为 unknown，让 ViTMatte 推断 alpha"""
        if not boxes:
            return trimap
        result = trimap.copy()
        h, w = result.shape
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            region = result[y1:y2, x1:x2]
            region[region == 255] = 127
        return result

    @staticmethod
    def _dump_trimap(trimap: np.ndarray):
        total = trimap.size
        bg = np.sum(trimap == 0) / total * 100
        unk = np.sum(trimap == 127) / total * 100
        fg = np.sum(trimap == 255) / total * 100
        n, _, stats, _ = cv2.connectedComponentsWithStats((trimap == 127).astype(np.uint8))
        min_area = stats[1:, cv2.CC_STAT_AREA].min() if n > 1 else 0
        max_area = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
        logger.debug("ViTMatte实际Trimap: bg=%.1f%% unknown=%.1f%% fg=%.1f%% | "
                     "unknown连通域=%d个 | unknown面积: min=%d max=%d",
                     bg, unk, fg, n - 1, min_area, max_area)

    def cleanup(self):
        with self._infer_lock, self._load_lock:
            if self.model is not None:
                del self.model
                del self.processor
                self.model = None
                self.processor = None
