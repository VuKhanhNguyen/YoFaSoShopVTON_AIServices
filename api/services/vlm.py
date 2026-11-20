"""VLM Service (Qwen3-VL-4B-Instruct · BF16) cho phân tích thuộc tính catalog phía Admin."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from PIL import Image

LOGGER = logging.getLogger("yofaso.vlm")


class VLMService:
    """Lazy-loaded Qwen3-VL-4B-Instruct ở BF16 gốc (không 4-bit) cho trích xuất thuộc tính Admin."""

    def __init__(self, settings):
        """Store settings; do not import large vision-language packages on init."""
        self.settings = settings
        self.model = None
        self.processor = None
        self._attempted = False

    def load(self):
        """Lazy load Qwen3-VL-4B-Instruct ở BF16 gốc trên CUDA (hoặc FP32 trên CPU)."""
        if self.model is not None or self._attempted:
            return self.model
        self._attempted = True
        if not getattr(self.settings, "enable_vlm", True):
            return None
        model_dir = getattr(self.settings, "vlm_model_dir", None)
        if not model_dir or not Path(model_dir).is_dir():
            return None
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            has_cuda = torch.cuda.is_available()
            if has_cuda:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32

            self.processor = AutoProcessor.from_pretrained(str(model_dir))
            load_kwargs = {"torch_dtype": dtype}
            if has_cuda:
                load_kwargs["attn_implementation"] = "sdpa"
                load_kwargs["device_map"] = "auto"

            try:
                self.model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **load_kwargs)
            except Exception:
                # Fallback nếu GPU/accelerate không nhận device_map hoặc sdpa
                self.model = AutoModelForImageTextToText.from_pretrained(str(model_dir), torch_dtype=dtype)
                if has_cuda:
                    self.model = self.model.to("cuda")

            self.model.eval()
            LOGGER.info("Loaded VLM %s in %s (cuda=%s)", Path(model_dir).name, dtype, has_cuda)
            return self.model
        except Exception as exc:
            # shortcut: graceful degradation when VLM checkpoint or transformers version unavailable
            LOGGER.warning("Could not load VLM model: %s. Skipping VLM attribute extraction.", exc)
            self.model = None
            return None

    def analyze_garment(self, image: Image.Image) -> dict | None:
        """Extract semantic attributes (garmentLength, collar, fabric, pattern, caption) for Admin catalog.

        Strict Dual-Resolution: Resize image to max 384x512 to keep latency ~0.8-1.5s in BF16.
        """
        model = self.load()
        if model is None or self.processor is None:
            return None
        try:
            import torch

            img = image.copy()
            img.thumbnail((384, 512), Image.Resampling.LANCZOS)

            prompt = (
                "You are a fashion catalog expert. Analyze this garment image and reply ONLY with a JSON object:\n"
                "{\n"
                '  "garmentLength": "SHORT" | "MIDI" | "LONG" | "UNKNOWN",\n'
                '  "collar": "string",\n'
                '  "fabric": "string",\n'
                '  "hasProminentText": true | false,\n'
                '  "caption": "concise description of pattern and texture"\n'
                "}\n"
                "Do not include markdown or explanations. Only raw JSON."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt")
            target_device = next(model.parameters()).device
            inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=180, do_sample=False)
            raw = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group(0))
        except Exception as exc:
            LOGGER.warning("VLM analyze_garment failed: %s", exc)
        return None
