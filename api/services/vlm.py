"""VLM Service (Qwen2-VL-2B / Florence-2) for attribute reasoning and QA gatekeeping."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from PIL import Image

LOGGER = logging.getLogger("yofaso.vlm")


class VLMService:
    """Lazy-loaded VLM inference for garment reasoning and Try-On QA."""

    def __init__(self, settings):
        """Store settings; do not import large vision-language packages on init."""
        self.settings = settings
        self.model = None
        self.processor = None
        self._attempted = False

    def load(self):
        """Lazy load Qwen2-VL model in 4-bit (or fp16); fail silently if missing."""
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
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(str(model_dir))
            try:
                # 4-bit quantization via bitsandbytes to stay within 24GB VRAM
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch.float16,
                    device_map="auto" if device == "cuda" else None,
                    load_in_4bit=(device == "cuda"),
                )
            except Exception:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                )
                if device == "cuda":
                    self.model = self.model.to("cuda")
            self.model.eval()
            return self.model
        except Exception as exc:
            # shortcut: graceful degradation, fallback to candidate_primitives heuristics
            LOGGER.warning("Could not load VLM model: %s. Using heuristic fallback.", exc)
            self.model = None
            return None

    def analyze_garment(self, image: Image.Image) -> dict | None:
        """Extract semantic attributes (garmentLength, fabric, collar, text) from image.

        Strict Dual-Resolution: Resize image to max 384x512 to prevent visual token explosion.
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
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.01)
            raw = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group(0))
        except Exception as exc:
            LOGGER.warning("VLM analyze_garment failed: %s", exc)
        return None

    def evaluate_tryon_candidate(
        self, person_img: Image.Image, garment_img: Image.Image, result_img: Image.Image
    ) -> dict | None:
        """QA Gatekeeper: Evaluate VTON result against original person and garment.

        Strict Dual-Resolution: All 3 images resized to max 384x512 (~900 visual tokens).
        """
        model = self.load()
        if model is None or self.processor is None:
            return None
        try:
            import torch

            p_img = person_img.copy()
            p_img.thumbnail((384, 512), Image.Resampling.LANCZOS)
            g_img = garment_img.copy()
            g_img.thumbnail((384, 512), Image.Resampling.LANCZOS)
            r_img = result_img.copy()
            r_img.thumbnail((384, 512), Image.Resampling.LANCZOS)

            prompt = (
                "You are an AI Virtual Try-On QA evaluator. Check the result image against the original person and garment.\n"
                "Evaluate:\n"
                "1. Hand and finger integrity (are hands deformed or blurred?).\n"
                "2. Natural blending at collar, neck, and waist.\n"
                "3. Garment color and pattern fidelity.\n"
                "Reply ONLY with a single JSON object:\n"
                "{\n"
                '  "passed": true | false,\n'
                '  "score": float between 0.0 and 1.0,\n'
                '  "reason": "short explanation"\n'
                "}\n"
                "Do not include markdown or explanations. Only raw JSON."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": p_img},
                        {"type": "image", "image": g_img},
                        {"type": "image", "image": r_img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[p_img, g_img, r_img], padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.01)
            raw = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group(0))
        except Exception as exc:
            LOGGER.warning("VLM evaluate_tryon_candidate failed: %s", exc)
        return None

