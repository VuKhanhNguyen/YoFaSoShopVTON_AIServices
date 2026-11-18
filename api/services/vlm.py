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
                from transformers import BitsAndBytesConfig
                # 4-bit quantization via bitsandbytes to stay within 24GB VRAM
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if device == "cuda" else None
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch.float16,
                    device_map="auto" if device == "cuda" else None,
                    quantization_config=bnb_config,
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
                outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, top_k=None, top_p=None, temperature=None)
            raw = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group(0))
        except Exception as exc:
            LOGGER.warning("VLM analyze_garment failed: %s", exc)
        return None

    def evaluate_tryon_candidate(
        self, person_img: Image.Image, garment_img: Image.Image, result_img: Image.Image,
        reference_images: list[Image.Image] | None = None, catalog_context: str = "",
        *, confirm_short_sleeve: bool = False,
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
            content = [
                {"type": "text", "text": "Image 1 (Person before try-on):"},
                {"type": "image", "image": p_img},
                {"type": "text", "text": "Image 2 (Target garment):"},
                {"type": "image", "image": g_img},
                {"type": "text", "text": "Image 3 (Try-on result to evaluate):"},
                {"type": "image", "image": r_img},
            ]
            images = [p_img, g_img, r_img]
            reference_instruction = ""
            if confirm_short_sleeve and reference_images:
                reference = reference_images[0].convert("RGB")
                reference.thumbnail((384, 512), Image.Resampling.LANCZOS)
                content.extend([
                    {"type": "text", "text": "Image 4 (Worn garment reference; ignore model identity and trousers):"},
                    {"type": "image", "image": reference},
                ])
                images.append(reference)
                reference_instruction = "Compare cuff height against Image 4 (ignore tucked styling, belts, trousers).\n"

            prompt = (
                "Assume garment category and sleeve detection ALREADY passed. Evaluate ONLY visual defects in Image 3 (Try-on result) against Image 2 (Target garment) and Image 1 (Person before try-on):\n"
                "1. Hand/arm integrity (deformed hands, missing forearms, or duplicated sleeves).\n"
                "2. Garment color fidelity against Image 2 (reject dark/gray stains or wrong split colors across a light garment body; ignore studio briefs/shorts visible below an untucked hem).\n"
                "3. Severe seams, mask halos, garbled text bands, hallucinated pockets/trousers on the torso, or unreplaced original shirt (HOWEVER, if Image 2 / Context is a layered blazer, vest, gile, or cardigan, seeing Image 1's inner shirt collar/chest inside the V-neck opening is expected and desirable — do NOT reject).\n"
                "4. NEVER reject because of clothing category or subtype naming (polo, t-shirt, knit top, sweatshirt are all valid).\n"
                f"Context: {catalog_context}.\n"
                f"{reference_instruction}"
                f"{'Check both cuffs explicitly. Set shortSleeveConfirmed to true only if both sleeves end above the elbows and there are no fabric strips below either cuff.' if confirm_short_sleeve else ''}\n"
                "Reply ONLY with a single JSON object:\n"
                "{\n"
                '  "passed": true | false,\n'
                '  "score": float between 0.0 and 1.0,\n'
                '  "shortSleeveConfirmed": true or false,\n'
                '  "reason": "short explanation of any visual defect, or concise pass summary"\n'
                "}\n"
                "Do not include markdown or explanations. Only raw JSON."
            )
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, top_k=None, top_p=None, temperature=None)
            raw = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group(0))
        except Exception as exc:
            LOGGER.warning("VLM evaluate_tryon_candidate failed: %s", exc)
        return None

