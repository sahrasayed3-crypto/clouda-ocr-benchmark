from time import perf_counter

from .base import InferenceRequest, InferenceResult
from .chat_vl import ChatVLAdapter


class DotsMOCRAdapter(ChatVLAdapter):
    name = "dots_mocr"
    model_class = "AutoModelForCausalLM"
    trust_remote_code = True

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Follow the repository's qwen_vl_utils-based Hugging Face example."""
        import torch
        from qwen_vl_utils import process_vision_info

        if self.model is None or self.processor is None:
            raise RuntimeError("adapter is not loaded")
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(request.image_path)},
                {"type": "text", "text": request.prompt},
            ]}]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self.model.device)
            preprocess = perf_counter() - started
            generation = {"max_new_tokens": 24000, "do_sample": False, **request.generation}
            inference_started = perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**inputs, **generation)
            inference = perf_counter() - inference_started
            post_started = perf_counter()
            generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
            text = self.processor.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            post = perf_counter() - post_started
            return InferenceResult(text, text, inference, preprocess, post, self._peak_vram(torch))
        except Exception as exc:
            return InferenceResult(error=f"{type(exc).__name__}: {exc}")
