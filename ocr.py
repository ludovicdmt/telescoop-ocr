import string

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, LightOnOcrConfig, LightOnOcrForConditionalGeneration, LogitsProcessor, LogitsProcessorList
from transformers.convert_slow_tokenizer import bytes_to_unicode

PROMPTS = {"PaddleOCR-VL": "OCR:"}
MAX_NEW_TOKENS = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
DEFAULT_MODEL = "lightonai/LightOnOCR-3-4B" if DEVICE == "cuda" else "lightonai/LightOnOCR-3-1B"
NUM_THREADS = 4
LIGHTON_MISTRAL3_KEYS = {r"^language_model\.model\.": "model.language_model."}
MARKUP_CHARACTERS = set("<>#|`")
FRENCH_CHARACTERS = set(string.ascii_letters + string.digits + string.punctuation + " \n  " + "àâäçéèêëîïôöùûüÿœæÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸŒÆ«»‘’“”–—…°€") - MARKUP_CHARACTERS
FRENCH_LEAD_BYTES = {0xC2, 0xC3, 0xC5, 0xE2}
FOREIGN_SCRIPT_LEAD_BYTES = set(range(0xCD, 0xE0)) | set(range(0xE3, 0xEE))
BYTE_DECODER = {char: byte for byte, char in bytes_to_unicode().items()}

torch.set_num_threads(NUM_THREADS)


def is_french_text(text):
    return all(char in FRENCH_CHARACTERS for char in text)


def is_french_bytes(raw):
    if any(byte >= 0xC0 and byte not in FRENCH_LEAD_BYTES for byte in raw):
        return False
    return is_french_text(raw.decode("utf-8", errors="ignore"))


def token_bytes(tokenizer, token_id):
    token = tokenizer.convert_ids_to_tokens(token_id)
    if all(char in BYTE_DECODER for char in token):
        return bytes(BYTE_DECODER[char] for char in token)
    return tokenizer.decode([token_id]).encode()


def is_foreign_script(raw):
    if any(byte in FOREIGN_SCRIPT_LEAD_BYTES for byte in raw):
        return True
    return any(char.isalpha() and char not in FRENCH_CHARACTERS for char in raw.decode("utf-8", errors="ignore"))


def classify_tokens(tokenizer, eos_token_ids):
    special = set(tokenizer.added_tokens_decoder)
    allowed, foreign = list(eos_token_ids), []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        raw = token_bytes(tokenizer, token_id)
        if is_french_bytes(raw):
            allowed.append(token_id)
        elif is_foreign_script(raw):
            foreign.append(token_id)
    return torch.tensor(allowed), torch.tensor(foreign)


class FrenchOnly(LogitsProcessor):
    def __init__(self, allowed_ids, foreign_ids):
        self.allowed_ids = allowed_ids
        self.foreign_ids = foreign_ids
        self.chose_foreign_script = False

    def __call__(self, input_ids, scores):
        self.chose_foreign_script |= bool(torch.isin(scores.argmax(-1), self.foreign_ids).any())
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_ids] = 0
        return scores + mask


def load_model(model_id):
    if "LightOnOCR-3-1B" in model_id:
        config = LightOnOcrConfig.from_pretrained(model_id)
        return LightOnOcrForConditionalGeneration.from_pretrained(model_id, config=config, dtype=DTYPE, key_mapping=LIGHTON_MISTRAL3_KEYS)
    return AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPE, trust_remote_code=True)


class Transcriber:
    def __init__(self, model_id=DEFAULT_MODEL):
        self.model_id = model_id
        self.prompt = next((prompt for key, prompt in PROMPTS.items() if key in model_id), "")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = load_model(model_id).to(DEVICE).eval()
        eos = self.model.generation_config.eos_token_id
        allowed_ids, foreign_ids = classify_tokens(self.processor.tokenizer, eos if isinstance(eos, list) else [eos])
        self.allowed_ids, self.foreign_ids = allowed_ids.to(DEVICE), foreign_ids.to(DEVICE)

    @torch.inference_mode()
    def transcribe(self, image):
        content = [{"type": "image", "image": image.convert("RGB")}]
        if self.prompt:
            content.append({"type": "text", "text": self.prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(DEVICE, self.model.dtype) if value.is_floating_point() else value.to(DEVICE) for key, value in inputs.items()}
        french = FrenchOnly(self.allowed_ids, self.foreign_ids)
        output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, logits_processor=LogitsProcessorList([french]))
        if french.chose_foreign_script:
            return ""
        text = self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return "".join(char for char in text if char in FRENCH_CHARACTERS).strip()
