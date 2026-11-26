"""Original-mode model service for I8.6b; no database writes or training."""
import hashlib
import json
import math
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = 'distilbert_food_nova'
MAX_TOKENS = 256
MAX_CHARS = 4000
LABELS = [f'NOVA {n}' for n in range(1, 5)]
FILES = {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'}


class InputError(ValueError):
    """Safe, client-facing input error."""


def verify_policy():
    policy = json.loads((ROOT / 'docs/NOVA_INFERENCE_POLICY.json').read_text())
    default = policy['default']
    if (policy['model'] != MODEL_NAME or policy['labels'] != [1, 2, 3, 4]
            or (default['mode'], default['alpha'], default['temperature']) != ('original', 0, 1)
            or set(policy['model_files_sha256']) != FILES):
        raise RuntimeError('Unexpected inference policy; no model loaded')
    for name, expected in policy['model_files_sha256'].items():
        with (ROOT / 'data/deep_learning/models' / MODEL_NAME / name).open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError(f'Model fingerprint mismatch: {name}')
    return policy['version']


def validate_text(text, source, reviewed):
    if source not in ('manual', 'ocr'):
        raise InputError('Unsupported ingredient source')
    if source == 'ocr' and reviewed is not True:
        raise InputError('OCR text requires user review before prediction')
    if not isinstance(text, str) or len(text) > MAX_CHARS:
        raise InputError('Ingredient text must be a string of at most 4000 characters')
    text = text.strip()
    if not any(c.isalpha() for c in text):
        raise InputError('Enter readable ingredient text containing letters')
    if any(ord(c) < 32 and c not in '\t\n\r' for c in text) or '\x7f' in text:
        raise InputError('Ingredient text contains unsupported control characters')
    return text


class NovaService:
    def __init__(self):
        self.policy_version = verify_policy()
        from src.deep_learning.inference.predict import load_model, predict_nova
        self.tokenizer, self.model = load_model()
        self.predict_nova = predict_nova
        self.lock = Lock()
        self.device = str(self.model.device)

    def predict(self, text, source, reviewed):
        text = validate_text(text, source, reviewed)
        # Serialize tokenizer + model calls for initial local GPU use.
        with self.lock:
            tokens = self.tokenizer(text, add_special_tokens=True, truncation=False,
                                    verbose=False)['input_ids']
            if len(tokens) > MAX_TOKENS:
                raise InputError('Ingredient list exceeds 256 model tokens; shorten or review it')
            result = self.predict_nova(text, self.tokenizer, self.model)
        scores = result['class_probabilities']
        if (set(scores) != set(LABELS) or any(not math.isfinite(v) or not 0 <= v <= 100 for v in scores.values())
                or abs(sum(scores.values()) - 100) > 0.03
                or result['predicted_nova'] not in (1, 2, 3, 4)
                or result['label'] != LABELS[result['predicted_nova'] - 1]
                or result['confidence'] != scores[result['label']]
                or result['confidence'] != max(scores.values())):
            raise RuntimeError('Unexpected model output')
        return {'status': 'predicted', **result, 'prediction_mode': 'original',
                'model': MODEL_NAME, 'policy_version': self.policy_version,
                'input_source': source, 'token_count': len(tokens),
                'confidence_note': 'Model score, not a guarantee of correctness; not a health rating.'}


def self_test():
    assert validate_text(' Milk ', 'manual', False) == 'Milk'
    assert validate_text('Milk', 'ocr', True) == 'Milk'
    for text, source, reviewed in [('', 'manual', False), ('  ', 'manual', False),
            ('=', 'manual', False), ('1234', 'manual', False), ('Milk\x00', 'manual', False),
            ('Milk', 'ocr', False), ('a' * 4001, 'manual', False)]:
        try:
            validate_text(text, source, reviewed)
        except InputError:
            continue
        raise AssertionError('Invalid input accepted')
    print('SUCCESS: basic ingredient validation self-tests passed (no model loaded).')


if __name__ == '__main__':
    self_test()
