"""Emotion recognition — r-f/wav2vec-english-speech-emotion-recognition + AMD GPU (DirectML)."""
from pathlib import Path
from typing import Dict
import logging, numpy as np

_model = None
_extractor = None
_device = None
_id2label = None

MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
SR = 16000

LABEL_VI = {
    "angry": "Tức giận", "disgust": "Ghê tởm", "fear": "Sợ hãi",
    "happy": "Vui vẻ", "neutral": "Bình thường", "sad": "Buồn", "surprise": "Ngạc nhiên",
}


def _get_device():
    """Get best available device: DirectML (AMD GPU) > CUDA > CPU."""
    import torch
    try:
        import torch_directml
        dev = torch_directml.device()
        logging.info("Using DirectML (AMD GPU): %s", dev)
        return dev
    except ImportError:
        pass
    if torch.cuda.is_available():
        logging.info("Using CUDA GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logging.info("Using CPU")
    return torch.device("cpu")


def _load_model():
    global _model, _extractor, _device, _id2label
    if _model is None:
        import torch, torch.nn as nn
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoConfig
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.info("Loading emotion model: %s ...", MODEL_NAME)

        config = AutoConfig.from_pretrained(MODEL_NAME)
        _id2label = config.id2label
        num_labels = len(_id2label)
        hidden = config.hidden_size

        _extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

        class EmotionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, num_labels),
                )

            def forward(self, input_values):
                out = self.wav2vec2(input_values).last_hidden_state
                pooled = out.mean(dim=1)
                return self.classifier(pooled)

        _model = EmotionModel()

        sf_path = hf_hub_download(MODEL_NAME, "model.safetensors", revision="refs/pr/6")
        state = load_file(sf_path)

        base_state = {}
        cls_state = {}
        for k, v in state.items():
            if k.startswith("classifier.dense."):
                cls_state[k.replace("classifier.dense.", "classifier.0.")] = v
            elif k.startswith("classifier.out_proj."):
                cls_state[k.replace("classifier.out_proj.", "classifier.2.")] = v
            elif k.startswith("wav2vec2."):
                base_state[k] = v
            else:
                base_state[k] = v

        _model.load_state_dict({**base_state, **cls_state}, strict=False)
        _device = _get_device()
        _model = _model.to(_device)
        _model.eval()
        logging.info("Emotion model loaded on %s. Labels: %s", _device, list(_id2label.values()))
    return _model, _extractor, _device, _id2label


def predict_emotion_from_array(y: np.ndarray, sr: int = SR) -> Dict:
    """Predict emotion from numpy audio array using GPU."""
    import torch
    model, extractor, device, id2label = _load_model()
    inputs = extractor(y.astype(np.float32), sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    scores = []
    for i, p in enumerate(probs):
        label = id2label[i]
        scores.append({"label": label, "label_vi": LABEL_VI.get(label, label), "score": round(float(p), 4)})
    scores.sort(key=lambda x: x["score"], reverse=True)
    top = scores[0]
    return {"emotion": top["label"], "emotion_vi": top["label_vi"],
            "confidence": top["score"], "all_scores": scores,
            "device": str(device)}


def predict_emotion_from_file(audio_path: Path) -> Dict:
    """Predict emotion from audio file."""
    import librosa
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    return predict_emotion_from_array(y)
