"""Phân loại tiếng ho: ho khan (dry), ho có đờm (wet), không phải ho (none).

Features: MFCC, spectral centroid/bandwidth/rolloff, ZCR, RMS, duration.
Model: RandomForest trained on extracted features.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import joblib, librosa, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

SR = 16000
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
COUGH_MODEL_PATH = MODELS_DIR / "cough_classifier.joblib"

LABEL_VI = {"dry_cough": "Ho khan", "wet_cough": "Ho có đờm", "not_cough": "Không phải ho"}


def _extract_cough_features(y: np.ndarray, sr: int = SR) -> Dict[str, float]:
    """Extract features optimized for cough detection."""
    _m = lambda v: float(np.mean(v))
    _s = lambda v: float(np.std(v))

    # Trim silence
    yt, _ = librosa.effects.trim(y, top_db=20)
    if len(yt) < sr * 0.05:
        yt = y

    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=20)
    cent = librosa.feature.spectral_centroid(y=yt, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=yt, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=yt, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(yt)
    rms = librosa.feature.rms(y=yt)
    flatness = librosa.feature.spectral_flatness(y=yt)

    feats = {}
    for i in range(13):
        feats[f"mfcc_{i}_mean"] = _m(mfcc[i])
        feats[f"mfcc_{i}_std"] = _s(mfcc[i])
    feats["centroid_mean"] = _m(cent)
    feats["centroid_std"] = _s(cent)
    feats["bandwidth_mean"] = _m(bw)
    feats["rolloff_mean"] = _m(rolloff)
    feats["zcr_mean"] = _m(zcr)
    feats["zcr_std"] = _s(zcr)
    feats["rms_mean"] = _m(rms)
    feats["rms_std"] = _s(rms)
    feats["rms_max"] = float(np.max(rms))
    feats["flatness_mean"] = _m(flatness)
    feats["duration"] = len(yt) / sr
    # Temporal energy pattern (cough has sharp attack)
    if len(rms[0]) > 4:
        q = len(rms[0]) // 4
        feats["energy_attack"] = float(np.mean(rms[0][:q]))
        feats["energy_sustain"] = float(np.mean(rms[0][q:]))
        feats["attack_ratio"] = feats["energy_attack"] / (feats["energy_sustain"] + 1e-10)
    else:
        feats["energy_attack"] = feats["rms_mean"]
        feats["energy_sustain"] = feats["rms_mean"]
        feats["attack_ratio"] = 1.0
    return feats


@dataclass
class CoughTrainResult:
    model_path: Path
    report: str
    train_size: int
    test_size: int


def train_cough_model(data_dir: Path = None) -> CoughTrainResult:
    """Train cough classifier from labeled audio folders."""
    data_dir = data_dir or (Path(__file__).resolve().parents[1] / "data" / "raw_medical" / "cough")
    rows = []
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            try:
                y, _ = librosa.load(wav, sr=SR, mono=True)
                row = _extract_cough_features(y)
                row["label"] = label_dir.name
                rows.append(row)
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No cough data found in {data_dir}. Need subfolders: dry_cough/, wet_cough/, not_cough/")
    x = df.drop(columns=["label"])
    y = df["label"]
    xt, xe, yt, ye = train_test_split(x, y, test_size=0.2, random_state=42,
                                       stratify=y if len(y.unique()) > 1 else None)
    m = RandomForestClassifier(n_estimators=150, random_state=42)
    m.fit(xt, yt)
    rpt = classification_report(ye, m.predict(xe), zero_division=0)
    COUGH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, COUGH_MODEL_PATH)
    return CoughTrainResult(model_path=COUGH_MODEL_PATH, report=rpt,
                            train_size=len(xt), test_size=len(xe))


def load_cough_model():
    return joblib.load(COUGH_MODEL_PATH)


def predict_cough(audio_path: Path) -> Dict:
    """Predict cough type from audio file."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    feats = _extract_cough_features(y)
    model = load_cough_model()
    df = pd.DataFrame([feats])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    classes = model.classes_
    scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    return {
        "prediction": pred,
        "prediction_vi": LABEL_VI.get(pred, pred),
        "confidence": round(float(max(proba)), 4),
        "scores": scores,
        "scores_vi": {LABEL_VI.get(k, k): v for k, v in scores.items()},
    }
