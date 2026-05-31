"""Sàng lọc dấu hiệu trầm cảm qua giọng nói.

Phân tích các đặc trưng giọng nói liên quan đến trầm cảm:
- Pitch thấp & ít biến thiên (monotone)
- Energy thấp, giọng yếu
- Nhịp nói chậm, nhiều khoảng lặng
- Jitter & Shimmer cao (giọng run, không ổn định)
- HNR thấp (giọng breathy/rough)

Dựa trên nghiên cứu: Cummins et al. (2015), Low et al. (2011)
"""
from pathlib import Path
from typing import Dict
import librosa, numpy as np

SR = 16000


def _jitter(f0: np.ndarray) -> float:
    """Relative jitter — biến thiên chu kỳ giọng nói."""
    voiced = f0[f0 > 0]
    if len(voiced) < 3:
        return 0.0
    periods = 1.0 / voiced
    diffs = np.abs(np.diff(periods))
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _shimmer(y: np.ndarray, sr: int = SR) -> float:
    """Shimmer — biến thiên biên độ giữa các chu kỳ."""
    frames = librosa.util.frame(y, frame_length=int(sr * 0.03), hop_length=int(sr * 0.01))
    amps = np.max(np.abs(frames), axis=0)
    if len(amps) < 3:
        return 0.0
    diffs = np.abs(np.diff(amps))
    return float(np.mean(diffs) / (np.mean(amps) + 1e-10))


def _hnr(y: np.ndarray, sr: int = SR) -> float:
    """Harmonics-to-Noise Ratio (dB) — độ trong trẻo giọng nói."""
    ac = librosa.autocorrelate(y, max_size=sr // 60)
    if len(ac) < 2 or ac[0] == 0:
        return 0.0
    peak = np.max(ac[1:]) if len(ac) > 1 else 0
    ratio = peak / ac[0]
    return float(10 * np.log10(ratio / (1 - ratio + 1e-10) + 1e-10))


def _speech_rate(y: np.ndarray, sr: int = SR) -> float:
    """Ước tính tốc độ nói (onset/giây)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    dur = len(y) / sr
    return len(onsets) / dur if dur > 0 else 0


def _pause_ratio(y: np.ndarray, sr: int = SR) -> float:
    """Tỷ lệ khoảng lặng trong audio."""
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    threshold = np.mean(rms) * 0.3
    silent = np.sum(rms < threshold)
    return float(silent / (len(rms) + 1e-10))


def extract_depression_features(audio_path: Path) -> Dict[str, float]:
    """Trích xuất toàn bộ đặc trưng liên quan trầm cảm."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < SR * 0.5:
        yt = y

    # Pitch
    f0, _, _ = librosa.pyin(yt, fmin=50, fmax=500, sr=SR)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]

    # Energy
    rms = librosa.feature.rms(y=yt)[0]

    feats = {
        "pitch_mean": float(np.mean(voiced)) if len(voiced) > 0 else 0,
        "pitch_std": float(np.std(voiced)) if len(voiced) > 0 else 0,
        "pitch_range": float(np.ptp(voiced)) if len(voiced) > 0 else 0,
        "energy_mean": float(np.mean(rms)),
        "energy_std": float(np.std(rms)),
        "jitter": _jitter(f0),
        "shimmer": _shimmer(yt),
        "hnr": _hnr(yt),
        "speech_rate": _speech_rate(yt),
        "pause_ratio": _pause_ratio(yt),
        "duration_sec": round(len(yt) / SR, 2),
    }
    return feats


def screen_depression(audio_path: Path) -> Dict:
    """Sàng lọc dấu hiệu trầm cảm — trả về điểm risk + phân tích chi tiết."""
    feats = extract_depression_features(audio_path)

    indicators = []
    risk_score = 0.0

    # 1. Pitch analysis — trầm cảm: pitch thấp, ít biến thiên
    if feats["pitch_mean"] > 0:
        if feats["pitch_std"] < 15:
            indicators.append({"feature": "Giọng đơn điệu", "detail": f"Biến thiên pitch thấp ({feats['pitch_std']:.1f} Hz)",
                               "severity": "cao", "weight": 0.20})
            risk_score += 0.20
        elif feats["pitch_std"] < 25:
            indicators.append({"feature": "Giọng ít biến thiên", "detail": f"Pitch std = {feats['pitch_std']:.1f} Hz",
                               "severity": "trung bình", "weight": 0.10})
            risk_score += 0.10

    # 2. Energy — trầm cảm: giọng yếu, energy thấp
    if feats["energy_mean"] < 0.02:
        indicators.append({"feature": "Giọng yếu", "detail": f"Năng lượng trung bình thấp ({feats['energy_mean']:.4f})",
                           "severity": "cao", "weight": 0.15})
        risk_score += 0.15
    elif feats["energy_mean"] < 0.04:
        indicators.append({"feature": "Giọng hơi yếu", "detail": f"Energy = {feats['energy_mean']:.4f}",
                           "severity": "trung bình", "weight": 0.08})
        risk_score += 0.08

    # 3. Speech rate — trầm cảm: nói chậm
    if feats["speech_rate"] < 2.0:
        indicators.append({"feature": "Nhịp nói chậm", "detail": f"{feats['speech_rate']:.1f} onset/giây",
                           "severity": "cao", "weight": 0.15})
        risk_score += 0.15
    elif feats["speech_rate"] < 3.0:
        indicators.append({"feature": "Nhịp nói hơi chậm", "detail": f"{feats['speech_rate']:.1f} onset/giây",
                           "severity": "trung bình", "weight": 0.08})
        risk_score += 0.08

    # 4. Pause ratio — trầm cảm: nhiều khoảng lặng
    if feats["pause_ratio"] > 0.5:
        indicators.append({"feature": "Nhiều khoảng lặng", "detail": f"{feats['pause_ratio']:.0%} thời gian im lặng",
                           "severity": "cao", "weight": 0.15})
        risk_score += 0.15
    elif feats["pause_ratio"] > 0.35:
        indicators.append({"feature": "Khoảng lặng khá nhiều", "detail": f"{feats['pause_ratio']:.0%}",
                           "severity": "trung bình", "weight": 0.08})
        risk_score += 0.08

    # 5. Jitter — trầm cảm: giọng run
    if feats["jitter"] > 0.03:
        indicators.append({"feature": "Giọng run (jitter cao)", "detail": f"Jitter = {feats['jitter']:.4f}",
                           "severity": "cao", "weight": 0.10})
        risk_score += 0.10
    elif feats["jitter"] > 0.015:
        indicators.append({"feature": "Jitter hơi cao", "detail": f"Jitter = {feats['jitter']:.4f}",
                           "severity": "trung bình", "weight": 0.05})
        risk_score += 0.05

    # 6. Shimmer
    if feats["shimmer"] > 0.15:
        indicators.append({"feature": "Biên độ không ổn định (shimmer)", "detail": f"Shimmer = {feats['shimmer']:.4f}",
                           "severity": "cao", "weight": 0.10})
        risk_score += 0.10

    # 7. HNR — trầm cảm: giọng breathy, HNR thấp
    if feats["hnr"] < 5:
        indicators.append({"feature": "Giọng thở/khàn (HNR thấp)", "detail": f"HNR = {feats['hnr']:.1f} dB",
                           "severity": "trung bình", "weight": 0.10})
        risk_score += 0.10

    risk_score = min(risk_score, 1.0)
    risk_pct = round(risk_score * 100, 1)

    if risk_pct >= 60:
        level = "cao"
        level_vi = "⚠️ Nguy cơ cao"
        advice = "Có nhiều dấu hiệu bất thường trong giọng nói. Khuyến nghị tham vấn chuyên gia tâm lý."
    elif risk_pct >= 30:
        level = "trung bình"
        level_vi = "🔶 Nguy cơ trung bình"
        advice = "Một số đặc trưng giọng nói cho thấy dấu hiệu mệt mỏi/căng thẳng. Nên nghỉ ngơi và theo dõi."
    else:
        level = "thấp"
        level_vi = "✅ Nguy cơ thấp"
        advice = "Giọng nói trong phạm vi bình thường. Không phát hiện dấu hiệu bất thường rõ rệt."

    return {
        "risk_score": risk_pct,
        "risk_level": level,
        "risk_level_vi": level_vi,
        "advice": advice,
        "indicators": indicators,
        "indicator_count": len(indicators),
        "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()},
        "disclaimer": "Đây chỉ là công cụ sàng lọc sơ bộ, KHÔNG thay thế chẩn đoán y khoa. "
                      "Nếu có lo ngại, hãy liên hệ chuyên gia tâm lý/bác sĩ.",
    }
