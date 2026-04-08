"""Đánh giá ngữ điệu tiếng Việt — so sánh MFCC, pitch, rhythm, energy qua DTW."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import librosa, numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13
HOP_LENGTH = 256
DATASET_DIR: Optional[Path] = None


def _find_dataset_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    for c in [root / "760-Hours-Vietnamese-Speech-Data-by-Mobile-Phone-main"
                   / "760-Hours-Vietnamese-Speech-Data-by-Mobile-Phone-main",
              root / "760-Hours-Vietnamese-Speech-Data-by-Mobile-Phone-main"]:
        if c.exists() and any(c.glob("*.wav")):
            return c
    for d in root.rglob("*.wav"):
        if any(d.parent.glob("*.txt")):
            return d.parent
    raise FileNotFoundError("Không tìm thấy thư mục dataset Vietnamese Speech.")


def get_dataset_dir() -> Path:
    global DATASET_DIR
    if DATASET_DIR is None:
        DATASET_DIR = _find_dataset_dir()
    return DATASET_DIR


def list_sentences(limit: int = 50) -> List[Dict]:
    ds_dir = get_dataset_dir()
    sentences = []
    for txt_path in sorted(ds_dir.glob("*.txt"))[:limit]:
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        try:
            dur = librosa.get_duration(path=wav_path)
        except Exception:
            dur = 0
        sentences.append({"id": txt_path.stem, "text": text,
                          "wav_file": wav_path.name, "duration_sec": round(dur, 2)})
    return sentences


def _load_and_trim(fp: Path) -> np.ndarray:
    y, _ = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def _extract_mfcc(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return np.vstack([mfcc, librosa.feature.delta(mfcc)]).T


def _extract_pitch(y: np.ndarray) -> np.ndarray:
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return np.nan_to_num(f0, nan=0.0)


def _extract_energy(y: np.ndarray) -> np.ndarray:
    return librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]


def _fast_dtw(seq1: np.ndarray, seq2: np.ndarray, radius: int = 10) -> float:
    if seq1.ndim == 1: seq1 = seq1.reshape(-1, 1)
    if seq2.ndim == 1: seq2 = seq2.reshape(-1, 1)
    n, m = len(seq1), len(seq2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    scale = m / n if n > 0 else 1.0
    for i in range(1, n + 1):
        jc = int(round(i * scale))
        for j in range(max(1, jc - radius), min(m, jc + radius) + 1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return D[n, m] / (n + m)


def _dist_to_score(dist: float, scale: float = 15.0) -> float:
    return float(np.clip(100.0 * np.exp(-dist / scale), 0, 100))


def _pitch_score(f0_ref: np.ndarray, f0_user: np.ndarray) -> Tuple[float, str]:
    rv, uv = f0_ref[f0_ref > 0], f0_user[f0_user > 0]
    if len(rv) < 5 or len(uv) < 5:
        return 70.0, "Không đủ dữ liệu pitch để đánh giá chi tiết."
    rs = 12.0 * np.log2(rv / np.median(rv) + 1e-10)
    us = 12.0 * np.log2(uv / np.median(uv) + 1e-10)
    score = _dist_to_score(_fast_dtw(rs, us, radius=15), scale=3.0)
    fb = []
    rr = (np.percentile(us, 95) - np.percentile(us, 5)) / (np.percentile(rs, 95) - np.percentile(rs, 5) + 1e-10)
    if rr < 0.6:
        fb.append("Ngữ điệu hơi đơn điệu — thử nhấn nhá rõ hơn các thanh sắc/huyền/hỏi/ngã"); score *= 0.85
    elif rr > 1.5:
        fb.append("Ngữ điệu dao động quá mạnh — thử nói đều hơn"); score *= 0.9
    if abs(np.std(us) - np.std(rs)) > 1.5:
        fb.append("Độ biến thiên cao độ khác mẫu — chú ý thanh điệu từng từ")
    if not fb:
        fb.append("Ngữ điệu tốt! Thanh điệu khá chính xác" if score >= 80
                   else "Ngữ điệu khá ổn, cần cải thiện một số thanh" if score >= 60
                   else "Cần luyện thêm ngữ điệu — nghe kỹ mẫu và bắt chước đường nét lên/xuống")
    return float(np.clip(score, 0, 100)), "; ".join(fb)


def _rhythm_score(y_ref: np.ndarray, y_user: np.ndarray) -> Tuple[float, str]:
    dur_ratio = (len(y_user) / SAMPLE_RATE) / (len(y_ref) / SAMPLE_RATE + 1e-10)
    o_ref = librosa.onset.onset_strength(y=y_ref, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    o_usr = librosa.onset.onset_strength(y=y_user, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    o_ref, o_usr = o_ref / (np.max(o_ref) + 1e-10), o_usr / (np.max(o_usr) + 1e-10)
    score = _dist_to_score(_fast_dtw(o_ref, o_usr, radius=15), scale=5.0)
    fb = []
    if dur_ratio < 0.65:
        fb.append("Nói quá nhanh — thử chậm lại để rõ từng từ"); score *= 0.8
    elif dur_ratio > 1.5:
        fb.append("Nói hơi chậm — thử nói tự nhiên hơn"); score *= 0.9
    if not fb:
        fb.append("Nhịp điệu tự nhiên, tốc độ phù hợp" if score >= 80
                   else "Nhịp điệu tạm ổn, cần đều hơn" if score >= 60
                   else "Cần cải thiện nhịp điệu — nghe mẫu và bắt chước tốc độ")
    return float(np.clip(score, 0, 100)), "; ".join(fb)


def evaluate_pronunciation(user_audio_path: Path, reference_id: str) -> Dict:
    ds_dir = get_dataset_dir()
    ref_wav = ds_dir / f"{reference_id}.wav"
    if not ref_wav.exists():
        raise FileNotFoundError(f"Không tìm thấy audio mẫu: {reference_id}.wav")
    ref_txt = ds_dir / f"{reference_id}.txt"
    ref_text = ref_txt.read_text(encoding="utf-8").strip() if ref_txt.exists() else ""

    y_ref, y_user = _load_and_trim(ref_wav), _load_and_trim(user_audio_path)

    mfcc_sc = _dist_to_score(_fast_dtw(_extract_mfcc(y_ref), _extract_mfcc(y_user), 20), 12.0)
    pitch_sc, pitch_fb = _pitch_score(_extract_pitch(y_ref), _extract_pitch(y_user))
    rhythm_sc, rhythm_fb = _rhythm_score(y_ref, y_user)
    energy_sc = _dist_to_score(_fast_dtw(_extract_energy(y_ref), _extract_energy(y_user), 15), 8.0)

    W = {"mfcc": 0.30, "pitch": 0.40, "rhythm": 0.15, "energy": 0.15}
    overall = mfcc_sc * W["mfcc"] + pitch_sc * W["pitch"] + rhythm_sc * W["rhythm"] + energy_sc * W["energy"]

    grade = ("Xuất sắc 🌟" if overall >= 85 else "Tốt 👍" if overall >= 70
             else "Khá 📝" if overall >= 55 else "Trung bình ⚠️" if overall >= 40 else "Cần cố gắng 💪")

    fb = [pitch_fb, rhythm_fb]
    if mfcc_sc < 50: fb.append("Phát âm nguyên âm/phụ âm cần rõ ràng hơn")
    if mfcc_sc >= 75 and pitch_sc >= 75: fb.append("Phát âm và ngữ điệu khá tốt!")

    return {
        "overall_score": round(overall, 1), "grade": grade,
        "mfcc_score": round(mfcc_sc, 1), "pitch_score": round(pitch_sc, 1),
        "rhythm_score": round(rhythm_sc, 1), "energy_score": round(energy_sc, 1),
        "reference_text": ref_text, "reference_id": reference_id,
        "feedback": " | ".join(fb),
        "details": {"weights": W,
                    "user_duration_sec": round(len(y_user) / SAMPLE_RATE, 2),
                    "ref_duration_sec": round(len(y_ref) / SAMPLE_RATE, 2)},
    }
