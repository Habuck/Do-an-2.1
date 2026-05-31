"""Train & predict giới tính bằng KNN từ 6 đặc trưng âm thanh."""
from dataclasses import dataclass
from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from audio_project.core import MODELS_DIR, RAND

GENDER_FEATURE_COLUMNS = ["meanfreq", "sd", "centroid", "meanfun", "IQR", "median"]
GENDER_LABEL_COLUMN = "label"


@dataclass
class GenderTrainResult:
    model_path: Path
    report: str
    train_size: int
    test_size: int


def find_voice_csv(base_dir: Path | None = None) -> Path:
    search_root = base_dir or Path(__file__).resolve().parents[1]
    candidates = sorted(search_root.rglob("voice.csv"))
    if not candidates:
        raise FileNotFoundError("Không tìm thấy file voice.csv trong project.")
    return candidates[0]


def train_gender_model(csv_path: Path | None = None, model_path: Path | None = None) -> GenderTrainResult:
    df = pd.read_csv(csv_path or find_voice_csv())
    missing = [c for c in GENDER_FEATURE_COLUMNS + [GENDER_LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"voice.csv thiếu cột: {missing}")
    x, y = df[GENDER_FEATURE_COLUMNS], df[GENDER_LABEL_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RAND,
        stratify=y if len(y.unique()) > 1 else None)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    report = classification_report(y_test, model.predict(x_test), zero_division=0)
    model_path = model_path or (MODELS_DIR / "gender_classifier.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return GenderTrainResult(model_path=model_path, report=report,
                             train_size=len(x_train), test_size=len(x_test))


def load_gender_model(model_path: Path | None = None):
    return joblib.load(model_path or (MODELS_DIR / "gender_classifier.joblib"))


def predict_gender(features: dict[str, float], model=None) -> str:
    model = model or load_gender_model()
    frame = pd.DataFrame([[features[c] for c in GENDER_FEATURE_COLUMNS]], columns=GENDER_FEATURE_COLUMNS)
    return str(model.predict(frame)[0])
