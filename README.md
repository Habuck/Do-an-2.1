# Easy Kit Audio Detection

Phân tích âm thanh real-time, nhận diện cảm xúc bằng AI, luyện phát âm tiếng Việt.

## Cấu trúc

```
project/
  audio_project/
    preprocessing/audio.py
    features/extractor.py
    models/train.py, gender.py, emotion.py, pronunciation.py
    inference/predict.py
    config.py, pipeline.py
  data/raw/{speech,noise,music,silence}/
  models/audio_classifier.joblib, gender_classifier.joblib
  main.py          # CLI
  web_app.py       # Flask backend
  main.html        # Frontend
```

## Setup

```bash
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

## CLI

```bash
python main.py prepare          # Tiền xử lý audio
python main.py train            # Train audio classifier
python main.py all              # Prepare + train
python main.py infer --file data/raw/speech/example.wav
python main.py gender-train     # Train gender KNN
python main.py gender-infer --meanfreq 0.12 --sd 0.05 --centroid 0.12 --meanfun 0.14 --IQR 0.04 --median 0.13
```

## Web App

```bash
python web_app.py
```

Mở http://127.0.0.1:5000

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/health` | Trạng thái server + models |
| POST | `/api/train` | Train audio classifier |
| POST | `/api/predict` | Dự đoán file audio (speech/noise/music/silence) |
| POST | `/api/gender/train` | Train gender KNN |
| POST | `/api/gender/predict` | Dự đoán giới tính (JSON 6 features) |
| POST | `/api/emotion/predict` | AI nhận diện cảm xúc (wav2vec2) |
| GET | `/api/pronunciation/sentences` | Danh sách câu mẫu tiếng Việt |
| GET | `/api/pronunciation/audio/<id>` | Audio mẫu |
| POST | `/api/pronunciation/evaluate` | Chấm điểm phát âm (DTW) |

## Models

- **Audio Classifier** — RandomForest, 4 classes (speech, noise, music, silence)
- **Gender KNN** — K-Nearest Neighbors từ voice.csv (98% accuracy)
- **Emotion wav2vec2** — Pretrained Hugging Face `superb/wav2vec2-base-superb-er`
- **Pronunciation DTW** — So sánh MFCC, pitch, rhythm, energy với audio mẫu
