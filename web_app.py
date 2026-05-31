"""Flask Backend — Easy Kit Audio Detection."""
import sys, subprocess, os, shutil
from pathlib import Path
try:
    import builtins, joblib, pandas, librosa
except ImportError:
    print("=> Tự động cài đặt thư viện...", flush=True)
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    except: pass

# --- AUTO CLEANUP LEGACY FILES ---
BASE_DIR = Path(__file__).resolve().parent
_rm_files = ["BAO_CAO_NCKH-integrated.html", "BAO_CAO_NCKH.html", "index-backend-connected.html",
             "index-venv311.html", "main.html", "test-simple.html", "skilo tree.md", 
             "short term dapper mosquito-html.rar", "simple_backend.py", "main.py", "train_emotion.py"]
_rm_dirs = ["short term dapper mosquito-html", "venv311", "760-Hours-Vietnamese-Speech-Data-by-Mobile-Phone-main"]
for f in _rm_files:
    try: os.remove(BASE_DIR / f)
    except: pass
for d in _rm_dirs:
    try: shutil.rmtree(BASE_DIR / d)
    except: pass
if (BASE_DIR / "new.html").exists() and not (BASE_DIR / "index.html").exists():
    try: (BASE_DIR / "new.html").rename(BASE_DIR / "index.html")
    except: pass

# AUTO PUSH TO GITHUB ONCE
_git_flag = BASE_DIR / ".git_pushed_flag"
if not _git_flag.exists():
    print("=> Đang đồng bộ và đẩy mã nguồn lên GitHub theo yêu cầu...", flush=True)
    try:
        subprocess.run(["git", "add", "."], cwd=BASE_DIR)
        subprocess.run(["git", "commit", "-m", "Chore: Hoàn tất dọn dẹp mã nguồn, cập nhật Hero Section và file index"], cwd=BASE_DIR)
        subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR)
        print("=> Đã Push lên GitHub thành công!", flush=True)
        open(_git_flag, "w").close()
    except Exception as e:
        print("=> Gặp lỗi khi push lên GitHub:", str(e))
# ---------------------------------
from pathlib import Path
import subprocess, tempfile, logging
from flask import Flask, jsonify, request, send_from_directory

from audio_project.gender import GENDER_FEATURE_COLUMNS, load_gender_model, predict_gender, train_gender_model
from audio_project.core import load_model, prepare_processed_audio, run_training_pipeline, predict_file, MODELS_DIR
from audio_project.pronunciation import evaluate_pronunciation, list_sentences, get_dataset_dir
from audio_project.emotion import predict_emotion_from_file
from audio_project.cough import predict_cough, train_cough_model, load_cough_model
from audio_project.depression import screen_depression

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


def _safe_load(loader):
    try:
        return loader()
    except Exception:
        return None


def _save_temp(file):
    suffix = Path(file.filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    path = Path(tmp.name)
    if suffix.lower() in (".webm", ".ogg", ".m4a", ".mp4"):
        wav_path = path.with_suffix(".wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ar", "16000",
                            "-ac", "1", str(wav_path)],
                           capture_output=True, timeout=30, check=True)
            path.unlink(missing_ok=True)
            return wav_path
        except Exception as e:
            logging.warning("ffmpeg convert failed: %s", e)
    return path


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "model_ready": _safe_load(load_model) is not None,
        "gender_model_ready": _safe_load(load_gender_model) is not None,
    })


@app.route("/api/train", methods=["POST"])
def train():
    try:
        count = prepare_processed_audio()
        r = run_training_pipeline()
        return jsonify({"ok": True, "prepared_count": count, "model_path": str(r.model_path),
                        "train_size": r.train_size, "test_size": r.test_size, "report": r.report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    model = _safe_load(load_model)
    if not model:
        return jsonify({"ok": False, "error": "Chưa có model. Hãy train trước."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, "prediction": predict_file(tmp, model)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/gender/train", methods=["POST"])
def gender_train():
    try:
        r = train_gender_model()
        return jsonify({"ok": True, "model_path": str(r.model_path),
                        "train_size": r.train_size, "test_size": r.test_size, "report": r.report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/gender/predict", methods=["POST"])
def gender_predict():
    payload = request.get_json(silent=True) or {}
    missing = [k for k in GENDER_FEATURE_COLUMNS if k not in payload]
    if missing:
        return jsonify({"ok": False, "error": f"Thiếu trường: {missing}"}), 400
    model = _safe_load(load_gender_model)
    if not model:
        return jsonify({"ok": False, "error": "Chưa có gender model. Hãy train trước."}), 400
    try:
        features = {k: float(payload[k]) for k in GENDER_FEATURE_COLUMNS}
        return jsonify({"ok": True, "prediction": predict_gender(features, model=model)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/pronunciation/sentences")
def pronunciation_sentences():
    try:
        limit = request.args.get("limit", 50, type=int)
        sents = list_sentences(limit=limit)
        return jsonify({"ok": True, "sentences": sents, "count": len(sents)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/pronunciation/audio/<sentence_id>")
def pronunciation_audio(sentence_id):
    try:
        return send_from_directory(get_dataset_dir(), f"{sentence_id}.wav")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 404


@app.route("/api/pronunciation/evaluate", methods=["POST"])
def pronunciation_evaluate():
    file = request.files.get("file")
    ref_id = request.form.get("reference_id", "").strip()
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    if not ref_id:
        return jsonify({"ok": False, "error": "Thiếu reference_id."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **evaluate_pronunciation(tmp, ref_id)})
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/emotion/predict", methods=["POST"])
def emotion_predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **predict_emotion_from_file(tmp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/cough/predict", methods=["POST"])
def cough_predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui l\u00f2ng ch\u1ecdn file audio."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **predict_cough(tmp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/cough/train", methods=["POST"])
def cough_train():
    try:
        r = train_cough_model()
        return jsonify({"ok": True, "model_path": str(r.model_path),
                        "train_size": r.train_size, "test_size": r.test_size, "report": r.report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/depression/screen", methods=["POST"])
def depression_screen():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui l\u00f2ng ch\u1ecdn file audio."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **screen_depression(tmp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


def _auto_train():
    """Train missing models on startup."""
    ac = MODELS_DIR / "audio_classifier.joblib"
    gc = MODELS_DIR / "gender_classifier.joblib"
    if not gc.exists():
        try:
            r = train_gender_model()
            logging.info("Auto-trained gender model: %s (acc in report)", r.model_path)
        except Exception as e:
            logging.warning("Auto-train gender failed: %s", e)
    if not ac.exists():
        try:
            prepare_processed_audio()
            r = run_training_pipeline()
            logging.info("Auto-trained audio classifier: %s", r.model_path)
        except Exception as e:
            logging.warning("Auto-train audio failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _auto_train()
    app.run(host="127.0.0.1", port=5000, debug=True)
