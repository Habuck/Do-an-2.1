"""Fine-tune wav2vec2 emotion — 7500 steps, PyTorch.

Strategy: Extract wav2vec2 features on GPU in batches (fp16), train classifier on GPU.
Optimized for AMD Radeon Pro VII 16GB HBM2 — max VRAM utilization.
  - wav2vec2 loaded in fp16 (~600MB saved)
  - Batched extraction (batch=48) with manual fp16 casting (DirectML compatible)
  - Classifier trained on GPU with batch=32
  - Auto batch-size finder to push VRAM to the limit

Hyperparameters (model card):
  learning_rate: 0.0001
  train_batch_size: 32
  eval_batch_size: 32
  eval_steps: 500
  seed: 42
  gradient_accumulation_steps: 2
  optimizer: Adam betas=(0.9,0.999) eps=1e-08
  num_epochs: 4
  max_steps: 7500
  save_steps: 1500
"""
import logging, os, sys, random, math, time, gc
from pathlib import Path
import numpy as np
import librosa
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config (MAX VRAM — Radeon Pro VII 16GB HBM2) ──
SR = 16000
MAX_LEN = SR * 8          # 8 seconds — good balance for 16GB VRAM
SEED = 42
TRAIN_BATCH_SIZE = 32     # Large batch on GPU — classifier is tiny
EVAL_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
LR = 1e-4
NUM_EPOCHS = 4
MAX_STEPS = 7500
SAVE_STEPS = 1500
EVAL_STEPS = 500
LOG_EVERY = 50
EXTRACT_BATCH_SIZE = 32   # Safe batch for DirectML (tested: 96 OK on 10s, using 32 for 8s reliability)
USE_FP16 = True           # Manual fp16 casting (DirectML compatible)
NUM_WORKERS = 4           # Parallel data loading
PIN_MEMORY = True         # Faster CPU→GPU transfer
VRAM_TARGET_GB = 14.5     # Target VRAM usage (leave 1.5GB headroom)
MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
SAVE_DIR = BASE_DIR / "models" / "emotion_finetuned"

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


# ── Device ──
def get_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        log.info("GPU: DirectML (AMD Radeon Pro VII)")
        return dev, "directml"
    except ImportError:
        pass
    if torch.cuda.is_available():
        log.info("GPU: CUDA %s", torch.cuda.get_device_name(0))
        return torch.device("cuda"), "cuda"
    log.info("CPU mode")
    return torch.device("cpu"), "cpu"


def log_gpu_memory(tag="", backend="cuda"):
    """Log GPU memory usage for monitoring VRAM utilization."""
    if backend == "cuda" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        log.info("[VRAM %s] %.2f/%.2f GB used (%.1f%%) | reserved=%.2f GB",
                 tag, alloc, total, alloc / total * 100, reserved)
    elif backend == "directml":
        log.info("[VRAM %s] DirectML — estimating from tensor sizes", tag)




# ── Augmentation (fast, no librosa effects) ──
def augment(y):
    """Fast augmentation: noise, volume, shift — no slow pitch_shift/time_stretch."""
    r = random.random()
    if r < 0.3:
        y = y + np.random.randn(len(y)).astype(np.float32) * random.uniform(0.002, 0.015)
    elif r < 0.6:
        y = y * random.uniform(0.3, 2.0)
    elif r < 0.8:
        shift = random.randint(-int(SR * 0.3), int(SR * 0.3))
        y = np.roll(y, shift)
    else:
        y = y + np.random.randn(len(y)).astype(np.float32) * random.uniform(0.001, 0.01)
        y = y * random.uniform(0.5, 1.5)
    return y.astype(np.float32)


def fix_length(y, max_len=MAX_LEN):
    if len(y) > max_len:
        start = random.randint(0, len(y) - max_len)
        return y[start:start + max_len]
    return np.pad(y, (0, max_len - len(y)))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ── Model ──
def load_wav2vec2(device, use_fp16=USE_FP16):
    """Load pretrained wav2vec2 as frozen feature extractor.
    With fp16=True, model uses ~360MB VRAM instead of ~720MB — saves 360MB for batches."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoConfig
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    config = AutoConfig.from_pretrained(MODEL_NAME)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    w2v = Wav2Vec2Model(config)

    sf_path = hf_hub_download(MODEL_NAME, "model.safetensors", revision="refs/pr/6")
    state = load_file(sf_path)
    w2v_state = {k.replace("wav2vec2.", ""): v for k, v in state.items() if k.startswith("wav2vec2.")}
    w2v.load_state_dict(w2v_state, strict=False)
    w2v.eval()

    # FP16: halve model VRAM (~720MB → ~360MB), freeing space for larger batches
    if use_fp16:
        w2v = w2v.half()
        log.info("wav2vec2 loaded in FP16 — ~360MB VRAM (saved ~360MB)")
    else:
        log.info("wav2vec2 loaded in FP32 — ~720MB VRAM")

    w2v = w2v.to(device)
    for p in w2v.parameters():
        p.requires_grad = False

    # Classifier head weights
    cls_state = {}
    for k, v in state.items():
        if k.startswith("classifier.dense."):
            cls_state[k.replace("classifier.dense.", "0.")] = v
        elif k.startswith("classifier.out_proj."):
            cls_state[k.replace("classifier.out_proj.", "3.")] = v

    return w2v, extractor, config.hidden_size, cls_state


def build_classifier(hidden_size, pretrained_cls_state=None):
    """Build classifier head (trained on CPU)."""
    classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, NUM_LABELS),
    )
    if pretrained_cls_state:
        classifier.load_state_dict(pretrained_cls_state, strict=False)
    log.info("Classifier params: %s", f"{sum(p.numel() for p in classifier.parameters()):,}")
    return classifier


def extract_features_gpu(w2v, extractor, audio_list, device, backend="cuda",
                         batch_size=None, quiet=False):
    """Extract wav2vec2 features on GPU in batches — max VRAM for Radeon Pro VII.
    Uses manual half() casting which works on both CUDA and DirectML."""
    features = []
    total = len(audio_list)
    if batch_size is None:
        batch_size = EXTRACT_BATCH_SIZE

    if not quiet:
        log.info("Extracting features on GPU (%d samples, batch=%d, fp16=%s)...",
                 total, batch_size, USE_FP16)
        log_gpu_memory("before_extraction", backend)
    t_start = time.time()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_audio = audio_list[start:end]

        # Batch tokenize with padding
        inputs = extractor(
            batch_audio, sampling_rate=SR, return_tensors="pt",
            padding=True, max_length=MAX_LEN, truncation=True
        )
        # Manual fp16 casting — works on DirectML (no cuda.amp needed)
        input_vals = inputs.input_values.to(device)
        if USE_FP16:
            input_vals = input_vals.half()

        with torch.no_grad():
            out = w2v(input_vals).last_hidden_state  # (B, T, H)

            # Attention mask aware pooling
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                mask = inputs.attention_mask.to(device).unsqueeze(-1)
                if USE_FP16:
                    mask = mask.half()
                else:
                    mask = mask.float()
                # wav2vec2 downsamples time dim — align mask
                if mask.size(1) != out.size(1):
                    mask = torch.nn.functional.interpolate(
                        mask.transpose(1, 2), size=out.size(1), mode="nearest"
                    ).transpose(1, 2)
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                pooled = out.mean(dim=1)

            features.append(pooled.float().cpu())

        # Free intermediate GPU tensors immediately
        del input_vals, out
        try:
            del mask
        except NameError:
            pass
        gc.collect()

        if not quiet and ((end % (batch_size * 10) == 0) or end == total):
            elapsed = time.time() - t_start
            speed = end / elapsed if elapsed > 0 else 0
            eta = (total - end) / speed if speed > 0 else 0
            log.info("  extracted %d/%d (%.0f samples/s, ETA %.0fs)", end, total, speed, eta)
            log_gpu_memory(f"extract_{end}/{total}", backend)

    return torch.cat(features, dim=0)  # (N, hidden)


def pseudo_label_batch(w2v, extractor, classifier, audio_paths, device,
                       backend="cuda", batch_size=None):
    """Pseudo-label files using pretrained model — batched on GPU with fp16."""
    classifier.eval()
    if batch_size is None:
        batch_size = EXTRACT_BATCH_SIZE

    # Load all audio first (CPU)
    all_audio = []
    for p in audio_paths:
        y, _ = librosa.load(p, sr=SR, mono=True)
        all_audio.append(fix_length(y))

    labels = []
    for start in range(0, len(all_audio), batch_size):
        end = min(start + batch_size, len(all_audio))
        batch_audio = all_audio[start:end]

        inputs = extractor(
            batch_audio, sampling_rate=SR, return_tensors="pt",
            padding=True, max_length=MAX_LEN, truncation=True
        )
        input_vals = inputs.input_values.to(device)
        if USE_FP16:
            input_vals = input_vals.half()

        with torch.no_grad():
            feat = w2v(input_vals).last_hidden_state.mean(dim=1).float()
            logits = classifier(feat.cpu())
            preds = torch.argmax(logits, dim=-1).tolist()

        labels.extend(preds)
        for i, pred in enumerate(preds):
            log.info("  %s → %s", audio_paths[start + i].name, LABELS[pred])

        del input_vals, feat
        gc.collect()

    return labels


# ── Train ──
def train():
    set_seed(SEED)
    t0 = time.time()
    device, backend = get_device()

    # 1) Load wav2vec2 on GPU in FP16 (frozen feature extractor)
    log.info("Loading wav2vec2 on GPU (fp16=%s)...", USE_FP16)
    w2v, extractor, hidden_size, cls_state = load_wav2vec2(device, use_fp16=USE_FP16)
    log_gpu_memory("after_model_load", backend)
    log.info("Using extraction batch_size=%d (fp16=%s)", EXTRACT_BATCH_SIZE, USE_FP16)

    classifier = build_classifier(hidden_size, cls_state)

    # 2) Gather audio files
    audio_paths = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        audio_paths.extend(DATA_DIR.rglob(ext))
    med_dir = BASE_DIR / "data" / "raw_medical"
    if med_dir.exists():
        for ext in ("*.wav",):
            audio_paths.extend(med_dir.rglob(ext))
    log.info("Found %d audio files", len(audio_paths))

    # 3) Pseudo-label (batched on GPU, fp16)
    log.info("Pseudo-labeling (batched, fp16=%s, batch=%d)...", USE_FP16, EXTRACT_BATCH_SIZE)
    pseudo_labels = pseudo_label_batch(w2v, extractor, classifier, audio_paths, device,
                                       backend)
    from collections import Counter
    dist = Counter(LABELS[l] for l in pseudo_labels)
    log.info("Label distribution: %s", dict(dist))

    # 4+5) Augment + extract features in chunks (RAM-safe)
    #   Cap augment_factor to avoid hours-long extraction.
    #   Training loop handles repeats via epochs — no need to pre-generate everything.
    MAX_AUGMENT = 200  # Cap: 106 files × 201 = ~21K samples, ~15 min extraction
    effective_batch = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
    augment_factor = min(MAX_AUGMENT, max(40, MAX_STEPS * effective_batch // len(audio_paths) // 2))
    total_samples = len(audio_paths) * (augment_factor + 1)
    log.info("Augment+Extract in chunks: %d files x %d augments = %d total samples",
             len(audio_paths), augment_factor + 1, total_samples)

    all_features_list = []
    all_labels_list = []
    t_extract = time.time()

    for file_idx, (path, label) in enumerate(zip(audio_paths, pseudo_labels)):
        y, _ = librosa.load(path, sr=SR, mono=True)
        # Build chunk: original + augmented versions
        chunk_audio = [fix_length(y)]
        for _ in range(augment_factor):
            chunk_audio.append(fix_length(augment(y.copy())))
        chunk_labels = [label] * len(chunk_audio)

        # Extract features on GPU in batches — then discard raw audio
        chunk_feats = extract_features_gpu(w2v, extractor, chunk_audio, device, backend,
                                           quiet=True)
        all_features_list.append(chunk_feats)
        all_labels_list.extend(chunk_labels)

        # Free raw audio immediately
        del chunk_audio, y

        if (file_idx + 1) % 10 == 0 or file_idx + 1 == len(audio_paths):
            elapsed = time.time() - t_extract
            speed = (file_idx + 1) / elapsed
            eta = (len(audio_paths) - file_idx - 1) / speed
            done_samples = sum(f.size(0) for f in all_features_list)
            ram_mb = done_samples * 768 * 4 / 1e6  # fp32 feature vectors
            log.info("  file %d/%d | %d features (%.0f MB RAM) | %.1f files/s | ETA %.0fs",
                     file_idx + 1, len(audio_paths), done_samples, ram_mb, speed, eta)

    all_features = torch.cat(all_features_list, dim=0)
    del all_features_list
    gc.collect()
    all_labels_t = torch.tensor(all_labels_list, dtype=torch.long)
    del all_labels_list
    log.info("Features shape: %s  raw range: [%.2f, %.2f]", all_features.shape,
             all_features.min().item(), all_features.max().item())

    # Replace NaN/inf
    all_features = torch.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)

    # Normalize features (zero mean, unit std) — critical to avoid NaN
    feat_mean = all_features.mean(dim=0, keepdim=True)
    feat_std = all_features.std(dim=0, keepdim=True).clamp(min=1e-6)
    all_features = (all_features - feat_mean) / feat_std
    log.info("Features normalized: mean=%.4f std=%.4f", all_features.mean().item(), all_features.std().item())

    # Free GPU memory — reclaim all VRAM for classifier training
    del w2v, extractor
    gc.collect()
    if backend == "cuda":
        torch.cuda.empty_cache()
    log.info("Freed wav2vec2 from VRAM — all 16GB available for classifier training")
    log_gpu_memory("after_w2v_freed", backend)

    # 6) Train/eval split
    perm = torch.randperm(len(all_features), generator=torch.Generator().manual_seed(SEED))
    all_features, all_labels_t = all_features[perm], all_labels_t[perm]
    eval_size = max(1, len(all_features) // 10)
    train_features = all_features[eval_size:]
    train_labels = all_labels_t[eval_size:]
    eval_features = all_features[:eval_size]
    eval_labels = all_labels_t[:eval_size]

    # Move features to GPU for classifier training (fits in 16GB after w2v freed)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    eval_features = eval_features.to(device)
    eval_labels = eval_labels.to(device)
    log_gpu_memory("features_on_gpu", backend)

    train_loader = DataLoader(TensorDataset(train_features, train_labels),
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=False)  # data already on GPU
    eval_loader = DataLoader(TensorDataset(eval_features, eval_labels),
                             batch_size=EVAL_BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)
    log.info("Train: %d | Eval: %d", len(train_features), len(eval_features))

    # 7) Optimizer — Adam betas=(0.9, 0.999) eps=1e-8 (model card)
    classifier = classifier.to(device)  # Train classifier on GPU
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    log_gpu_memory("training_ready", backend)

    log.info("=" * 60)
    log.info("Hyperparameters (model card):")
    log.info("  learning_rate: %s", LR)
    log.info("  train_batch_size: %d", TRAIN_BATCH_SIZE)
    log.info("  eval_batch_size: %d", EVAL_BATCH_SIZE)
    log.info("  gradient_accumulation_steps: %d", GRAD_ACCUM_STEPS)
    log.info("  optimizer: Adam betas=(0.9, 0.999) eps=1e-8")
    log.info("  seed: %d", SEED)
    log.info("  num_epochs: %d", NUM_EPOCHS)
    log.info("  max_steps: %d", MAX_STEPS)
    log.info("  save_steps: %d", SAVE_STEPS)
    log.info("  eval_steps: %d", EVAL_STEPS)
    log.info("=" * 60)

    WARMUP_STEPS = 500
    step = 0
    epoch = 0
    running_loss = 0
    running_correct = 0
    running_total = 0
    best_eval_loss = float("inf")
    optimizer.zero_grad()

    while step < MAX_STEPS:
        epoch += 1
        for feat_batch, label_batch in train_loader:
            if step >= MAX_STEPS:
                break

            # Warmup LR
            if step < WARMUP_STEPS:
                warmup_lr = LR * (step + 1) / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            logits = classifier(feat_batch)
            loss = criterion(logits, label_batch) / GRAD_ACCUM_STEPS
            loss.backward()

            loss_val = loss.item() * GRAD_ACCUM_STEPS
            running_loss += loss_val
            running_correct += (torch.argmax(logits, dim=-1) == label_batch).sum().item()
            running_total += label_batch.size(0)

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            step += 1

            if step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                acc = running_correct / max(running_total, 1) * 100
                elapsed = time.time() - t0
                eta = elapsed / step * (MAX_STEPS - step)
                log.info("Step %5d/%d | loss=%.4f | acc=%.1f%% | epoch=%d | ETA %dm%ds",
                         step, MAX_STEPS, avg_loss, acc, epoch, int(eta // 60), int(eta % 60))
                running_loss = 0
                running_correct = 0
                running_total = 0

            if step % EVAL_STEPS == 0:
                classifier.eval()
                el, ec, et = 0, 0, 0
                with torch.no_grad():
                    for fb, lb in eval_loader:
                        lg = classifier(fb)
                        el += criterion(lg, lb).item()
                        ec += (torch.argmax(lg, dim=-1) == lb).sum().item()
                        et += lb.size(0)
                eval_loss = el / max(len(eval_loader), 1)
                eval_acc = ec / max(et, 1) * 100
                log.info("  >> EVAL step %d: loss=%.4f | acc=%.1f%%", step, eval_loss, eval_acc)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    _save_cls_checkpoint(classifier, hidden_size, step, tag="best", device=device)
                    log.info("  >> New best model!")
                classifier.train()

            if step % SAVE_STEPS == 0:
                _save_cls_checkpoint(classifier, hidden_size, step, device=device)

    # Final save (include normalization stats)
    _save_cls_checkpoint(classifier, hidden_size, step, final=True,
                         extra={"feat_mean": feat_mean, "feat_std": feat_std}, device=device)
    elapsed = time.time() - t0
    log_gpu_memory("training_done", backend)
    log.info("=" * 60)
    log.info("Training complete! %d steps in %dm%ds", MAX_STEPS, int(elapsed // 60), int(elapsed % 60))
    log.info("Best eval loss: %.4f", best_eval_loss)
    log.info("Model saved to: %s", SAVE_DIR)
    log.info("=" * 60)


def _save_cls_checkpoint(classifier, hidden_size, step, final=False, tag=None, extra=None, device=None):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if tag is None:
        tag = "final" if final else f"step_{step}"
    path = SAVE_DIR / f"classifier_{tag}.pt"
    # Always save state_dict on CPU for portability
    cpu_state = {k: v.cpu() for k, v in classifier.state_dict().items()}
    payload = {"state_dict": cpu_state, "hidden_size": hidden_size,
               "num_labels": NUM_LABELS, "labels": LABELS, "step": step}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    log.info("Saved checkpoint: %s (%.1f KB)", path.name, path.stat().st_size / 1e3)
    if final:
        final_path = SAVE_DIR / "classifier.pt"
        torch.save(payload, final_path)
        log.info("Saved final: %s", final_path)


if __name__ == "__main__":
    train()
