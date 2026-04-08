"""CLI interface for audio project pipeline."""
import argparse
from pathlib import Path
from audio_project.gender import GENDER_FEATURE_COLUMNS, predict_gender, train_gender_model
from audio_project.core import load_model, predict_file, prepare_processed_audio, run_training_pipeline


def _print_result(r, prefix=""):
    print(f"{prefix}Model saved to: {r.model_path}")
    print(f"Train/Test: {r.train_size}/{r.test_size}\n{r.report}")


def main() -> None:
    p = argparse.ArgumentParser(description="Audio project CLI")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    sp.add_parser("train")
    ip = sp.add_parser("infer"); ip.add_argument("--file", type=Path, required=True)
    sp.add_parser("gender-train")
    gp = sp.add_parser("gender-infer")
    for f in GENDER_FEATURE_COLUMNS: gp.add_argument(f"--{f}", type=float, required=True)
    sp.add_parser("all")
    args = p.parse_args()

    if args.cmd == "prepare":
        print(f"Prepared {prepare_processed_audio()} files.")
    elif args.cmd == "train":
        _print_result(run_training_pipeline())
    elif args.cmd == "infer":
        print(f"Prediction: {predict_file(args.file, load_model())}")
    elif args.cmd == "gender-train":
        _print_result(train_gender_model(), "Gender ")
    elif args.cmd == "gender-infer":
        print(f"Gender: {predict_gender({f: getattr(args, f) for f in GENDER_FEATURE_COLUMNS})}")
    elif args.cmd == "all":
        print(f"Prepared {prepare_processed_audio()} files.")
        _print_result(run_training_pipeline())


if __name__ == "__main__":
    main()
