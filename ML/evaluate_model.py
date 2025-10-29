"""
Evaluate a trained model (Keras H5 or ONNX) on a dataset directory.

Features:
- Supports model_v2.h5 (Keras) or model_v2.onnx (ONNX Runtime)
- Computes: accuracy, top-3 accuracy, confusion matrix, precision/recall/F1
- Plots: confusion matrix; per-class ROC curves
- Displays misclassified samples (paths with actual vs predicted and confidence)
- Saves a summary report to evaluation_report.txt

Usage:
  python ML/evaluate_model.py --model ./best_model.h5 --data ./data --out ./eval_out
  python ML/evaluate_model.py --model ./backend/model_v2.onnx --data ./data --out ./eval_out --limit 200
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

import tensorflow as tf


def infer_input_size_from_keras(model) -> Tuple[int, int]:
    try:
        ishape = model.input_shape
        # Expect (None, H, W, C)
        h = int(ishape[1]) if isinstance(ishape[1], int) else 224
        w = int(ishape[2]) if isinstance(ishape[2], int) else 224
        return (h, w)
    except Exception:
        return (224, 224)


def infer_input_size_from_onnx(session) -> Tuple[int, int]:
    try:
        inp = session.get_inputs()[0]
        shape = inp.shape
        ints = [d for d in shape if isinstance(d, int)]
        if len(ints) >= 2:
            return (int(ints[0]), int(ints[1])) if len(shape) == 2 else (int(ints[0]), int(ints[1]))
    except Exception:
        pass
    return (224, 224)


def load_class_labels(labels_path: Path, data_dir: Path) -> List[str]:
    if labels_path.exists():
        with labels_path.open('r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback: infer from data subfolders
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


def list_dataset_images(data_dir: Path, class_names: List[str], limit: int | None) -> Tuple[List[Path], np.ndarray]:
    file_paths: List[Path] = []
    labels: List[int] = []
    for idx, cname in enumerate(class_names):
        cdir = data_dir / cname
        if not cdir.exists():
            continue
        imgs = []
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            imgs.extend(list(cdir.glob(f'*{ext}')))
            imgs.extend(list(cdir.glob(f'*{ext.upper()}')))
        if limit is not None:
            imgs = imgs[:max(0, limit // max(1, len(class_names)))]
        for p in imgs:
            file_paths.append(p)
            labels.append(idx)
    return file_paths, np.array(labels, dtype=np.int64)


def preprocess_pil(img: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # per-image standardization (as training/inference)
    t = tf.convert_to_tensor(arr)
    t = tf.image.per_image_standardization(t)
    return t.numpy()


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]


def evaluate_keras(model_path: Path, files: List[Path], labels: np.ndarray, target_size: Tuple[int, int], class_names: List[str], out_dir: Path):
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    # If model input differs, override target size
    target_size = infer_input_size_from_keras(model)

    y_true = labels.copy()
    probs_list = []
    BATCH = 32

    for batch_files in tqdm(list(batched(files, BATCH)), desc="Keras inference", unit="batch"):
        batch_arr = [preprocess_pil(Image.open(p), target_size) for p in batch_files]
        x = np.stack(batch_arr, axis=0)
        preds = model.predict(x, verbose=0)
        probs_list.append(preds)

    probs = np.concatenate(probs_list, axis=0)
    return compute_and_report_metrics(probs, y_true, files, class_names, out_dir)


def evaluate_onnx(model_path: Path, files: List[Path], labels: np.ndarray, target_size: Tuple[int, int], class_names: List[str], out_dir: Path):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    except Exception:
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    # If model input differs, override target size
    target_size = infer_input_size_from_onnx(session)

    y_true = labels.copy()
    probs_list = []
    BATCH = 32

    for batch_files in tqdm(list(batched(files, BATCH)), desc="ONNX inference", unit="batch"):
        batch_arr = [preprocess_pil(Image.open(p), target_size) for p in batch_files]
        x = np.stack(batch_arr, axis=0)
        preds = session.run(None, {input_name: x})[0]
        probs_list.append(preds)

    probs = np.concatenate(probs_list, axis=0)
    return compute_and_report_metrics(probs, y_true, files, class_names, out_dir)


def compute_topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    correct = 0
    for i in range(y_true.shape[0]):
        if y_true[i] in topk[i]:
            correct += 1
    return correct / max(1, y_true.shape[0])


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_roc_per_class(probs: np.ndarray, y_true: np.ndarray, class_names: List[str], out_dir: Path):
    # One-vs-rest ROC per class
    n_classes = len(class_names)
    y_true_bin = np.eye(n_classes, dtype=np.int32)[y_true]

    for c in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], probs[:, c])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC - {class_names[c]}')
            ax.legend(loc="lower right")
            out_path = out_dir / f"roc_{c:02d}_{class_names[c].replace(' ', '_')}.png"
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            continue


def compute_and_report_metrics(probs: np.ndarray, y_true: np.ndarray, files: List[Path], class_names: List[str], out_dir: Path):
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    top3 = compute_topk_accuracy(probs, y_true, k=3)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    # Save confusion matrix plot
    cm_path = out_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path)

    # Save per-class ROC plots
    plot_roc_per_class(probs, y_true, class_names, out_dir)

    # Misclassified samples
    mis_idx = np.where(y_pred != y_true)[0]
    mis_samples = []
    for i in mis_idx[:50]:  # cap display to 50
        conf = float(probs[i, y_pred[i]])
        mis_samples.append((str(files[i]), class_names[y_true[i]], class_names[y_pred[i]], conf))

    # Write evaluation report
    report_path = out_dir / 'evaluation_report.txt'
    with report_path.open('w', encoding='utf-8') as f:
        f.write("EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Samples evaluated: {len(y_true)}\n")
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Top-3 Accuracy: {top3:.4f} ({top3*100:.2f}%)\n\n")

        f.write("Classification Report:\n")
        f.write(report + "\n")

        f.write(f"Confusion matrix image: {cm_path}\n")
        f.write("ROC plots saved per class in output directory.\n\n")

        if mis_samples:
            f.write("Misclassified Samples (up to 50):\n")
            for path, truth, pred, conf in mis_samples:
                f.write(f"  {path} | true={truth} | pred={pred} | conf={conf:.4f}\n")
        else:
            f.write("No misclassified samples.\n")

    # Also print a short console summary
    print("\nEvaluation Summary:")
    print(f"  Samples: {len(y_true)}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Top-3 Accuracy: {top3:.4f} ({top3*100:.2f}%)")
    print(f"  Report saved: {report_path}")
    print(f"  Confusion matrix: {cm_path}")

    return {
        'accuracy': acc,
        'top3': top3,
        'cm_path': str(cm_path),
        'report_path': str(report_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Keras or ONNX model")
    parser.add_argument('--model', type=str, required=True, help='Path to model .h5 or .onnx')
    parser.add_argument('--data', type=str, default='../data', help='Path to dataset root (class folders)')
    parser.add_argument('--labels', type=str, default='../backend/class_labels.txt', help='Path to class_labels.txt (optional)')
    parser.add_argument('--out', type=str, default='./evaluation', help='Output directory for plots and report')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for quick evaluation')
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    data_dir = (Path(__file__).resolve().parent / args.data).resolve() if not os.path.isabs(args.data) else Path(args.data)
    labels_path = (Path(__file__).resolve().parent / args.labels).resolve() if not os.path.isabs(args.labels) else Path(args.labels)
    out_dir = (Path(__file__).resolve().parent / args.out).resolve() if not os.path.isabs(args.out) else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    class_names = load_class_labels(labels_path, data_dir)
    if not class_names:
        raise RuntimeError("No class names found; ensure labels file or class folders exist")

    # List dataset images and labels
    files, labels = list_dataset_images(data_dir, class_names, args.limit)
    if not files:
        raise RuntimeError("No images found in dataset")

    target_size = (224, 224)
    if model_path.suffix.lower() == '.h5':
        evaluate_keras(model_path, files, labels, target_size, class_names, out_dir)
    elif model_path.suffix.lower() == '.onnx':
        evaluate_onnx(model_path, files, labels, target_size, class_names, out_dir)
    else:
        raise ValueError("Unsupported model format. Use .h5 or .onnx")


if __name__ == '__main__':
    main()
