"""Tek bir CWRU .mat dosyasını pencerelere böler, FELA-PdM modeline verir
ve sağlıksız pencere oranını yazdırır.

Kullanım:
    .venv\\Scripts\\python.exe predict_file.py
    (program "Dosya yolu:" diye sorar, yolu ayrı satıra yazın)

    ya da tek satırda:
    .venv\\Scripts\\python.exe predict_file.py data\\cwru\\IR007_0.mat
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "fela-pdm"
sys.path.insert(0, str(MODEL_DIR))
from modeling import load_model, preprocess_cwru  # noqa: E402

WINDOW = 2048  # modelin beklediği pencere uzunluğu (örnek sayısı)
STRIDE = 1024  # eğitimdeki gibi %50 örtüşme
BATCH = 64  # modele aynı anda verilen pencere sayısı
HEALTHY_CLASS = 0  # model çıktısında sınıf 0 = Normal (sağlıklı)

# Sağlıksız pencere oranına (%) göre seviye bantları: (üst sınır, seviye adı)
LEVELS = [
    (10, "Normal"),
    (30, "İzlenmeli (hafif belirti)"),
    (60, "Olası arıza"),
    (90, "Güçlü arıza belirtisi"),
    (float("inf"), "Çok güçlü arıza belirtisi"),
]

# Arıza türü -> model sınıf numaraları (küçük/orta/büyük hasar boyutları birleştirildi)
FAULT_TYPES = {
    "İç halka": (1, 4, 7),
    "Bilya": (2, 5, 8),
    "Dış halka": (3, 6, 9),
}
TYPE_MIN_PERCENT = 30  # arıza türü, "Olası arıza" seviyesinden itibaren gösterilir


def load_signal(path):
    """.mat dosyasından sürücü ucu (DE) titreşim sinyalini okur."""
    mat = sio.loadmat(path)
    keys = [k for k in mat if k.endswith("DE_time")]
    if not keys:
        raise SystemExit(f"{path}: 'DE_time' sinyali bulunamadı.")
    return mat[keys[0]].ravel().astype(np.float32)


def make_windows(signal):
    """Uzun sinyali WINDOW uzunluğunda, STRIDE kaydırmalı pencerelere böler."""
    if len(signal) < WINDOW:
        raise SystemExit(
            f"Sinyal çok kısa: {len(signal)} örnek, en az {WINDOW} gerekli."
        )
    starts = range(0, len(signal) - WINDOW + 1, STRIDE)
    return [signal[s: s + WINDOW] for s in starts]


def predict_windows(model, windows):
    """Her pencere için modelin seçtiği sınıf numarasını döndürür."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(windows), BATCH):
            chunk = windows[i: i + BATCH]
            batch = torch.cat([preprocess_cwru(w) for w in chunk], dim=0)
            logits = model(batch, task="cls")
            preds.append(logits.argmax(dim=-1))
    return torch.cat(preds)


def get_level(percent):
    """Sağlıksız pencere oranına (%) karşılık gelen seviye adını döndürür."""
    for upper, name in LEVELS:
        if percent < upper:
            return name


def dominant_fault_type(preds):
    """Sağlıksız pencereler arasında en çok seçilen arıza türünü ve payını (%) döndürür."""
    unhealthy = preds[preds != HEALTHY_CLASS]
    counts = {
        name: int(torch.isin(unhealthy, torch.tensor(ids)).sum())
        for name, ids in FAULT_TYPES.items()
    }
    name = max(counts, key=counts.get)
    return name, 100 * counts[name] / len(unhealthy)


def analyze_file(mat_path):
    """Verilen .mat dosyasını analiz eder ve sonuçları sözlük olarak döndürür.
     Mevcut fonksiyonları sırayla çalıştırır:
    - load_signal()
    - make_windows()
    - load_model()
    - predict_windows()
    - get_level()
    """

    path = Path(mat_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    signal = load_signal(path)
    windows = make_windows(signal)
    model = load_model(str(MODEL_DIR), variant="cwru")
    preds = predict_windows(model, windows)

    n_total = len(preds)
    n_unhealthy = int((preds != HEALTHY_CLASS).sum())
    percent = 100 * n_unhealthy / n_total
    level = get_level(percent)

    fault = None
    share = None

    if percent >= TYPE_MIN_PERCENT:
        fault, share = dominant_fault_type(preds)

    return {
        "file_name": path.name,
        "n_total": n_total,
        "n_unhealthy": n_unhealthy,
        "n_healthy": n_total - n_unhealthy,
        "percent": percent,
        "level": level,
        "fault_name": fault,
        "fault_share": share,
        "preds": preds.cpu().numpy(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "mat_file",
        type=Path,
        nargs="?",
        help="CWRU .mat dosyasının yolu (verilmezse program sorar)",
    )
    args = ap.parse_args()

    if args.mat_file is None:
        raw = input("Dosya yolu: ").strip().strip('"')
        if not raw:
            raise SystemExit("Dosya yolu girilmedi.")
        args.mat_file = Path(raw)

    if not args.mat_file.is_file():
        raise SystemExit(f"Dosya bulunamadı: {args.mat_file}")

    result = analyze_file(args.mat_file)

    print(f"Dosya                   : {result['file_name']}")
    print(f"Sağlıksız pencere oranı : %{result['percent']:.1f}")
    print(f"Seviye                  : {result['level']}")

    if result["fault_name"] is not None:
        print(
            f"En olası arıza türü     : {result['fault_name']} "
            f"[sağlıksız pencerelerdeki payı: %{result['fault_share']:.1f}]"
        )


if __name__ == "__main__":
    main()
