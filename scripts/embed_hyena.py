"""Extract HyenaDNA-tiny embeddings for one TF and cache them to disk.

Writes data/processed/{TF}_hyena.h5 with, for each split:
  x_{split}       full embeddings (N, 202, 256) in float16, for the authors' CNN
  x_mean_{split}  mean over the 200 base tokens (N, 256) in float32, for logistic regression
  y_{split}       labels
The authors stored float32; float16 halves the size (about 9 GB for CTCF) and changes
values by less than 0.1%.

Usage:
    python scripts/embed_hyena.py --tf CTCF
"""

import argparse
import time

import h5py
import numpy as np

from ctcf.data import ROOT, SPLITS, load_onehot
from ctcf.hyena import embed, load_model

CHUNK = 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tf", default="CTCF")
    args = parser.parse_args()

    model = load_model()
    out_path = ROOT / "data" / "processed" / f"{args.tf}_hyena.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".h5.part")
    with h5py.File(tmp, "w") as out:
        for split in SPLITS:
            x, y = load_onehot(split, ROOT / "data" / "raw" / f"{args.tf}_200.h5")
            full = out.create_dataset(f"x_{split}", (len(y), x.shape[2] + 2, 256), dtype="float16",
                                      chunks=(64, x.shape[2] + 2, 256))
            mean = out.create_dataset(f"x_mean_{split}", (len(y), 256), dtype="float32")
            out.create_dataset(f"y_{split}", data=y)
            start = time.time()
            for i in range(0, len(y), CHUNK):
                e = embed(model, x[i:i + CHUNK])
                full[i:i + len(e)] = e.astype(np.float16)
                mean[i:i + len(e)] = e[:, 1:-1].mean(axis=1)  # skip [CLS] and [SEP]
                done = min(i + CHUNK, len(y))
                print(f"  {args.tf} {split}: {done}/{len(y)} ({time.time() - start:.0f} s)", flush=True)
    tmp.replace(out_path)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
