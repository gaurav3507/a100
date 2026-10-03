"""CAFE: locate where the FVD of specific ours.csv rows becomes NaN (read-only diagnostic, v1).

Run with the FVD environment's Python from C-MET/evaluation (see the command in the chat):
    python ../../cafe_fvd_nan_diag_v1.py 54 69 0

For each row it rebuilds the exact fvd.py computation (fvd.py's own load_video(), the same
[16, 15, 64, 64, 3] placeholders, frechet_video_distance.preprocess/create_id3_embedding/
calculate_fvd), then inspects every intermediate:
  1. the two I3D embeddings (16 x 400 each): finite, and whether the 16 copies of one video give
     bit-identical rows (fvd.py tiles each video 16 times);
  2. the float64 covariances tfgan builds from them (tfp.stats.covariance, n/(n-1) scaling);
  3. tf.linalg.svd of each covariance, which tfgan uses for its matrix square roots;
  4. tfgan's trace_sqrt_product and the final distance;
  5. a numpy float64 reference of the same formula (eigenvalue route) for comparison.
Nothing is written anywhere; the output is printed.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import fvd as fvd_script  # noqa: E402  (fvd.py of C-MET: load_video, TF1 setup)
import frechet_video_distance as fvdlib  # noqa: E402
import tensorflow.compat.v1 as tf  # noqa: E402
from tensorflow_gan.python.eval import classifier_metrics as cm  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402


def cov64(e):
    x = tf.cast(tf.constant(e), tf.float64)
    n = tf.cast(tf.shape(x)[0], tf.float64)
    return n / (n - 1) * tfp.stats.covariance(x)


def ref_fvd(e1, e2):
    a, b = e1.astype(np.float64), e2.astype(np.float64)
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh((s1 + s1.T) / 2)
    r = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    ev = np.linalg.eigvalsh(r @ s2 @ r)
    return float((m1 - m2) @ (m1 - m2) + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


def main():
    rows = [int(a) for a in sys.argv[1:]] or [54, 69, 0]
    df = pd.read_csv("runs/mead_ours/ours.csv", dtype=str)
    with tf.Graph().as_default():
        ph1 = tf.placeholder(tf.float32, [16, 15, 64, 64, 3])
        ph2 = tf.placeholder(tf.float32, [16, 15, 64, 64, 3])
        e1 = fvdlib.create_id3_embedding(fvdlib.preprocess(ph1, (224, 224)))
        e2 = fvdlib.create_id3_embedding(fvdlib.preprocess(ph2, (224, 224)))
        result = fvdlib.calculate_fvd(e1, e2)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.Session(config=cfg) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            for i in rows:
                gt, gen = df.at[i, "gt_video_path"], df.at[i, df.columns[4]]
                v1, v2 = fvd_script.load_video(gt), fvd_script.load_video(gen)
                b1 = np.tile(v1[np.newaxis, ...], (16, 1, 1, 1, 1))
                b2 = np.tile(v2[np.newaxis, ...], (16, 1, 1, 1, 1))
                r, a, b = sess.run([result, e1, e2], {ph1: b1, ph2: b2})
                print("row %d: %s vs %s" % (i, "/".join(gt.split("/")[-3:]), os.path.basename(gen)))
                print("  fvd.py value            %r" % r)
                for tag, e in (("GT ", a), ("gen", b)):
                    spread = float(np.abs(e - e[0]).max())
                    print("  %s embedding          finite %s, |value| max %.3e, 16 copies bit-identical %s "
                          "(max diff %.3e)" % (tag, bool(np.isfinite(e).all()), float(np.abs(e).max()),
                                               spread == 0.0, spread))
                with tf.Graph().as_default():
                    s1, s2 = cov64(a), cov64(b)
                    sv1, sv2 = tf.linalg.svd(s1), tf.linalg.svd(s2)
                    tsp = cm.trace_sqrt_product(s1, s2)
                    fd = cm.frechet_classifier_distance_from_activations(tf.constant(a), tf.constant(b))
                    with tf.Session() as s2sess:
                        S1, S2, SV1, SV2, TSP, FD = s2sess.run([s1, s2, sv1, sv2, tsp, fd])
                for tag, S, SV in (("GT ", S1, SV1), ("gen", S2, SV2)):
                    fin = [bool(np.isfinite(x).all()) for x in SV]
                    print("  %s covariance float64    max |entry| %.3e, all exactly zero %s; SVD s/u/v finite %s"
                          % (tag, float(np.abs(S).max()), bool((S == 0).all()), fin))
                print("  trace_sqrt_product      %r" % TSP)
                print("  tfgan distance (rerun)  %r" % FD)
                print("  numpy float64 reference %.6f" % ref_fvd(a, b))


if __name__ == "__main__":
    main()
