#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAFE Gate 2: evaluation asset placement and readiness check (v2).

Run from the C-MET repository root (the folder that contains inference.py),
with the C_MET conda environment active:

    python3 ../cafe_eval_assets_v2.py

The script is idempotent and safe to re-run. It never moves an unvalidated
file into the evaluation tree, and it keeps (never deletes) any non-empty
file it replaces.

Checks, in order:
  1. Code provenance: every tracked file under evaluation/ is byte-identical
     to C-MET commit 44ccf3a (SHA-256 manifest embedded below).
  2. Pinned upstream files: face_align_cuda.py and the two dlib models come
     from Open-Debin/Emotion-FAN at a pinned commit, the source that the C-MET
     evaluation README names for missing Emotion-FAN files. Each file is
     verified by SHA-256 before placement.
  3. Checkpoints: syncnet_v2.model, Emotion-FAN_MEAD.pth,
     Resnet18_FER+_pytorch.pth.tar, sfd_face.pth and the FID Inception
     weights. Uploads are searched for in this script's folder and one level
     below. Each file is validated by loading it the way the evaluation code
     does, and only then moved into place.
  4. Packages imported by every evaluation step, the interpreter that
     subprocess calls to `python` resolve to, CUDA, and the exact media
     commands of run_pipeline.py (including the OpenCV XVID writer and the
     audio cut and mux whose failure drops into pdb), SyncNetInstance.py and
     vide2frame_custom.py, run on one generated video.
  5. Functional checks: the scene_detect() sequence of run_pipeline.py and the
     MFCC sequence of SyncNetInstance.py on media derived from one generated
     video; dlib model loads; a face-crop smoke test that decodes one generated
     video with OpenCV and runs the exact frame2face_custom.py command; the FID
     InceptionV3 built from the cached weights.

v2 adds the two run_pipeline.py/SyncNetInstance.py functional checks: an import
test alone passes scenedetect 0.5.1, which then crashes inside ContentDetector
with OpenCV >= 4.5.4 (cv2.split returns a tuple).

A JSON record (paths, sizes, SHA-256, versions) is written to
<script folder>/reports/ for the reproducibility audit.
Exit code: 0 when nothing is FAIL or MISSING, 1 otherwise, 2 on bad usage.
"""
from __future__ import print_function

import bz2
import datetime
import glob
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request

SCRIPT = "cafe_eval_assets_v2"
CMET_COMMIT = "44ccf3a88cf46b4db56755ca93f2fad12a660fbb"
EFAN_COMMIT = "874e871999a2002cd5dd9dffff2c4400c2e1805b"
EFAN_LIB_URL = ("https://raw.githubusercontent.com/Open-Debin/Emotion-FAN/%s/"
                "data/face_alignment_code/lib/" % EFAN_COMMIT)
OXFORD_HTTPS = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/"
FID_NAME = "pt_inception-2015-12-05-6726825d.pth"
FID_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/" + FID_NAME
FID_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
# SHA-256 of the canonical files as mirrored, identically, by ByteDance/LatentSync-1.6 and
# chunyu-li/LatentSync on Hugging Face. Used only to label a file (OK vs WARN), never to reject one.
SYNCNET_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"  # 54573114 B
SFD_SHA256 = "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491"  # 89844381 B
TORCH_RECORDED = "2.1.0"
# scenedetect 0.5.6.1 = 0.5.1 plus the OpenCV >= 4.5.4 tuple fix; per-frame ContentDetector scores and
# scene lists were verified identical to a tuple-patched 0.5.1 (single-shot clip and hard-cut clip).
SCENEDETECT_OK = ("0.5.1", "0.5.6.1")

EVAL = "evaluation"
LIB = EVAL + "/Emotion-FAN/data/face_alignment_code/lib"
GEN_GLOB = EVAL + "/runs/mead_cmet_repro/gen/*.mp4"
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# SHA-256 of every file tracked under evaluation/ at C-MET commit 44ccf3a.
EVAL_MANIFEST = {
    "evaluation/Emotion-FAN/LICENSE": "3ac3ec2921d890285b49786b30d5c0fd9f11bc9ed470ecdbadf2d5c4437715e9",
    "evaluation/Emotion-FAN/README.md": "c07cf651c7c19cdbde21481df820c9a907c834b66545d3fb99fb8e435d325731",
    "evaluation/Emotion-FAN/all.py": "6410df41cffa177df49e8e475424a37f62d469eec981eeeb5439020adbf42440",
    "evaluation/Emotion-FAN/all_crema.py": "c8a633776fe53d6b536aae7ead882a2795a39dcf71db60e885c921b5c8257706",
    "evaluation/Emotion-FAN/all_emo.py": "15d3351594366d92f7abdaa57969d17e6a56a1e3e930ec5cc7ae4ab13aee4842",
    "evaluation/Emotion-FAN/all_rav.py": "9191cec4aa6c7ab25e771b5540baa8bca7013d8e12c3af91a88c3e2b9f540900",
    "evaluation/Emotion-FAN/baseline_afew.py": "b9528d406f5a750e50e895e8f7735d3cc10bf8eece26049b711f5dc869817076",
    "evaluation/Emotion-FAN/baseline_ck_plus.py": "cb7c8af7afd9fcd4c5379c2457774f4ea3ece1199a2ece35dacfc8d54508d1c0",
    "evaluation/Emotion-FAN/basic_code/README.md": "4103a0379ecfb3c5aae08a40ddc6117b3c2a8c7ddc9cb7fe0e2b52107e46e7ee",
    "evaluation/Emotion-FAN/basic_code/data_generator.py": "f7c1910a9f703e8a35a737ad5f1a536b444680b402ae394f13ddc03e3e7078be",
    "evaluation/Emotion-FAN/basic_code/load.py": "deaf1b0bbe4fc8e5d29eded05df8b3249141954708c38d16d90a3d516c617db8",
    "evaluation/Emotion-FAN/basic_code/networks.py": "8c1f86ba30f131f671f053568129d49021b2df1891f5de5f96e97834203b6512",
    "evaluation/Emotion-FAN/basic_code/util.py": "8b07d1d29b345069c933436d2e48ca22c62aaf9bf360820c207f97ee443f60d1",
    "evaluation/Emotion-FAN/emotion-fan.py": "a28bf2aaf06689fe0db2eb5d64cdcc968eb02c99199e7f25e9f582852d170a06",
    "evaluation/Emotion-FAN/emotion-fan_crema.py": "1fb4d1da3a2fe928a5238ac8785ca1f3c2c381f058f8a945f194135569a2eb90",
    "evaluation/Emotion-FAN/emotion-fan_ravdess.py": "b9c51d5e37e275df9906fc400097ed40ea0c11dbf1e53519f2e4d61be3d25ab6",
    "evaluation/Emotion-FAN/emotion_finetune.py": "c21b56186182271e2229efbb1699fe4f7137acf1a4fe8d07143ed4b3accb95da",
    "evaluation/Emotion-FAN/fan_afew_traintest.py": "9fa4feefe051ebec0e4580ca5bfd90407d78ac79f6d32a755862b91ab93ea74e",
    "evaluation/Emotion-FAN/fan_ckplus_traintest.py": "15fead6c9cc8be7ddee338d84ec3f6a717cd179e8cfe463a162cfdc1c0eab26f",
    "evaluation/Emotion-FAN/print_acc.py": "81bc7774fb19b4d5c9cba1fa23c159bdc31deb9e08bb351d93d1d3390d1d6c35",
    "evaluation/Emotion-FAN/requirements.txt": "2f90100978e7df19e12ccb757230da279ba39f9fd5d9a81cf7d8c6d69a9bb391",
    "evaluation/Emotion-FAN/split.py": "0442f026b6ceb30aee4bccf62cd013d66950059c8979a9ee3bb8c51700a23786",
    "evaluation/README.md": "a586364e12a88674199823692c03dbc2a2ff4908c5d653dfe68c27cd216e7156",
    "evaluation/all_fid.py": "40fef9333f86f6334e4392bc09c575cbb0cde53c930dbeb2fc7e40800207eee9",
    "evaluation/check_quantitative_all.py": "c4412bc282922db16a0bb55dae319b971ed7c5837fc8b10c17c02e4901a3a11e",
    "evaluation/common_metrics_on_video_quality_NOTICE.md": "0b02d94f973197649c85dc4ff867f862d0c1248caa550af1adea6604443b82e9",
    "evaluation/copy_csv.py": "b153a0c257220caeab373a2f4c83666b0af83baf7d1b65e3fc27c6c4a9324c74",
    "evaluation/fid.py": "69b48df135ee42d6506a03c9c8534995859bfaee92c59aacc8217fd82026c954",
    "evaluation/frame2face_custom.py": "f47bca36e466fba1cd66b4eb8a694e4b2231eb64d9e4d957a4e9b9eaa7f05305",
    "evaluation/frechet_video_distance.py": "2501cd28a80f061cd2b31ab4b0eb0cadb5abc6dec2b9cf1ecd048ed29ac7de6c",
    "evaluation/fvd.py": "db1a4af5823d2469c1c210190d48564b82fb959be4a137cc5a6fbdebe7741723",
    "evaluation/google-research/frechet_video_distance/README.md": "9b80bcf4ae0b18909b317bdacfe40835a0e2b1f9761ec295c8ded7397d40c6ce",
    "evaluation/google-research/frechet_video_distance/__init__.py": "a0619d42a60eef2f92156dba3abdeb211ee6c86eec43de4177bdc22116113e24",
    "evaluation/google-research/frechet_video_distance/example.py": "c525664dd93b7438336ab769f8463fe020c6e725d68af36e1144953bc0a54537",
    "evaluation/google-research/frechet_video_distance/frechet_video_distance.py": "b83a60da58b2a155af46dc14327818b18aed86288d63edf44b647e5a86fb6ddd",
    "evaluation/google-research/frechet_video_distance/fvd_bair.png": "32633ac3d8e106526709e0b594b19d49901ce474dbd04122cb0ed77c2e9c9503",
    "evaluation/make_train.py": "06c5c9c7ecfe3779601925a33d825a2de514b80e68023cbdd54fa013279842b8",
    "evaluation/pytorch-fid/.flake8": "b0e2e5aabf53633e3439fa5079ff1f9c2e6e4302d96efa5540087ef66cd5e2d6",
    "evaluation/pytorch-fid/CHANGELOG.md": "389484f46ab9c09949e8bd08e4187bde965af36ffe1b01d39c2e80ad0d1d1476",
    "evaluation/pytorch-fid/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "evaluation/pytorch-fid/README.md": "e9253b61d4585665827162817a268f00b4d1d97f653b04232ee47f390a9c5799",
    "evaluation/pytorch-fid/custom.py": "077fedb8b9a2ef8d0561c918cb5cf6513f8eb21bfb12178d6c5a244f0c535664",
    "evaluation/pytorch-fid/noxfile.py": "62db728447fef8279b4217e18bc65fdd76d812e02e5ae44b62314d7bb9dab349",
    "evaluation/pytorch-fid/pyproject.toml": "9dea70772699363f16edbcedeab84658f8ccf59045415d3029e6ee9472957fca",
    "evaluation/pytorch-fid/setup.py": "6aad583da2530698ee47aa47bac2ef62773e8dcb0e688d9e9ae732ac21d63242",
    "evaluation/pytorch-fid/src/pytorch_fid/__init__.py": "56b5e91c3bb77ab933c25fd65eba883419bdc5691cc94cb9dc840e8f4e3674eb",
    "evaluation/pytorch-fid/src/pytorch_fid/__main__.py": "238ba52926d7539d3831216d6f7edf2fefac1779d1f5df578803f766167da2f9",
    "evaluation/pytorch-fid/src/pytorch_fid/fid_score.py": "2dddf8b46e354efcb068537d85423339822b2a34822b00c9e09ea9bc950b6e61",
    "evaluation/pytorch-fid/src/pytorch_fid/inception.py": "c6183fff54dd240fe66d53d207f4bd28c06fde98c21b5525f10ca0cc5cef7780",
    "evaluation/pytorch-fid/tests/test_fid_score.py": "bfecd5cd5e1a4989a53352a19fc30137b8bce52800381b1916f7f1a8b31c8dd0",
    "evaluation/syncnet_python/.gitignore": "5d69bb06cd4ee977460f17c517fe9c2e298a9e372a541e8b23b477035a29b11b",
    "evaluation/syncnet_python/LICENSE.md": "70155ceb20fb7aae10fc1e9d5c9753de6a10a50a8937e0f5ab3afcdf96c8754d",
    "evaluation/syncnet_python/README.md": "bbaefce22c323d933c222e520dc84dade359d01c2e4cc2749d4129dd2abcce68",
    "evaluation/syncnet_python/SyncNetInstance.py": "4bfb7ebbef0a7ceecb1cb7c5ed4bd2472f302d12a300b586fc9d12ad98c59359",
    "evaluation/syncnet_python/SyncNetModel.py": "93e47d993bbf702ab8bb5223771931ea964eb6804787d35ac086885971e1f20c",
    "evaluation/syncnet_python/all_pipeline.py": "3e4f3339569d744596298463132d82573554eeb5d9711497878add9e91d3d8d9",
    "evaluation/syncnet_python/all_syncnet.py": "ae6da62b528d4859c082073d77ef518fc918a8a67d54784a69df01a1d37719d8",
    "evaluation/syncnet_python/conf_mean.py": "a266bd819abadbe6e6ff7d1691fbb833518c6dc76c6553eda5e02406f9eae337",
    "evaluation/syncnet_python/demo_feature.py": "956c1f9e1e460c0243f0843a7940a5034d67f00e4a91ffa329845c87c58d0d94",
    "evaluation/syncnet_python/demo_syncnet.py": "be8c4a29da5205242d80c92ba9fa10162d762e23370004873e2d78d4197da9f4",
    "evaluation/syncnet_python/detectors/README.md": "7856aa251e81304b32c5fff167d17f5a474eef4d20ef7a1137ddad995b23a2a0",
    "evaluation/syncnet_python/detectors/__init__.py": "1f3f1baac2c3ce57d4190ea351da93648abfd8e4f6e361b21201d8f31686bf18",
    "evaluation/syncnet_python/detectors/s3fd/__init__.py": "7326ced0828156c46fce03580cfc1ea3d309a8318a79af607cc45bafb0fa9db1",
    "evaluation/syncnet_python/detectors/s3fd/box_utils.py": "b0b81afdc7837b26823dc3a556d40aee7cb438d775be090ae2c0441b35a9d986",
    "evaluation/syncnet_python/detectors/s3fd/nets.py": "3e9dad22a2f1234014b46e4f96f46d5ff578d524eb332d063a69a476924f0ee5",
    "evaluation/syncnet_python/download_model.sh": "3d3d63d1e6cda76192ca9d045ac80a89626db4bc1c2e0937e8d976dba055a26f",
    "evaluation/syncnet_python/read_pckl.py": "5861d467fca47a57133d08d2e0ae43fb59c7f9e26f8ed289b4d931f80c31c3a7",
    "evaluation/syncnet_python/requirements.txt": "d0df4591b838b1590998f69579fbdc34c8c68e48e97a21eb205e6785af1db089",
    "evaluation/syncnet_python/run_pipeline.py": "a550ee82091a16d475755c9024140f8ceab8d7d3d8d2a8244dfcd5981a7c3b1a",
    "evaluation/syncnet_python/run_syncnet.py": "cc33844fcd93d146bfac12e0ed16d040749078939918eaf6cf925930d0d7dc69",
    "evaluation/syncnet_python/run_visualise.py": "c14f86833c57e6796fe7bfbb97cf4a3ef502da6a15bc97397da257ccfa3764c1",
    "evaluation/test.py": "1a96e38e63a7fd1b7ad6347426370c487072dce8eb7ace75d672484477d05da4",
    "evaluation/vide2frame_custom.py": "e0291d4e2ce5fbeb14d316a0731eadf3eeb6b0ed2ea06e378dfd8b825fd3860a"
}

PINNED = [
    {"label": "face_align_cuda.py", "dest": LIB + "/face_align_cuda.py",
     "sha256": "69c1169ea4a3467e6506e122a625ab33117c69e61fcfa821f7478b4009823176",
     "patterns": ["face_align_cuda*.py"]},
    {"label": "shape_predictor_5 (dlib)", "dest": LIB + "/shape_predictor_5_face_landmarks.dat",
     "sha256": "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888",
     "patterns": ["shape_predictor_5_face_landmarks.dat*"]},
    {"label": "mmod_human_face_detector", "dest": LIB + "/mmod_human_face_detector.dat",
     "sha256": "4cb19393e2fbaf2b1609a9319ad5386618c886a6234ec1b971f3e87c85d87fe6",
     "patterns": ["mmod_human_face_detector.dat*"]},
]

# (import name, distribution name, evaluation step that imports it)
PACKAGES = [
    ("numpy", "numpy", "all steps"),
    ("pandas", "pandas", "all steps"),
    ("torch", "torch", "FID, Acc_emo, Sync_conf"),
    ("torchvision", "torchvision", "FID, Acc_emo, S3FD"),
    ("PIL", "Pillow", "FID, Acc_emo"),
    ("scipy", "scipy", "FID, Sync_conf"),
    ("tqdm", "tqdm", "FID, Acc_emo, Sync_conf"),
    ("cv2", "opencv-python", "face crops, Sync_conf"),
    ("natsort", "natsort", "Acc_emo (emotion-fan.py)"),
    ("sklearn", "scikit-learn", "summary (check_quantitative_all.py)"),
    ("scenedetect", "scenedetect", "Sync_conf (run_pipeline.py)"),
    ("python_speech_features", "python_speech_features", "Sync_conf (SyncNetInstance.py)"),
    ("dlib", "dlib", "face crops (face_align_cuda.py)"),
    ("pytorch_fid", "pytorch-fid", "FID (pytorch-fid/custom.py)"),
]


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, status, group, name, detail="", **extra):
        row = {"status": status, "group": group, "name": name, "detail": detail}
        row.update(extra)
        self.rows.append(row)
        print("[%-7s] %-5s %-28s %s" % (status, group, name, detail))
        sys.stdout.flush()
        return row

    def count(self, status):
        return sum(1 for r in self.rows if r["status"] == status)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def looks_like_html(path):
    try:
        with open(path, "rb") as f:
            head = f.read(512).lstrip().lower()
    except Exception:
        return False
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head


def err_text(e, limit=170):
    lines = [ln.strip() for ln in str(e).strip().splitlines() if ln.strip()]
    text = lines[0] if lines else ""
    if text.endswith(":") and len(lines) > 1:  # load_state_dict: keep the lines that say what is wrong
        text = "%s %s" % (text, " / ".join(ln[:limit] for ln in lines[1:3]))
        limit *= 2
    return "%s: %s" % (type(e).__name__, text[:limit])


def fail_detail(path, e):
    msg = err_text(e)
    if looks_like_html(path):
        msg = "file is an HTML page, not a checkpoint (re-download); " + msg
    return msg


def remove_quiet(path):
    try:
        if os.path.lexists(path):
            os.remove(path)
    except Exception:
        pass


def find_uploads(upload_dir, patterns):
    seen, found = set(), []
    for pat in patterns:
        hits = glob.glob(os.path.join(upload_dir, pat)) + glob.glob(os.path.join(upload_dir, "*", pat))
        for p in hits:
            rp = os.path.realpath(p)
            if rp in seen or not os.path.isfile(p):
                continue
            seen.add(rp)
            found.append(p)
    found.sort(key=os.path.getmtime, reverse=True)
    return found


def set_aside(dest, tag):
    """Move an existing destination file out of the way; 0-byte files are removed."""
    if not os.path.lexists(dest):
        return ""
    if os.path.isfile(dest) and os.path.getsize(dest) == 0:
        os.remove(dest)
        return "; removed 0-byte file"
    kept = "%s.%s_%s" % (dest, tag, STAMP)
    os.replace(dest, kept)
    return "; previous file kept as " + os.path.basename(kept)


def move_into_place(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        os.replace(src, dest)
    except OSError:
        part = "%s.part_%d" % (dest, os.getpid())
        shutil.copy2(src, part)
        os.replace(part, dest)
        os.remove(src)


def fetch(url, out_path, timeout=60):
    errors = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": SCRIPT})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(out_path, "wb") as f:
            shutil.copyfileobj(resp, f, 1 << 20)
        if os.path.getsize(out_path) > 0:
            return True, "urllib"
        errors.append("urllib: empty body")
    except Exception as e:
        errors.append("urllib " + err_text(e))
        if isinstance(e, socket.timeout) or "timed out" in str(e):
            remove_quiet(out_path)
            return False, " | ".join(errors)
    curl = shutil.which("curl")
    if curl:
        try:
            res = subprocess.run([curl, "-fsSL", "--max-time", str(timeout), "-o", out_path, url],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=timeout + 30)
            if res.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                return True, "curl"
            errors.append("curl rc=%d %s" % (res.returncode, res.stdout.decode("utf-8", "replace").strip()[:120]))
        except Exception as e:
            errors.append("curl " + err_text(e))
    remove_quiet(out_path)
    return False, " | ".join(errors)


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_S3FD_PKG = []


def load_s3fd_package(repo):
    """Import evaluation/syncnet_python/detectors/s3fd as a package (nets.py uses relative imports)."""
    if _S3FD_PKG:
        return _S3FD_PKG[0]
    pkg_dir = os.path.join(repo, EVAL, "syncnet_python", "detectors", "s3fd")
    spec = importlib.util.spec_from_file_location(
        "cafe_s3fd", os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cafe_s3fd"] = mod
    spec.loader.exec_module(mod)
    _S3FD_PKG.append(mod)
    return mod


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu")


def compare_state(state, ref, what):
    """Mirror SyncNetInstance.loadParameters: copy every file tensor into the model state.

    An unknown key raises KeyError there, so it fails here. A model tensor absent from the file
    would silently stay at random init, so it fails here too, except BatchNorm num_batches_tracked
    counters, which checkpoints saved before PyTorch 0.4.1 lack and eval mode never reads.
    """
    if not isinstance(state, dict):
        return False, "not a state dict (%s)" % type(state).__name__
    extra = [k for k in state if k not in ref]
    absent = [k for k in ref if k not in state]
    counters = [k for k in absent if k.endswith("num_batches_tracked")]
    missing = [k for k in absent if k not in counters]
    shape = [k for k in state if k in ref and
             (not hasattr(state[k], "shape") or tuple(state[k].shape) != tuple(ref[k].shape))]
    if missing or extra or shape:
        return False, ("does not match %s: %d missing, %d unexpected, %d shape mismatch (first: %s)"
                       % (what, len(missing), len(extra), len(shape), (missing + extra + shape)[:1]))
    try:
        for k in state:  # the copy that SyncNetInstance.loadParameters performs
            ref[k].copy_(state[k])
    except Exception as e:
        return False, "tensor copy failed: " + err_text(e)
    note = ("; %d BatchNorm num_batches_tracked counters absent (unused in eval mode)" % len(counters)
            if counters else "")
    return True, "%d tensors match %s%s" % (len(state), what, note)


def make_validate_syncnet(repo):
    def validate(path):
        try:
            mod = load_file_module("cafe_syncnet_model",
                                   os.path.join(repo, EVAL, "syncnet_python", "SyncNetModel.py"))
            ref = mod.S(num_layers_in_fc_layers=1024).state_dict()
            state = torch_load(path)
        except Exception as e:
            return False, fail_detail(path, e)
        return compare_state(state, ref, "SyncNetModel.S(1024)")
    return validate


def validate_efan_mead(path):
    """Mirror emotion-fan.py: ResNet18(7) in DataParallel in VideoClassifierWrapper, strict load."""
    try:
        import torch
        from torchvision import models

        class ModuleHolder(torch.nn.Module):  # same parameter naming as nn.DataParallel (.module.)
            def __init__(self, module):
                super(ModuleHolder, self).__init__()
                self.module = module

        class VideoClassifierWrapper(torch.nn.Module):  # same attribute name as emotion-fan.py
            def __init__(self, base_model):
                super(VideoClassifierWrapper, self).__init__()
                self.base_model = base_model

        model = VideoClassifierWrapper(ModuleHolder(models.resnet18(num_classes=7)))
        ckpt = torch_load(path)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            return False, "no 'state_dict' key (emotion-fan.py reads checkpoint['state_dict'])"
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        return False, fail_detail(path, e)
    extras = sorted(str(k) for k in ckpt if k != "state_dict")
    tail = ("; other keys: " + ",".join(extras[:4])) if extras else ""
    return True, "strict load OK (ResNet18, 7 classes, %d tensors)%s" % (len(ckpt["state_dict"]), tail)


def validate_ferplus(path):
    """Mirror basic_code/load.py model_parameters(): skip module.fc.*, strip 'module.', strict load."""
    try:
        from torchvision import models
        ckpt = torch_load(path)
        state = ckpt["state_dict"]
        struct = models.resnet18(num_classes=7)
        merged = struct.state_dict()
        for key in state:
            if key in ("module.fc.weight", "module.fc.bias"):
                continue
            merged[key.replace("module.", "")] = state[key]
        struct.load_state_dict(merged)
    except Exception as e:
        return False, fail_detail(path, e)
    fc = state.get("module.fc.weight")
    fc_txt = str(tuple(fc.shape)) if hasattr(fc, "shape") else "absent"
    return True, "model_parameters replica OK (%d tensors, pretrain fc %s)" % (len(state), fc_txt)


def make_validate_s3fd(repo):
    def validate(path):
        try:
            pkg = load_s3fd_package(repo)
            net = pkg.S3FDNet(device="cpu")
            state = torch_load(path)
            net.load_state_dict(state)
        except Exception as e:
            return False, fail_detail(path, e)
        return True, "strict load into S3FDNet OK (%d tensors)" % len(state)
    return validate


def validate_fid(path):
    got = sha256_file(path)
    if got == FID_SHA256:
        return True, "sha256 matches the pytorch-fid release"
    return False, "sha256 %s differs from release %s" % (got[:12], FID_SHA256[:12])


def check_code(rep, repo):
    missing, modified = [], []
    for rel, want in sorted(EVAL_MANIFEST.items()):
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            missing.append(rel)
        elif sha256_file(p) != want:
            modified.append(rel)
    n = len(EVAL_MANIFEST)
    if not missing and not modified:
        rep.add("OK", "code", "evaluation/ provenance", "%d/%d tracked files match C-MET@%s" % (n, n, CMET_COMMIT[:7]))
    else:
        rep.add("FAIL", "code", "evaluation/ provenance",
                "%d modified, %d missing vs C-MET@%s: %s" % (len(modified), len(missing), CMET_COMMIT[:7],
                                                             ", ".join((modified + missing)[:6])),
                modified=modified, missing=missing)


def handle_pinned(rep, spec, repo, upload_dir):
    dest = os.path.join(repo, spec["dest"])
    want = spec["sha256"]
    if os.path.isfile(dest) and sha256_file(dest) == want:
        rep.add("OK", "asset", spec["label"], "in place; sha256 %s matches Emotion-FAN@%s (%d B)"
                % (want[:12], EFAN_COMMIT[:7], os.path.getsize(dest)), path=dest, sha256=want)
        return
    notes = []
    if os.path.lexists(dest):
        notes.append("existing file differs from pinned (%d B)" % os.path.getsize(dest))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = "%s.part_%d" % (dest, os.getpid())
    source, rejected = None, []
    for up in find_uploads(upload_dir, spec["patterns"]):
        try:
            if up.endswith(".bz2"):
                with bz2.open(up, "rb") as src, open(part, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            else:
                shutil.copyfile(up, part)
        except Exception as e:
            rejected.append("%s: %s" % (os.path.basename(up), err_text(e)))
            continue
        if sha256_file(part) == want:
            source = "copied from upload " + os.path.basename(up)
            break
        rejected.append(os.path.basename(up) + ": sha256 mismatch")
    notes.extend(rejected)
    if source is None:
        url = EFAN_LIB_URL + os.path.basename(spec["dest"])
        ok, how = fetch(url, part)
        if ok and sha256_file(part) == want:
            source = "fetched from Emotion-FAN@%s (%s)" % (EFAN_COMMIT[:7], how)
        elif ok:
            notes.append("fetched copy sha256 mismatch")
        else:
            notes.append("fetch failed: " + how)
    if source is None:
        remove_quiet(part)
        status = "FAIL" if (os.path.isfile(dest) and os.path.getsize(dest) > 0) else "MISSING"
        rep.add(status, "asset", spec["label"], "; ".join(notes), path=dest)
        return
    note = set_aside(dest, "replaced")
    os.replace(part, dest)
    tail = ("; not used: " + "; ".join(rejected)) if rejected else ""
    rep.add("OK", "asset", spec["label"], "%s; sha256 %s verified%s%s" % (source, want[:12], note, tail),
            path=dest, sha256=want)


def identity(spec, path):
    """Compare a functionally valid file with the canonical SHA-256, when one is known."""
    digest = sha256_file(path)
    ref = spec.get("ref_sha256")
    if not ref:
        return "OK", "", digest
    if digest == ref:
        return "OK", "; sha256 %s is the canonical file" % ref[:12], digest
    return "WARN", ("; sha256 %s differs from the canonical %s, confirm the source"
                    % (digest[:12], ref[:12])), digest


def handle_checked(rep, spec, upload_dir):
    dest, validate, label = spec["dest"], spec["validate"], spec["label"]
    uploads = [u for u in find_uploads(upload_dir, spec["patterns"])
               if os.path.realpath(u) != os.path.realpath(dest)]
    notes = []
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        ok, detail = validate(dest)
        if ok:
            status, idnote, digest = identity(spec, dest)
            extra = ("; unused upload(s): " + ", ".join(os.path.basename(u) for u in uploads)) if uploads else ""
            rep.add(status, "asset", label, "in place; %s%s%s" % (detail, idnote, extra), path=dest,
                    size=os.path.getsize(dest), sha256=digest)
            return
        notes.append("file in place is invalid: " + detail)
    for up in uploads:
        if os.path.getsize(up) == 0:
            notes.append(os.path.basename(up) + ": 0 bytes, ignored")
            continue
        ok, detail = validate(up)
        if not ok:
            notes.append("%s: %s" % (os.path.basename(up), detail))
            continue
        note = set_aside(dest, "invalid")
        move_into_place(up, dest)
        ok2, detail2 = validate(dest)
        status, idnote, digest = identity(spec, dest) if ok2 else ("FAIL", "", sha256_file(dest))
        tail = ("; earlier: " + "; ".join(notes)) if notes else ""
        rep.add(status, "asset", label,
                "placed from upload %s; %s%s%s%s" % (os.path.basename(up), detail2, idnote, note, tail),
                path=dest, size=os.path.getsize(dest), sha256=digest)
        return
    for url in spec.get("urls", []):
        part = "%s.part_%d" % (dest, os.getpid())
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        ok, how = fetch(url, part)
        if not ok:
            notes.append("fetch %s failed" % url.split("://")[0])
            continue
        ok, detail = validate(part)
        if not ok:
            notes.append("fetched copy invalid: " + detail)
            remove_quiet(part)
            continue
        note = set_aside(dest, "invalid")
        os.replace(part, dest)
        status, idnote, digest = identity(spec, dest)
        tail = ("; earlier: " + "; ".join(notes)) if notes else ""
        rep.add(status, "asset", label, "fetched from %s (%s); %s%s%s%s" % (url, how, detail, idnote, note, tail),
                path=dest, size=os.path.getsize(dest), sha256=digest)
        return
    status = "FAIL" if (os.path.lexists(dest) or uploads) else "MISSING"
    notes.append("get it from: " + spec["howto"])
    rep.add(status, "asset", label, "; ".join(notes), path=dest)


def dlib_checks(rep, repo):
    try:
        import dlib
    except Exception:
        rep.add("WARN", "func", "dlib model loads", "skipped: dlib not importable (re-run after install)")
        return False
    placed = dict((r["name"], r["status"]) for r in rep.rows if r["group"] == "asset")
    ok_all = True
    for label, asset, fname, loader in (
            ("dlib loads shape_predictor_5", "shape_predictor_5 (dlib)", "shape_predictor_5_face_landmarks.dat",
             dlib.shape_predictor),
            ("dlib loads mmod CNN detector", "mmod_human_face_detector", "mmod_human_face_detector.dat",
             dlib.cnn_face_detection_model_v1)):
        if placed.get(asset) != "OK":
            ok_all = False
            rep.add("WARN", "func", label, "skipped: %s is not in place" % fname)
            continue
        try:
            loader(os.path.join(repo, LIB, fname))
            rep.add("OK", "func", label, "loaded (dlib CUDA %s)" % getattr(dlib, "DLIB_USE_CUDA", "?"))
        except Exception as e:
            ok_all = False
            rep.add("FAIL", "func", label, err_text(e))
    return ok_all


def sample_video(repo):
    vids = sorted(glob.glob(os.path.join(repo, GEN_GLOB)))
    return vids[0] if vids else None


def face_crop_smoke(rep, repo, python_ok):
    """Mirror frame2face_custom.py on one generated video: cv2 decode, sample 3 frames, exact crop command."""
    video = sample_video(repo)
    if video is None:
        rep.add("WARN", "func", "face-crop smoke test", "skipped: no generated video under " + GEN_GLOB)
        return
    if not python_ok:
        rep.add("WARN", "func", "face-crop smoke test", "skipped: `python` on PATH is not this interpreter")
        return
    if not any(r["name"] == "face_align_cuda.py" and r["status"] == "OK" for r in rep.rows):
        rep.add("WARN", "func", "face-crop smoke test", "skipped: face_align_cuda.py is not in place")
        return
    tmp = tempfile.mkdtemp(prefix="cafe_facecrop_")
    try:
        import cv2
        import numpy as np
        out_dir = os.path.join(tmp, "0000000")
        all_dir, src_dir = os.path.join(out_dir, "all_frames"), os.path.join(out_dir, "sampled_frames")
        os.makedirs(all_dir)
        os.makedirs(src_dir)
        cap = cv2.VideoCapture(video)  # frame2face_custom.py decodes with OpenCV, not ffmpeg
        n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(all_dir, "frame_%06d.jpg" % n), frame)
            n += 1
        cap.release()
        if n == 0:
            rep.add("FAIL", "func", "face-crop smoke test", "cv2.VideoCapture decoded 0 frames of %s "
                    "(frame2face_custom.py would skip every video silently)" % os.path.basename(video))
            return
        files = sorted(glob.glob(os.path.join(all_dir, "*.jpg")))
        for i in np.linspace(0, len(files) - 1, 3, dtype=int):
            shutil.copy2(files[i], src_dir)
        frames = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
        # Same command string as frame2face_custom.py (paths relative to evaluation/).
        cmd = 'python {:} {:} "{:}" "{:}" {:} {:}'.format(
            "./Emotion-FAN/data/face_alignment_code/lib/face_align_cuda.py",
            "./Emotion-FAN/data/face_alignment_code/lib/shape_predictor_5_face_landmarks.dat",
            src_dir, out_dir,
            "./Emotion-FAN/data/face_alignment_code/lib/mmod_human_face_detector.dat", 0)
        res = subprocess.run(cmd, shell=True, cwd=os.path.join(repo, EVAL), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        out = res.stdout.decode("utf-8", "replace").strip()
        crops = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
        shapes = [tuple(cv2.imread(c).shape) for c in crops]
        last = out.splitlines()[-1][:120] if out else ""
        if res.returncode != 0:
            rep.add("FAIL", "func", "face-crop smoke test", "rc=%d: %s" % (res.returncode, last))
        elif len(crops) == len(frames) and all(s == (224, 224, 3) for s in shapes):
            rep.add("OK", "func", "face-crop smoke test", "cv2 decoded %d frames; %d/%d sampled frames cropped "
                    "to 224x224 (%s)" % (n, len(crops), len(frames), os.path.basename(video)))
        elif crops and all(s == (224, 224, 3) for s in shapes):
            rep.add("WARN", "func", "face-crop smoke test", "cv2 decoded %d frames; only %d/%d sampled frames "
                    "gave a face (%s)" % (n, len(crops), len(frames), os.path.basename(video)))
        elif crops:
            rep.add("FAIL", "func", "face-crop smoke test", "unexpected crop shapes %s" % shapes[:3])
        else:
            rep.add("FAIL", "func", "face-crop smoke test", "no face found in %d sampled frames of %s; output: %s"
                    % (len(frames), os.path.basename(video), last))
    except Exception as e:
        rep.add("FAIL", "func", "face-crop smoke test", err_text(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fid_build_check(rep):
    """Build InceptionV3(2048) the way pytorch-fid/custom.py does and run one CPU forward pass."""
    name = "FID InceptionV3 build"
    rows = [r for r in rep.rows if r["name"] == "FID InceptionV3 weights"]
    if not rows or rows[-1]["status"] != "OK":
        rep.add("WARN", "func", name, "skipped: FID weights not in place")
        return
    try:
        import torch
        from pytorch_fid.inception import InceptionV3
    except Exception:
        rep.add("WARN", "func", name, "skipped: pytorch_fid not importable (re-run after install)")
        return
    try:
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
        with torch.no_grad():
            shape = tuple(model(torch.rand(2, 3, 256, 256))[0].shape)
        if shape == (2, 2048, 1, 1):
            rep.add("OK", "func", name, "built from the cached weights; forward output %s" % (shape,))
        else:
            rep.add("FAIL", "func", name, "forward output %s, expected (2, 2048, 1, 1)" % (shape,))
    except Exception as e:
        rep.add("FAIL", "func", name, err_text(e))


def dist_version(dist, mod):
    try:
        import importlib.metadata as md
        return md.version(dist)
    except Exception:
        return str(getattr(mod, "__version__", "?"))


def installed_dists(names):
    found = []
    for d in names:
        try:
            import importlib.metadata as md
            found.append("%s %s" % (d, md.version(d)))
        except Exception:
            pass
    return found


def check_packages(rep, repo):
    versions = {}
    for name, dist, used_by in PACKAGES:
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            versions[name] = None
            rep.add("MISSING", "pkg", name, "pip name %s; needed by %s (%s)" % (dist, used_by, err_text(e)[:60]))
            continue
        ver = dist_version(dist, mod)
        versions[name] = ver
        status, detail = "OK", ver
        if name == "torch":
            cuda = getattr(getattr(mod, "version", None), "cuda", None)
            detail = "%s (CUDA build %s)" % (ver, cuda)
            if not ver.startswith(TORCH_RECORDED):
                status, detail = "WARN", detail + "; recorded env has %s" % TORCH_RECORDED
        elif name == "scenedetect":
            try:
                from scenedetect.video_manager import VideoManager  # noqa: F401
                from scenedetect.scene_manager import SceneManager  # noqa: F401
                from scenedetect.frame_timecode import FrameTimecode  # noqa: F401
                from scenedetect.stats_manager import StatsManager  # noqa: F401
                from scenedetect.detectors import ContentDetector  # noqa: F401
                api = "0.5 API imports OK"
            except Exception as e:
                status, api = "FAIL", "run_pipeline.py imports fail: " + err_text(e)[:80]
            detail = "%s; %s" % (ver, api)
            if status == "OK" and ver.lstrip("v") not in SCENEDETECT_OK:
                status, detail = "WARN", detail + "; expected 0.5.6.1 (requirements.txt pins 0.5.1)"
        elif name == "dlib":
            detail = "%s (CUDA %s)" % (getattr(mod, "__version__", ver), getattr(mod, "DLIB_USE_CUDA", "?"))
        elif name == "cv2":
            dists = installed_dists(("opencv-python", "opencv-contrib-python", "opencv-python-headless",
                                     "opencv-contrib-python-headless"))
            detail = "%s (%s)" % (getattr(mod, "__version__", "?"), "; ".join(dists) or "no pip dist found")
            if len(dists) > 1:
                status, detail = "WARN", detail + "; several OpenCV wheels installed"
        elif name == "pytorch_fid":
            try:
                inc = importlib.import_module("pytorch_fid.inception")
                vendored = os.path.join(repo, EVAL, "pytorch-fid", "src", "pytorch_fid", "inception.py")
                same = sha256_file(inc.__file__) == sha256_file(vendored)
                detail = "%s from %s; inception.py %s vendored copy" % (
                    ver, os.path.dirname(inc.__file__), "identical to" if same else "DIFFERS from")
                if not same:
                    status = "WARN"
            except Exception as e:
                status, detail = "FAIL", "%s; pytorch_fid.inception import fails: %s" % (ver, err_text(e)[:80])
        rep.add(status, "pkg", name, detail)
    return versions


def scenedetect_run(rep, avi):
    """Run the exact scene_detect() sequence of run_pipeline.py on the converted video."""
    name = "scenedetect run"
    try:
        import scenedetect
        from scenedetect.video_manager import VideoManager
        from scenedetect.scene_manager import SceneManager
        from scenedetect.stats_manager import StatsManager
        from scenedetect.detectors import ContentDetector
    except Exception:
        rep.add("WARN", "func", name, "skipped: scenedetect not importable (re-run after install)")
        return
    ver = str(getattr(scenedetect, "__version__", "?")).lstrip("v")
    try:
        video_manager = VideoManager([avi])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)
        if scene_list == []:
            scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
        video_manager.release()
    except Exception as e:
        hint = ("; install scenedetect==0.5.6.1" if "tuple" in str(e) else "")
        rep.add("FAIL", "func", name, "%s: %s%s" % (ver, err_text(e), hint))
        return
    detail = "%s: run_pipeline scene_detect() sequence OK (%d scene(s))" % (ver, len(scene_list))
    if not ver.startswith("0.5"):
        rep.add("WARN", "func", name, detail + "; not the 0.5 series run_pipeline.py was written for")
    else:
        rep.add("OK", "func", name, detail)


def mfcc_run(rep, wav):
    """Run the MFCC sequence of SyncNetInstance.evaluate on the extracted 16 kHz audio."""
    name = "python_speech_features mfcc"
    try:
        import numpy
        import python_speech_features
        from scipy.io import wavfile
    except Exception:
        rep.add("WARN", "func", name, "skipped: python_speech_features not importable (re-run after install)")
        return
    try:
        sample_rate, audio = wavfile.read(wav)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        ok = mfcc.ndim == 2 and mfcc.shape[0] == 13 and mfcc.shape[1] > 0 and bool(numpy.isfinite(mfcc).all())
    except Exception as e:
        rep.add("FAIL", "func", name, err_text(e))
        return
    if ok:
        rep.add("OK", "func", name, "SyncNetInstance MFCC sequence OK (13 x %d at %d Hz)" % (mfcc.shape[1], sample_rate))
    else:
        rep.add("FAIL", "func", name, "unexpected MFCC output shape %s or non-finite values" % (mfcc.shape,))


def check_ffmpeg(rep, repo):
    ff = shutil.which("ffmpeg")
    if not ff:
        rep.add("FAIL", "env", "ffmpeg", "not on PATH (vide2frame_custom.py and run_pipeline.py call it)")
        return
    try:
        ver = subprocess.run([ff, "-version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=30).stdout.decode("utf-8", "replace").splitlines()[0]
        ver = ver.split(" Copyright")[0]
    except Exception as e:
        ver = err_text(e)
    video = sample_video(repo)
    if video is None:
        rep.add("WARN", "env", "ffmpeg", "%s (%s); flag probe skipped: no generated video" % (ff, ver[:40]))
        return
    tmp = tempfile.mkdtemp(prefix="cafe_ffprobe_")
    failed = []

    def shell(label, cmd, out):
        res = subprocess.run(cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=300)
        if os.path.isdir(out):
            produced = len(glob.glob(os.path.join(out, "*.jpg"))) > 0
        else:
            produced = os.path.isfile(out) and os.path.getsize(out) > 0
        if res.returncode != 0 or not produced:
            tail = res.stdout.decode("utf-8", "replace").strip().splitlines()
            failed.append("%s rc=%d %s" % (label, res.returncode, tail[-1][:90] if tail else ""))
            return False
        return True

    try:
        j = lambda *a: os.path.join(tmp, *a)  # noqa: E731
        for d in ("frames", "v2f", "sn"):
            os.makedirs(j(d))
        # Commands copied from run_pipeline.py (unquoted, as there) and vide2frame_custom.py.
        shell("run_pipeline convert", "ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (video, j("video.avi")),
              j("video.avi"))
        shell("run_pipeline frames", "ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s"
              % (j("video.avi"), j("frames", "%06d.jpg")), j("frames"))
        shell("run_pipeline wav", "ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s"
              % (j("video.avi"), j("audio.wav")), j("audio.wav"))
        shell("vide2frame", 'ffmpeg -i "%s" -f image2 "%s/%%07d.jpg"' % (video, j("v2f")), j("v2f"))
        # run_pipeline.py crop_video: OpenCV XVID writer, audio cut, stream-copy mux. A failure of
        # either ffmpeg call there drops into pdb.set_trace(), so it is probed here first.
        crop = j("00000")
        try:
            import cv2
            import numpy as np
            vout = cv2.VideoWriter(crop + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224))
            for i in range(25):
                vout.write(np.full((224, 224, 3), 4 * i, dtype=np.uint8))
            vout.release()
            if not (os.path.isfile(crop + "t.avi") and os.path.getsize(crop + "t.avi") > 0):
                failed.append("cv2.VideoWriter XVID wrote no file (run_pipeline crop_video)")
        except Exception as e:
            failed.append("cv2.VideoWriter XVID " + err_text(e)[:90])
        if (shell("run_pipeline audio cut", "ffmpeg -y -i %s -ss %.3f -to %.3f %s"
                  % (j("audio.wav"), 0.0, 1.0, j("audio_cut.wav")), j("audio_cut.wav"))
                and os.path.isfile(crop + "t.avi")
                and shell("run_pipeline mux", "ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi"
                          % (crop, j("audio_cut.wav"), crop), crop + ".avi")):
            try:
                from scipy.io import wavfile
                rate, _ = wavfile.read(j("audio_cut.wav"))
                if rate != 16000:
                    failed.append("cut audio sample rate %d, expected 16000" % rate)
            except Exception as e:
                failed.append("scipy wavfile.read " + err_text(e)[:90])
            # SyncNetInstance.evaluate on the cropped track.
            shell("SyncNetInstance frames", "ffmpeg -y -i %s -threads 1 -f image2 %s"
                  % (crop + ".avi", j("sn", "%06d.jpg")), j("sn"))
            shell("SyncNetInstance wav", "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s"
                  % (crop + ".avi", j("sn", "audio.wav")), j("sn", "audio.wav"))
        if failed:
            rep.add("FAIL", "env", "ffmpeg", "%s; %s" % (ver[:40], " | ".join(failed)))
        else:
            rep.add("OK", "env", "ffmpeg", "%s; all 10 media steps of run_pipeline, SyncNetInstance and "
                    "vide2frame OK on %s" % (ver[:40], os.path.basename(video)))
        if os.path.isfile(j("video.avi")):
            scenedetect_run(rep, j("video.avi"))
        else:
            rep.add("WARN", "func", "scenedetect run", "skipped: no converted video")
        if os.path.isfile(j("sn", "audio.wav")):
            mfcc_run(rep, j("sn", "audio.wav"))
        else:
            rep.add("WARN", "func", "python_speech_features mfcc", "skipped: no extracted audio")
    except Exception as e:
        rep.add("FAIL", "env", "ffmpeg", "%s; probe error %s" % (ver[:40], err_text(e)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_environment(rep, repo):
    import torch
    exe = os.path.realpath(sys.executable)
    rep.add("INFO", "env", "interpreter", "%s (Python %s)" % (sys.executable, platform.python_version()))
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "C_MET":
        rep.add("WARN", "env", "conda env", "CONDA_DEFAULT_ENV=%r (expected C_MET)" % conda_env)
    on_path = shutil.which("python")
    python_ok = bool(on_path) and os.path.realpath(on_path) == exe
    rep.add("OK" if python_ok else "FAIL", "env", "`python` on PATH",
            ("same interpreter" if python_ok else "resolves to %s; subprocess steps would use it" % on_path))
    if torch.cuda.is_available():
        rep.add("OK", "env", "CUDA", "%d device(s); cuda:0 = %s" % (torch.cuda.device_count(),
                                                                    torch.cuda.get_device_name(0)))
    else:
        rep.add("FAIL", "env", "CUDA", "not available (emotion-fan and SyncNet call .cuda())")
    check_ffmpeg(rep, repo)
    free_gb = shutil.disk_usage(repo).free / 1e9
    rep.add("INFO", "env", "disk free", "%.1f GB on the repo filesystem" % free_gb)
    csvs = sorted(glob.glob(os.path.join(repo, EVAL, "runs", "mead_ours", "*.csv")))
    rep.add("INFO", "env", "runs/mead_ours CSVs", ", ".join(os.path.basename(c) for c in csvs) or "none yet")
    return python_ok


def main():
    repo = os.getcwd()
    upload_dir = os.path.dirname(os.path.abspath(__file__))
    if not (os.path.isfile(os.path.join(repo, "inference.py")) and
            os.path.isdir(os.path.join(repo, EVAL, "syncnet_python"))):
        print("Run this from the C-MET repository root (the folder with inference.py and evaluation/).")
        return 2
    try:
        import torch
    except Exception as e:
        print("torch is not importable (%s). Activate the C_MET environment first." % err_text(e))
        return 2
    try:
        import torch.hub
        hub_dir = torch.hub.get_dir()
    except Exception:
        hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
    print("%s | %s | host %s | repo %s | uploads %s" % (SCRIPT, STAMP, socket.gethostname(), repo, upload_dir))
    rep = Report()

    check_code(rep, repo)
    for spec in PINNED:
        handle_pinned(rep, spec, repo, upload_dir)
    checked = [
        {"label": "syncnet_v2.model", "dest": EVAL + "/syncnet_python/data/syncnet_v2.model",
         "patterns": ["syncnet_v2*.model"], "validate": make_validate_syncnet(repo), "ref_sha256": SYNCNET_SHA256,
         "urls": [OXFORD_HTTPS + "syncnet_v2.model"],
         "howto": OXFORD_HTTPS + "syncnet_v2.model"},
        {"label": "Emotion-FAN_MEAD.pth", "dest": EVAL + "/Emotion-FAN/checkpoints/Emotion-FAN_MEAD.pth",
         "patterns": ["Emotion-FAN_MEAD*.pth"], "validate": validate_efan_mead,
         "howto": "Google Drive id 1H0tqOEe5-EqlmomB_FujgbrG8C7dadf1 (C-MET evaluation README), rename"},
        {"label": "Resnet18_FER+ (pretrain)", "dest": EVAL + "/Emotion-FAN/pretrain_model/Resnet18_FER+_pytorch.pth.tar",
         "patterns": ["Resnet18_FER+_pytorch*.pth.tar"], "validate": validate_ferplus,
         "howto": "Emotion-FAN pretrain_model/readme.md (OneDrive or Baidu link)"},
        {"label": "sfd_face.pth (S3FD)", "dest": EVAL + "/syncnet_python/detectors/s3fd/weights/sfd_face.pth",
         "patterns": ["sfd_face*.pth"], "validate": make_validate_s3fd(repo), "ref_sha256": SFD_SHA256,
         "urls": [OXFORD_HTTPS + "sfd_face.pth"], "howto": OXFORD_HTTPS + "sfd_face.pth"},
        {"label": "FID InceptionV3 weights", "dest": os.path.join(hub_dir, "checkpoints", FID_NAME),
         "patterns": [FID_NAME.replace(".pth", "*.pth")], "validate": validate_fid,
         "urls": [FID_URL], "howto": FID_URL},
    ]
    for spec in checked:
        if not os.path.isabs(spec["dest"]):
            spec["dest"] = os.path.join(repo, spec["dest"])
        handle_checked(rep, spec, upload_dir)

    versions = check_packages(rep, repo)
    python_ok = check_environment(rep, repo)
    if dlib_checks(rep, repo):
        face_crop_smoke(rep, repo, python_ok)
    else:
        rep.add("WARN", "func", "face-crop smoke test", "skipped: dlib or its models not ready")
    fid_build_check(rep)

    blockers = [r["name"] for r in rep.rows if r["status"] in ("FAIL", "MISSING")]
    report_dir = os.path.join(upload_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "eval_assets_%s.json" % STAMP)
    with open(report_path, "w") as f:
        json.dump({"script": SCRIPT, "stamp": STAMP, "host": socket.gethostname(), "repo": repo,
                   "upload_dir": upload_dir, "python": sys.version, "executable": sys.executable,
                   "cmet_commit": CMET_COMMIT, "emotion_fan_commit": EFAN_COMMIT,
                   "packages": versions, "rows": rep.rows, "blockers": blockers}, f, indent=1)
    print("SUMMARY ok=%d warn=%d fail=%d missing=%d info=%d" % (
        rep.count("OK"), rep.count("WARN"), rep.count("FAIL"), rep.count("MISSING"), rep.count("INFO")))
    print("BLOCKERS: " + (", ".join(blockers) if blockers else "none"))
    print("REPORT: " + report_path)
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
