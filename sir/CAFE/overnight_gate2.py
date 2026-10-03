#!/usr/bin/env python3
"""
CAFE overnight chain: finish Gate 1 (sentence renumbering) and run the Gate 2
generation of all C-MET test videos with the OFFICIAL evaluation direction.

Stages (each idempotent, state in <meta>/overnight_state.json, safe to re-run):
  S1 transcribe  : wait for any running transcribe_mead.py, then a resume pass;
                   require every wav transcribed.
  S2 align_gate  : align_mead_levels analysis + automatic gate:
                   validation A disagree == 0 and agree >= --min_agree,
                   validation B misassigned == 0, no weak match inside a group
                   that would be renamed, every clip of a renamed group mapped.
  S3 renumber    : only if the gate passed. Per group, a renamed copy of the
                   folder is built OUTSIDE the tree with hard links, then two
                   atomic directory renames swap it in; the original folder is
                   kept OUTSIDE the tree (so no glob ever sees it). Crash-safe
                   recovery by filesystem state. Ledgers and transcripts are
                   rewritten to the new names.
  S3b post_verify: re-run the analysis on the renamed tree; every group must be
                   identity and validation A must still agree. Otherwise the
                   renumbering is rolled back automatically.
  S4 manifest    : rebuild test_present.csv / test_missing.txt from the
                   authors' test.csv against the (possibly renamed) tree.
  S5 generate    : C-MET generation for every test row, models loaded once.
                   Direction = C-MET's own Dataset('test', './dataset/MEAD/FPS25')
                   .get_raw_e2v('neutral', emo, level)  (the protocol line that is
                   commented out in the released inference.py). Generation steps
                   mirror inference.py (opt 'A'). If the gate failed, only levels
                   unaffected by renumbering are generated.
  S6 report      : summary JSON + text.

Declared assumption: the identity image for a test row is frame 0 of the
neutral source video (the released code does not specify it for the test set).

Run from the C-MET repo root (C_MET env):
  python ../overnight_gate2.py --selftest
  python ../overnight_gate2.py --smoke            # 2 real level_1 rows, then stop
  python ../overnight_gate2.py                    # full chain
"""

import argparse, csv, glob, json, os, shutil, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
FPS25 = "dataset/MEAD/FPS25"
FPS25_CMET = "./dataset/MEAD/FPS25"          # exact string C-MET's Dataset expects
META = "dataset/MEAD"
E2V_DIR = "emotion2vec+large_features"
STEM_FILES = ["{s}.mp4", "{s}.wav", "{s}_ED_exp.npy", "{s}_ED_pose.npy", "{s}_ED_lip.npy",
              E2V_DIR + "/{s}.npy"]


def log(msg, fh=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


# ---------------------------------------------------------------- state -----

class State:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.isfile(path) else {}
    def get(self, k, default=None):
        return self.d.get(k, default)
    def set(self, k, v):
        self.d[k] = v
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f, indent=1)
        os.replace(tmp, self.path)


# ------------------------------------------------------------- S2: gate -----

def folder_stems(folder):
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(folder, "*.mp4"))
                  if os.path.basename(p)[:-4].isdigit())

def decide_gate(rep, fps25_root, min_agree, accept_weak=()):
    va = rep.get("validation_a_authors_e2v") or {}
    vb = rep.get("validation_b_common_sentences") or {}
    groups = rep["groups"]
    nonid = sorted(k for k, g in groups.items() if g["verdict"] != "identity")
    reasons = []
    if va.get("disagree", 1) != 0:
        reasons.append(f"validation A disagree={va.get('disagree')}")
    if va.get("agree", 0) < min_agree:
        reasons.append(f"validation A agree={va.get('agree')} < {min_agree}")
    if vb.get("misassigned", 1) != 0:
        reasons.append(f"validation B misassigned={vb.get('misassigned')}")
    acc = set(accept_weak or ())
    weak_bad = [w for w in rep.get("weak_matches", []) if w["group"] in nonid
                and f"{w['group']}:{w['ours']}" not in acc]
    if weak_bad:
        reasons.append(f"{len(weak_bad)} weak matches inside groups to rename (not accepted): "
                       + ", ".join(f"{w['group']}:{w['ours']}" for w in weak_bad[:10]))
    for k in nonid:
        ident, emo, lvl = k.split("/")
        stems = folder_stems(os.path.join(fps25_root, ident, "front", emo, lvl))
        if sorted(groups[k]["map"]) != stems:
            reasons.append(f"{k}: mapping does not cover exactly the files on disk")
        if len(set(groups[k]["map"].values())) != len(groups[k]["map"]):
            reasons.append(f"{k}: mapping targets not unique")
    passed = not reasons
    # whole (ID, emotion) folders can be offset (level_1 included), so without a passed
    # gate no level is trustworthy: generate nothing rather than contaminate resumable outputs
    allowed = ["level_1", "level_2", "level_3"] if passed else []
    return {"passed": passed, "reasons": reasons, "rename_groups": nonid if passed else [],
            "allowed_levels": allowed}


# --------------------------------------------------------- S3: renumber -----

def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def _gkey(k):
    return k.replace("/", "__")

def build_renamed(live, work, mapping):
    """Create `work` = copy of `live` (hard links) with files renamed by mapping."""
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, E2V_DIR))
    moved = set()
    for src_stem, dst_stem in mapping.items():
        for pat in STEM_FILES:
            s = os.path.join(live, pat.format(s=src_stem))
            if not os.path.isfile(s):
                raise FileNotFoundError(f"missing {s}")
            _link_or_copy(s, os.path.join(work, pat.format(s=dst_stem)))
            moved.add(os.path.relpath(s, live))
    # anything else in the folder is carried over unchanged (should be nothing)
    for root, _, files in os.walk(live):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), live)
            if rel not in moved:
                d = os.path.join(work, rel)
                if os.path.exists(d):
                    raise FileExistsError(f"unexpected extra file collides: {rel}")
                os.makedirs(os.path.dirname(d), exist_ok=True)
                _link_or_copy(os.path.join(root, fn), d)
    n_live = sum(len(f) for _, _, f in os.walk(live))
    n_work = sum(len(f) for _, _, f in os.walk(work))
    if n_live != n_work:
        raise RuntimeError(f"file count mismatch {n_live} vs {n_work}")

def apply_group(fps25_root, meta, key, mapping):
    ident, emo, lvl = key.split("/")
    live = os.path.join(fps25_root, ident, "front", emo, lvl)
    work = os.path.join(meta, "renum_work", _gkey(key))
    backup = os.path.join(meta, "renum_backup", _gkey(key))
    os.makedirs(os.path.dirname(work), exist_ok=True)
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    # recovery by filesystem state
    if os.path.isdir(backup) and os.path.isdir(live) and not os.path.isdir(work):
        return "already_applied"
    if os.path.isdir(backup) and not os.path.isdir(live):
        if os.path.isdir(work):
            os.rename(work, live)                 # finish an interrupted swap
            return "recovered_swap"
        os.rename(backup, live)                   # nothing to swap in: restore
    if os.path.isdir(backup) and os.path.isdir(live) and os.path.isdir(work):
        shutil.rmtree(work)
        return "already_applied"
    build_renamed(live, work, mapping)
    os.rename(live, backup)                       # atomic: original out of the tree
    os.rename(work, live)                         # atomic: renamed copy in
    return "applied"

def rollback_group(fps25_root, meta, key):
    ident, emo, lvl = key.split("/")
    live = os.path.join(fps25_root, ident, "front", emo, lvl)
    backup = os.path.join(meta, "renum_backup", _gkey(key))
    if not os.path.isdir(backup):
        return "no_backup"
    trash = os.path.join(meta, "renum_trash", _gkey(key) + f"_{int(time.time()*1000)}")
    os.makedirs(os.path.dirname(trash), exist_ok=True)
    if os.path.isdir(live):
        os.rename(live, trash)
    os.rename(backup, live)
    return "rolled_back"

def rewrite_ledgers_and_transcripts(fps25_root, meta, remaps):
    """remaps: {group_key: {old_stem: new_stem}}. Ledgers become the on-disk truth
       (valid because every clip was fully processed before renumbering)."""
    mp4s = sorted(os.path.relpath(p, fps25_root)
                  for p in glob.glob(os.path.join(fps25_root, "*", "front", "*", "*", "*.mp4")))
    for name in ("crop_done.txt", "fps_done.txt"):
        p = os.path.join(meta, name)
        if os.path.isfile(p):
            shutil.copy2(p, p + ".pre_renum")
            with open(p + ".tmp", "w") as f:
                f.write("\n".join(mp4s) + "\n")
            os.replace(p + ".tmp", p)
    tp = os.path.join(meta, "transcripts_all.csv")
    if os.path.isfile(tp):
        shutil.copy2(tp, tp + ".pre_renum")
        rows = list(csv.reader(open(tp, newline="", encoding="utf-8")))
        out = []
        for r in rows:
            if len(r) == 2 and r[0].endswith(".wav"):
                p = r[0].split("/")
                k = f"{p[0]}/{p[2]}/{p[3]}"
                if k in remaps and p[4][:-4] in remaps[k]:
                    p[4] = remaps[k][p[4][:-4]] + ".wav"
                    r = ["/".join(p), r[1]]
            out.append(r)
        with open(tp + ".tmp", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(out)
        os.replace(tp + ".tmp", tp)

def ledgers_complete(fps25_root, meta):
    mp4s = {os.path.relpath(p, fps25_root)
            for p in glob.glob(os.path.join(fps25_root, "*", "front", "*", "*", "*.mp4"))}
    for name in ("crop_done.txt", "fps_done.txt"):
        p = os.path.join(meta, name)
        got = {l.strip() for l in open(p)} if os.path.isfile(p) else set()
        if not mp4s <= got:
            return False, f"{name} misses {len(mp4s - got)} clips"
    return True, "ok"


# --------------------------------------------------------- S4: manifest -----

TEST_COLS = ["source_video_path", "gt_video_path", "gt_emotion", "intensity"]

def rebuild_manifest(meta):
    src = os.path.join(meta, "test.csv")
    rows = list(csv.DictReader(open(src, newline="")))
    present, missing = [], []
    for r in rows:
        need = [r["source_video_path"], r["gt_video_path"], r["source_video_path"][:-4] + ".wav"]
        (present if all(os.path.isfile(p) for p in need) else missing).append(r)
    with open(os.path.join(meta, "test_present.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEST_COLS); w.writeheader()
        for r in present:
            w.writerow({k: r[k] for k in TEST_COLS})
    with open(os.path.join(meta, "test_missing_clips.txt"), "w") as f:
        for r in missing:
            f.write(r["gt_video_path"] + "\n")
    return {"total": len(rows), "present": len(present), "missing": len(missing)}


# --------------------------------------------------------- S5: generate -----

def out_name(row):
    p = row["gt_video_path"].replace("\\", "/").split("/")
    return f"{p[-5]}_{p[-3]}_{p[-2]}_{p[-1][:-4]}.mp4"      # ID_emotion_level_NNN.mp4

def probe_ok(path, want=256):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False, "missing"
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,nb_frames", "-of", "json", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, "ffprobe_failed"
    st = json.loads(r.stdout or "{}").get("streams", [])
    v = [s for s in st if s.get("codec_type") == "video"]
    a = [s for s in st if s.get("codec_type") == "audio"]
    if not v:
        return False, "no_video_stream"
    if (v[0].get("width"), v[0].get("height")) != (want, want):
        return False, f"size_{v[0].get('width')}x{v[0].get('height')}"
    if int(v[0].get("nb_frames", 0) or 0) <= 0:
        return False, "zero_frames"
    if not a:
        return False, "no_audio_stream"
    return True, "ok"


class RealGen:
    """Mirror of C-MET inference.py (opt 'A'), models loaded once."""
    def __init__(self, log_fn):
        sys.path.insert(0, os.getcwd()); sys.path.append("src"); sys.path.append("src/metavoice")
        import torch, numpy as np
        from omegaconf import OmegaConf
        import inference as I                              # C-MET's own module (main not run)
        from src.EDTalk.networks.generator import Generator
        from src.EDTalk.networks.audio_encoder import Audio2Lip
        from src.connector import Connector_exp
        from src.util import vid_preprocessing, save_video, img_preprocessing, audio_preprocessing, conv_feat
        from src.dataset_emo12 import Dataset
        self.torch, self.np, self.I = torch, np, I
        self.vp, self.sv, self.ip, self.ap, self.cf = (vid_preprocessing, save_video, img_preprocessing,
                                                        audio_preprocessing, conv_feat)
        cfg = OmegaConf.load("./configs/inference.yaml")
        self.device = "cuda"
        pe = OmegaConf.to_container(cfg.pretrained_EDTalk, resolve=True)
        pk = OmegaConf.to_container(cfg.projector_kwargs, resolve=True)
        tk = OmegaConf.to_container(cfg.transformer_kwargs, resolve=True)
        self.T = tk["T"]
        a2l = Audio2Lip().cuda()
        a2l.load_state_dict(torch.load(cfg["audio2lip_model_path"], map_location=lambda s, l: s)["audio2lip"])
        self.a2l = a2l.eval()
        gen = Generator(pe["size"], style_dim=pe["latent_dim_style"], lip_dim=pe["latent_dim_lip"],
                        pose_dim=pe["latent_dim_pose"], exp_dim=pe["latent_dim_exp"],
                        channel_multiplier=pe["channel_multiplier"]).to(self.device)
        gen.load_state_dict(torch.load(pe["model_path"], map_location=lambda s, l: s)["gen"])
        self.gen = gen.eval()
        con = Connector_exp(pk, tk, self.device).to(self.device)
        con.load_state_dict(torch.load(cfg.connector_exp_path, map_location=lambda s, l: s)["state_dict"])
        self.con = con.eval()
        # official protocol direction source (commented line in inference.py)
        self.ds = Dataset("test", dataset_root=FPS25_CMET, T=self.T, mode="mean", direction="average",
                          num_samples=10, audio_encoder="emotion2vec+large")
        self.dir_cache = {}
        self.size = 256

    def direction(self, emo, level):
        k = (emo, level)
        if k not in self.dir_cache:
            e2v, _, _ = self.ds.get_raw_e2v(emotion_1="neutral", emotion_2=emo, intensity=level)
            self.dir_cache[k] = e2v
        return self.dir_cache[k]

    def generate(self, src_img, src_video, src_wav, emo, level, out_path, workdir):
        torch = self.torch
        self.I.fix_seed(42)
        e2v = self.direction(emo, level).unsqueeze(0).unsqueeze(0).to(self.device)
        img_source = self.ip(src_img, self.size).cuda()
        audio, audio_bs, audio_T = self.ap(src_wav, device=self.device)
        lip = self.a2l(audio, audio_bs, audio_T)[0]
        lip = self.cf(lip, k_size=3, sigma=1).to(self.device)
        pose, fps = self.vp(src_video); pose = pose.to(self.device)
        len_pose, lip_len = pose.shape[1], lip.size(0)
        srcv, fps = self.vp(src_video); srcv = srcv.to(self.device)
        vid_len = srcv.shape[1] - srcv.shape[1] % self.T
        T = self.T
        with torch.no_grad():
            b = srcv.view(-1, srcv.size(2), srcv.size(3), srcv.size(4))
            ED_neu, _, _ = self.gen.compute_alpha_D(b)
            ED_neu = ED_neu.unsqueeze(0).to(self.device)
        ED_ref_T = torch.zeros((1, T, ED_neu.size(2))).to(self.device)
        preds = []
        with torch.no_grad():
            for i in range(0, vid_len, T):
                ED_neu_T = ED_neu[:, i:i + T, :]
                d, _ = self.con(ED_ref_T, e2v, ED_neu_T)
                preds.append(ED_neu_T.squeeze(0) + d)
                ED_ref_T = d.unsqueeze(0)
        exp = torch.cat(preds, dim=0).unsqueeze(0)
        exp = exp[:, :-20]
        while exp.shape[1] < lip_len:
            exp = torch.cat([exp, torch.flip(exp, dims=[1])], dim=1)
        exp = exp[:lip_len]
        exp_len = exp.shape[1]
        frames = []
        with torch.no_grad():
            for i in range(lip_len):
                tl = lip[i:i + 1]
                tp = pose[:, -1] if i >= len_pose else pose[:, i]
                ae = exp[:, -1, :] if i >= exp_len else exp[:, i, :]
                frames.append(self.gen.test_EDTalk_AV_use_exp_weight(img_source, tl, tp, ae, h_start=None).unsqueeze(2))
        vid = torch.cat(frames, dim=2)
        tmp_v = os.path.join(workdir, "tmp_video.mp4")
        tmp_o = os.path.join(workdir, "tmp_out.mp4")
        for p in (tmp_v, tmp_o):
            if os.path.exists(p):
                os.remove(p)
        self.sv(vid, tmp_v, fps)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_v, "-i", src_wav,
                        "-vcodec", "copy", tmp_o], check=True)
        os.replace(tmp_o, out_path)
        os.remove(tmp_v)


def source_frame(src_video, cache_dir):
    p = src_video.replace("\\", "/").split("/")
    png = os.path.join(cache_dir, f"{p[-5]}_{p[-1][:-4]}.png")
    if not os.path.isfile(png):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = png + ".tmp.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_video, "-vf",
                        "select=eq(n\\,0)", "-vframes", "1", tmp], check=True)
        os.replace(tmp, png)
    return png

def generate_all(meta, gen, outdir, allowed_levels, limit=None, only_levels=None, logfh=None):
    rows = list(csv.DictReader(open(os.path.join(meta, "test_present.csv"), newline="")))
    levels = set(only_levels or allowed_levels)
    todo = [r for r in rows if r["intensity"] in levels]
    skipped_level = len(rows) - len(todo)
    gdir = os.path.join(outdir, "gen"); work = os.path.join(outdir, "work")
    os.makedirs(gdir, exist_ok=True); os.makedirs(work, exist_ok=True)
    pending = [r for r in todo if not probe_ok(os.path.join(gdir, out_name(r)))[0]]
    already = len(todo) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    ok_n, fails, t0 = 0, [], time.time()
    for i, r in enumerate(pending, 1):
        out = os.path.join(gdir, out_name(r))
        try:
            img = source_frame(r["source_video_path"], os.path.join(outdir, "src_frames"))
            gen.generate(img, r["source_video_path"], r["source_video_path"][:-4] + ".wav",
                         r["gt_emotion"], r["intensity"], out, work)
            good, why = probe_ok(out)
            if not good:
                raise RuntimeError(f"invalid output: {why}")
            ok_n += 1
        except Exception as e:
            fails.append((out_name(r), f"{type(e).__name__}: {e}"))
            if os.path.exists(out) and not probe_ok(out)[0]:
                os.remove(out)
        if i % 10 == 0 or i == len(pending):
            el = time.time() - t0
            eta = el / i * (len(pending) - i)
            log(f"[gen] {i}/{len(pending)} ok={ok_n} fail={len(fails)} "
                f"({el / i:.1f}s/row, ETA {eta / 60:.0f} min)", logfh)
    with open(os.path.join(outdir, "gen_failures.txt"), "w") as f:
        for n, why in fails:
            f.write(f"{n}\t{why}\n")
    # ours.csv: every row whose output is valid
    with open(os.path.join(outdir, "ours.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEST_COLS + ["generated_path", "source_audio_path"])
        w.writeheader()
        n_rows = 0
        for r in todo:
            out = os.path.join(gdir, out_name(r))
            if probe_ok(out)[0]:
                w.writerow({**{k: r[k] for k in TEST_COLS}, "generated_path": out,
                            "source_audio_path": r["source_video_path"][:-4] + ".wav"})
                n_rows += 1
    return {"rows": len(rows), "in_allowed_levels": len(todo), "skipped_by_level": skipped_level,
            "already_done": already, "processed": len(pending), "ok": ok_n, "failed": len(fails),
            "ours_csv_rows": n_rows}


# -------------------------------------------------------------- driver -----

def import_sibling(name):
    sys.path.insert(0, HERE)
    return __import__(name)

def run_chain(args, gen_factory=None, fps25_root=FPS25, meta=META):
    st = State(os.path.join(meta, "overnight_state.json"))
    if getattr(args, "redo_gate", False):
        if st.get("S3") == "done":
            raise RuntimeError("--redo_gate refused: renumbering already applied")
        for k in ("S2", "gate", "remap_file", "S4", "S5"):
            st.d.pop(k, None)
        st.set("redo_gate_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    logfh = open(os.path.join(meta, "overnight.log"), "a")
    log("=== overnight chain start ===", logfh)
    try:
        # S1
        if st.get("S1") != "done":
            if not args.skip_transcribe:
                while subprocess.run(["pgrep", "-f", "transcribe_mead.py"], capture_output=True).returncode == 0:
                    log("[S1] transcription still running, waiting 60s", logfh); time.sleep(60)
                tm = import_sibling("transcribe_mead")
                wavs = tm.discover(fps25_root)
                done = tm.load_done(os.path.join(meta, "transcripts_all.csv"))
                rels = {os.path.relpath(w, fps25_root) for w in wavs}
                if not rels <= set(done):
                    s = tm.run(fps25_root, tm.RealBackend("small.en"), log=lambda m: log(m, logfh))
                    done = tm.load_done(os.path.join(meta, "transcripts_all.csv"))
                if not rels <= set(done):
                    raise RuntimeError(f"[S1] transcription incomplete: {len(rels - set(done))} wavs missing")
            st.set("S1", "done"); log("[S1] done", logfh)
        # S2
        if st.get("S2") != "done":
            import numpy as np
            if args.canonical:
                cr = import_sibling("canonical_remap")
                rep = cr.run(fps25_root, args.audios, meta, np, log=lambda m: log(m, logfh))
                used = os.path.join(meta, "remap_used.json")
                shutil.copy2(os.path.join(meta, "canonical_remap.json"), used)
            else:
                al = import_sibling("align_mead_levels")
                rep = al.run(fps25_root, args.audios, os.path.join(meta, "transcripts_all.csv"),
                             os.path.join(meta, "level_remap.json"), np=np, log=lambda m: log(m, logfh))
                used = os.path.join(meta, "level_remap.json")
            st.set("remap_file", used)
            gate = decide_gate(rep, fps25_root, args.min_agree, accept_weak=args.accept_weak)
            if args.canonical:
                vc = rep["validation_c_test_pairs"]["after"]
                gaps = {e: v for e, v in rep["table_missing_numbers"].items() if v}
                extra = []
                if vc["bad"] > args.max_bad_pairs:
                    extra.append(f"validation C: {vc['bad']} test pairs with mismatched content > {args.max_bad_pairs}")
                if gaps:
                    extra.append(f"canonical table gaps: {gaps}")
                if extra:
                    gate = {**gate, "passed": False, "rename_groups": [], "allowed_levels": [],
                            "reasons": gate["reasons"] + extra}
            st.set("gate", gate); st.set("S2", "done")
            log(f"[S2] gate: passed={gate['passed']} rename={len(gate['rename_groups'])} "
                f"allowed={gate['allowed_levels']} reasons={gate['reasons']}", logfh)
        gate = st.get("gate")
        # S3 + S3b
        if gate["passed"] and st.get("S3") != "done":
            ok, why = ledgers_complete(fps25_root, meta)
            if not ok:
                raise RuntimeError(f"[S3] refuse to renumber: {why}")
            rep = json.load(open(st.get("remap_file")))
            remaps = {k: rep["groups"][k]["map"] for k in gate["rename_groups"]}
            for k, m in remaps.items():
                log(f"[S3] {k}: {apply_group(fps25_root, meta, k, m)}", logfh)
            rewrite_ledgers_and_transcripts(fps25_root, meta, remaps)
            st.set("S3", "done")
        if gate["passed"] and st.get("S3") == "done" and st.get("S3b") != "done":
            rep = json.load(open(st.get("remap_file")))
            remaps = {k: rep["groups"][k]["map"] for k in gate["rename_groups"]}
            import numpy as np
            if args.canonical:
                cr = import_sibling("canonical_remap")
                rep2 = cr.run(fps25_root, args.audios, meta, np, log=lambda m: log(m, logfh))
                shutil.copy2(os.path.join(meta, "canonical_remap.json"), os.path.join(meta, "canonical_remap_after.json"))
                vc2 = rep2["validation_c_test_pairs"]["before"]      # raw file numbers must now be right
            else:
                al = import_sibling("align_mead_levels")
                rep2 = al.run(fps25_root, args.audios, os.path.join(meta, "transcripts_all.csv"),
                              os.path.join(meta, "level_remap_after.json"), np=np, log=lambda m: log(m, logfh))
                vc2 = {"bad": 0}
            bad = [k for k, g in rep2["groups"].items() if g["verdict"] != "identity"]
            va2 = rep2.get("validation_a_authors_e2v", {})
            if bad or va2.get("disagree", 1) != 0 or vc2["bad"] > args.max_bad_pairs:
                log(f"[S3b] POST-VERIFY FAILED (non-identity={bad[:5]}, A={va2}); rolling back", logfh)
                for k in remaps:
                    log(f"[S3b] {k}: {rollback_group(fps25_root, meta, k)}", logfh)
                for name in ("crop_done.txt", "fps_done.txt", "transcripts_all.csv"):
                    p = os.path.join(meta, name)
                    if os.path.isfile(p + ".pre_renum"):
                        shutil.copy2(p + ".pre_renum", p)
                gate = {**gate, "passed": False, "rename_groups": [],
                        "reasons": gate["reasons"] + ["post-verify failed, rolled back"], "allowed_levels": []}
                st.set("gate", gate); st.set("S3", "rolled_back"); st.set("S3b", "rolled_back")
            else:
                st.set("S3b", "done")
                log("[S3b] post-verify OK: all groups identity, validation A agrees", logfh)
        # S4
        m = rebuild_manifest(meta)
        st.set("S4", m); log(f"[S4] manifest: {m}", logfh)
        # S5
        if gate["allowed_levels"]:
            gen = gen_factory() if gen_factory else RealGen(lambda x: log(x, logfh))
            g = generate_all(meta, gen, args.outdir, gate["allowed_levels"], limit=args.gen_limit, logfh=logfh)
        else:
            g = {"skipped": "gate not passed; nothing generated to avoid contaminating resumable outputs",
                 "ok": 0, "failed": 0}
            log("[S5] SKIPPED: gate not passed, no generation", logfh)
        st.set("S5", g); log(f"[S5] generation: {g}", logfh)
        # S6
        report = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "gate": st.get("gate"),
                  "S3": st.get("S3"), "manifest": st.get("S4"), "generation": st.get("S5")}
        json.dump(report, open(os.path.join(meta, "overnight_report.json"), "w"), indent=1)
        log(f"[S6] report written. gate_passed={report['gate']['passed']} gen_ok={g['ok']} gen_failed={g['failed']}", logfh)
        return report
    except Exception:
        log("CHAIN STOPPED WITH ERROR:\n" + traceback.format_exc(), logfh)
        raise
    finally:
        logfh.close()


# ------------------------------------------------------------ self-test -----

def _mk_mp4(path, frames=10, audio=True, size=256):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = ["-f", "lavfi", "-i", f"testsrc=size={size}x{size}:rate=25"]
    if audio:
        a += ["-f", "lavfi", "-i", "sine=sample_rate=16000"]
    a += ["-frames:v", str(frames), "-t", str(frames / 25), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    a += (["-c:a", "aac"] if audio else ["-an"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + a + [path], check=True)

def _tree_digest(root):
    out = {}
    for r, _, fs in os.walk(root):
        for fn in fs:
            p = os.path.join(r, fn)
            out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out

def selftest():
    import tempfile, random
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            root, meta = FPS25, META
            # --- small tree: one level_3 folder with a cyclic shift, one clean folder
            def make_clip(folder, stem, tag):
                os.makedirs(os.path.join(folder, E2V_DIR), exist_ok=True)
                for pat in STEM_FILES:
                    with open(os.path.join(folder, pat.format(s=stem)), "w") as f:
                        f.write(f"{tag}|{pat}")
            f3 = os.path.join(root, "M003", "front", "disgusted", "level_3")
            f1 = os.path.join(root, "M003", "front", "disgusted", "level_1")
            for n in range(1, 7):
                make_clip(f3, f"{n:03d}", f"clip{n}")
                make_clip(f1, f"{n:03d}", f"ref{n}")
            # T1 gate passes on clean report, maps exactly the on-disk files
            mapping = {f"{n:03d}": f"{(n + 2) % 6 + 1:03d}" for n in range(1, 7)}   # shift +3 mod 6
            rep = {"groups": {"M003/disgusted/level_3": {"verdict": "cyclic_shift_+3", "map": mapping},
                              "M003/disgusted/level_2": {"verdict": "identity", "map": {}}},
                   "weak_matches": [], "validation_a_authors_e2v": {"agree": 200, "disagree": 0},
                   "validation_b_common_sentences": {"misassigned": 0}}
            g = decide_gate(rep, root, 150)
            assert g["passed"] and g["rename_groups"] == ["M003/disgusted/level_3"], g
            # T2 gate fails safely on any disagreement -> level_1(+2) only, nothing renamed
            bad = json.loads(json.dumps(rep)); bad["validation_a_authors_e2v"]["disagree"] = 1
            gb = decide_gate(bad, root, 150)
            assert not gb["passed"] and gb["rename_groups"] == [] and gb["allowed_levels"] == []
            bad2 = json.loads(json.dumps(rep)); bad2["weak_matches"] = [{"group": "M003/disgusted/level_3", "ours": "002"}]
            assert not decide_gate(bad2, root, 150)["passed"]
            assert decide_gate(bad2, root, 150, accept_weak=["M003/disgusted/level_3:002"])["passed"]
            assert not decide_gate(bad2, root, 150, accept_weak=["M003/disgusted/level_3:003"])["passed"]
            bad3 = json.loads(json.dumps(rep)); bad3["groups"]["M003/disgusted/level_3"]["map"].pop("006")
            assert not decide_gate(bad3, root, 150)["passed"]             # mapping must cover disk
            # T3 apply: renamed correctly, original kept OUTSIDE the tree, no glob leakage
            before = _tree_digest(f3)
            assert apply_group(root, meta, "M003/disgusted/level_3", mapping) == "applied"
            after = _tree_digest(f3)
            for old, new in mapping.items():
                for pat in STEM_FILES:
                    assert after[pat.format(s=new)] == before[pat.format(s=old)], (old, new, pat)
            leak = glob.glob(os.path.join(root, "*", "front", "*", "*", "*.mp4"))
            assert not any("renum" in p for p in leak) and len(leak) == 12
            # T4 idempotent re-apply
            assert apply_group(root, meta, "M003/disgusted/level_3", mapping) == "already_applied"
            assert _tree_digest(f3) == after
            # T5 rollback restores the exact original tree
            assert rollback_group(root, meta, "M003/disgusted/level_3") == "rolled_back"
            assert _tree_digest(f3) == before
            # T6 crash between the two renames is recovered (live missing, work present)
            build_renamed(f3, os.path.join(meta, "renum_work", "M003__disgusted__level_3"), mapping)
            os.rename(f3, os.path.join(meta, "renum_backup", "M003__disgusted__level_3"))
            assert apply_group(root, meta, "M003/disgusted/level_3", mapping) == "recovered_swap"
            assert _tree_digest(f3) == after
            rollback_group(root, meta, "M003/disgusted/level_3")
            # T7 missing stem file aborts BEFORE touching the live tree
            os.remove(os.path.join(f3, "004_ED_lip.npy"))
            try:
                apply_group(root, meta, "M003/disgusted/level_3", mapping); raise AssertionError("no abort")
            except FileNotFoundError:
                pass
            assert os.path.isdir(f3) and not os.path.isdir(os.path.join(meta, "renum_backup", "M003__disgusted__level_3"))
            with open(os.path.join(f3, "004_ED_lip.npy"), "w") as f:
                f.write("clip4|{s}_ED_lip.npy")
            # T8 ledgers + transcripts rewritten; refuse renumber when ledgers incomplete
            open(os.path.join(meta, "crop_done.txt"), "w").write("")
            assert not ledgers_complete(root, meta)[0]
            mp4 = sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "*", "front", "*", "*", "*.mp4")))
            for n in ("crop_done.txt", "fps_done.txt"):
                open(os.path.join(meta, n), "w").write("\n".join(mp4) + "\n")
            assert ledgers_complete(root, meta)[0]
            with open(os.path.join(meta, "transcripts_all.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["relpath", "transcript"])
                w.writerow(["M003/front/disgusted/level_3/001.wav", "one"])
                w.writerow(["M003/front/disgusted/level_1/001.wav", "ref"])
            rewrite_ledgers_and_transcripts(root, meta, {"M003/disgusted/level_3": mapping})
            tr = list(csv.reader(open(os.path.join(meta, "transcripts_all.csv"))))
            assert tr[1] == [f"M003/front/disgusted/level_3/{mapping['001']}.wav", "one"] and tr[2][0].endswith("level_1/001.wav")
            assert os.path.isfile(os.path.join(meta, "transcripts_all.csv.pre_renum"))

            # --- generation stage with real mp4s + mock generator
            gtree = os.path.join(td, "g"); os.makedirs(gtree); os.chdir(gtree)
            rows = []
            for n in (1, 2, 3):
                src = f"./dataset/MEAD/FPS25/M003/front/neutral/level_1/{n:03d}.mp4"
                _mk_mp4(src, audio=True)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1",
                                "-ar", "16000", src[:-4] + ".wav"], check=True)
                for lvl in ("level_1", "level_3"):
                    gt = f"./dataset/MEAD/FPS25/M003/front/angry/{lvl}/{n:03d}.mp4"
                    _mk_mp4(gt)
                    rows.append({"source_video_path": src, "gt_video_path": gt, "gt_emotion": "angry", "intensity": lvl})
            rows.append({"source_video_path": "./dataset/MEAD/FPS25/M003/front/neutral/level_1/009.mp4",
                         "gt_video_path": "./dataset/MEAD/FPS25/M003/front/angry/level_1/009.mp4",
                         "gt_emotion": "angry", "intensity": "level_1"})                    # absent clip
            os.makedirs("dataset/MEAD", exist_ok=True)
            with open("dataset/MEAD/test.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TEST_COLS); w.writeheader(); w.writerows(rows)
            # T9 manifest: absent clip goes to missing list
            m = rebuild_manifest("dataset/MEAD")
            assert m == {"total": 7, "present": 6, "missing": 1}, m

            class MockGen:
                def __init__(self, fail_on=(), bad_on=()):
                    self.calls, self.fail_on, self.bad_on = [], set(fail_on), set(bad_on)
                def generate(self, img, sv, sw, emo, lvl, out, work):
                    self.calls.append((sv, lvl))
                    assert os.path.isfile(img) and os.path.isfile(sw)
                    key = os.path.basename(out)
                    if key in self.fail_on:
                        raise RuntimeError("cuda boom")
                    _mk_mp4(out, audio=key not in self.bad_on)
            mg = MockGen(fail_on={"M003_angry_level_1_002.mp4"}, bad_on={"M003_angry_level_3_003.mp4"})
            s = generate_all("dataset/MEAD", mg, "runs/x", ["level_1", "level_3"])
            # T10 failures isolated; invalid output (no audio) rejected and removed
            assert s["processed"] == 6 and s["ok"] == 4 and s["failed"] == 2, s
            assert not os.path.exists("runs/x/gen/M003_angry_level_3_003.mp4")
            fl = open("runs/x/gen_failures.txt").read()
            assert "cuda boom" in fl and "no_audio_stream" in fl
            # T11 resume regenerates only the failed rows; ours.csv lists only valid outputs
            mg2 = MockGen()
            s2 = generate_all("dataset/MEAD", mg2, "runs/x", ["level_1", "level_3"])
            assert s2["processed"] == 2 and s2["ok"] == 2 and s2["already_done"] == 4, s2
            assert s2["ours_csv_rows"] == 6
            oc = list(csv.DictReader(open("runs/x/ours.csv")))
            assert all(probe_ok(r["generated_path"])[0] and r["source_audio_path"].endswith(".wav") for r in oc)
            # T12 level filter (gate failed): level_3 rows are not generated
            mg3 = MockGen()
            s3 = generate_all("dataset/MEAD", mg3, "runs/y", ["level_1"])
            assert s3["in_allowed_levels"] == 3 and s3["skipped_by_level"] == 3 and all(l == "level_1" for _, l in mg3.calls)
            # T13 source frame cached and reused
            assert len(glob.glob("runs/x/src_frames/*.png")) == 3
            # T14 output naming is stable and unique per gt clip
            names = {out_name(r) for r in rows}
            assert len(names) == len(rows)
        finally:
            os.chdir(cwd)
    print("selftest OK: gate pass/fail rules, atomic renumber (links, backup outside tree, no glob leakage), "
          "idempotent re-apply, exact rollback, crash recovery, abort before touching tree, "
          "ledger/transcript rewrite + refusal, manifest rebuild, generation failure isolation, "
          "invalid-output rejection, resume, level filter, frame cache, stable naming")
    return True


def main():
    ap = argparse.ArgumentParser(description="CAFE overnight Gate 1 finish + Gate 2 generation")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="generate 2 real level_1 rows and stop")
    ap.add_argument("--audios", default="audios/MEAD")
    ap.add_argument("--min_agree", type=int, default=150)
    ap.add_argument("--outdir", default="evaluation/runs/mead_cmet_repro")
    ap.add_argument("--gen_limit", type=int, default=None)
    ap.add_argument("--skip_transcribe", action="store_true")
    ap.add_argument("--canonical", action="store_true", help="use canonical_remap (authors' numbering) for S2/S3b")
    ap.add_argument("--max_bad_pairs", type=int, default=0, help="allowed test pairs with mismatched content")
    ap.add_argument("--accept_weak", nargs="*", default=[], metavar="GROUP:NNN",
                    help="weak matches reviewed by a human and accepted, e.g. M003/disgusted/level_2:008")
    ap.add_argument("--redo_gate", action="store_true", help="re-run S2 (only before any renumbering)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not (os.path.isdir("src") and os.path.isfile("configs/inference.yaml")):
        print("run from the C-MET repo root"); sys.exit(2)
    if args.smoke:
        m = rebuild_manifest(META) if not os.path.isfile(os.path.join(META, "test_present.csv")) else None
        t0 = time.time()
        gen = RealGen(print)
        print(f"[smoke] models + Dataset loaded in {time.time() - t0:.1f}s")
        s = generate_all(META, gen, os.path.join(args.outdir + "_smoke"), ["level_1"], limit=2,
                         only_levels=["level_1"])
        print(f"[smoke] {s} total {time.time() - t0:.1f}s")
        sys.exit(0 if s["ok"] == 2 else 1)
    run_chain(args)


if __name__ == "__main__":
    main()
