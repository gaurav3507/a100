Step 0: reproduce CCRL baseline failure on A100

Runtime only. Claude Code cannot run on the A100 (university policy).
Run after the Mac scaffolding is pushed and pulled here.

  set -e
  git config --global http.version HTTP/1.1

  cd /workspace
  if [ -d /workspace/ccrl-newidea2 ]; then echo "DIR EXISTS - STOP"; exit 1; fi
  git clone https://github.com/gaurav3507/ccrl-newidea2.git
  cd /workspace/ccrl-newidea2

  # third-party benchmark + CCRL code into gitignored external/
  mkdir -p external && cd external
  git clone https://github.com/simonbing/CRLSanityCheck.git
  cd CRLSanityCheck
  git submodule update --init --recursive

  # NEW conda env, do NOT reuse dvae/cb (protect their numpy/torch)
  conda env create -f environment.yml  -n ccrl2 2>/dev/null || \
  conda env create -f environment.yaml -n ccrl2
  conda activate ccrl2

  # dataset: lt_crl_benchmark_v1 (Causal Chamber project). Follow the data-fetch
  # step in CRLSanityCheck/README.md; confirm CCRL light-tunnel files land UNDER
  # external/ (gitignored). Print the resolved data path before proceeding.

  # SMOKE TEST FIRST (small before expensive):
  # run CCRL a few steps / 1 seed on the SYNTHETIC ablation only; confirm the
  # pipeline runs end to end and emits ONE MCC. Classify any failure as
  # env / data / cuda / source-bug before continuing.

  # REAL REPRO (only if smoke passes), 5 seeds each, via nohup + disown:
  mkdir -p /workspace/ccrl-newidea2/results
  # CCRL on SYNTHETIC ablation  -> expect MCC ~0.89, SHD ~2
  # CCRL on REAL images         -> expect MCC ~0.29, SHD ~7.6
  # nohup <CCRL synthetic cmd> > results/repro_ccrl_synth.log 2>&1 &
  # disown
  # nohup <CCRL real cmd>      > results/repro_ccrl_real.log  2>&1 &
  # disown

  # Liveness: watch log/artefact count climb; do NOT trust ps|grep (Lesson 14).
  # When done, extract mean MCC + SHD over seeds for BOTH settings.
