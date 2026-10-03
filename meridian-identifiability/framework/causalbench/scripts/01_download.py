import os, sys
from causalscbench.data_access.datasets import download_weissmann

OUT = "/workspace/meridian-identifiability/causalbench/data"
os.makedirs(OUT, exist_ok=True)

st = os.statvfs(OUT)
free_gb = st.f_bavail * st.f_frsize / 1e9
print(f"free space on data dir: {free_gb:.1f} GB", flush=True)
if free_gb < 40:
    sys.exit("ABORT: less than 40GB free")

for name, fn in [("k562", download_weissmann.download_weissmann_k562),
                 ("rpe1", download_weissmann.download_weissmann_rpe1),
                 ("summary_stats", download_weissmann.download_summary_stats)]:
    print(f"=== {name} ===", flush=True)
    p = fn(OUT)
    print(f"{name} -> {p}  ({os.path.getsize(p)/1e9:.2f} GB)", flush=True)

print("DONE", flush=True)
