import os, sys
from playwright.sync_api import sync_playwright
from curl_cffi import requests

D = "/workspace/meridian-identifiability/causalbench/data"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
FILES = {
    "k562.h5ad": "https://plus.figshare.com/ndownloader/files/35773219",
    "rpe1.h5ad": "https://plus.figshare.com/ndownloader/files/35775606",
}

def get_token(url):
    with sync_playwright() as p:
        b = p.chromium.launch(headless=False, channel="chromium",
                              args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
        ctx = b.new_context(user_agent=UA, viewport={"width":1280,"height":900},
                            locale="en-US", accept_downloads=False)
        pg = ctx.new_page()
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=120000)
        except Exception:
            pass
        tok = None
        for _ in range(8):
            pg.wait_for_timeout(5000)
            for c in ctx.cookies():
                if c["name"] == "aws-waf-token":
                    tok = c["value"]
            if tok: break
        b.close()
        return tok

for name, url in FILES.items():
    out = os.path.join(D, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"SKIP {name} ({os.path.getsize(out)/1e9:.2f} GB)", flush=True); continue

    print(f"=== {name}: getting token ===", flush=True)
    tok = get_token(url)
    if not tok: sys.exit(f"ABORT: no token for {name}")
    print("token ok", flush=True)

    tmp = out + ".tmp"
    r = requests.get(url, impersonate="chrome",
                     headers={"User-Agent": UA},
                     cookies={"aws-waf-token": tok},
                     stream=True, allow_redirects=True, timeout=None)
    print("status:", r.status_code, "len:", r.headers.get("content-length"), flush=True)
    if r.status_code != 200: sys.exit(f"ABORT: {name} -> {r.status_code}")

    got = 0
    with open(tmp, "wb") as f:
        for c in r.iter_content(chunk_size=1 << 22):
            f.write(c); got += len(c)
            if got % (1 << 30) < (1 << 22):
                print(f"  {got/1e9:.2f} GB", flush=True)
    r.close()
    with open(tmp, "rb") as f:
        if f.read(4) != b"\x89HDF": sys.exit(f"ABORT: {name} not HDF5")
    os.rename(tmp, out)
    print(f"OK {name} -> {got/1e9:.2f} GB", flush=True)

print("ALL DONE", flush=True)
