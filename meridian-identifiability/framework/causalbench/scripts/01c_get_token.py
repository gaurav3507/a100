import json
from playwright.sync_api import sync_playwright

URL = "https://plus.figshare.com/ndownloader/files/35773219"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

with sync_playwright() as p:
    b = p.chromium.launch(headless=False, channel="chromium",
                          args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
    ctx = b.new_context(user_agent=UA, viewport={"width": 1280, "height": 900},
                        locale="en-US", accept_downloads=False)
    pg = ctx.new_page()
    try:
        pg.goto(URL, wait_until="domcontentloaded", timeout=120000)
    except Exception as e:
        print("nav ended:", type(e).__name__)
    for i in range(6):
        pg.wait_for_timeout(5000)
        names = [c["name"] for c in ctx.cookies()]
        print(f"t+{(i+1)*5}s cookies:", names, flush=True)
        if any("waf" in n.lower() for n in names):
            break
    json.dump(ctx.cookies(), open("data/cookies.json", "w"))
    b.close()
