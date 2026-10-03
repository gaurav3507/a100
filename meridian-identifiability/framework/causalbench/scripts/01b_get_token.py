import json
from playwright.sync_api import sync_playwright

URL = "https://plus.figshare.com/ndownloader/files/35773219"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(user_agent=UA, accept_downloads=False)
    pg = ctx.new_page()
    try:
        pg.goto(URL, wait_until="domcontentloaded", timeout=120000)
    except Exception as e:
        print("nav ended (expected if it started a download):", type(e).__name__)
    pg.wait_for_timeout(15000)
    cookies = ctx.cookies()
    print("cookies:", [c["name"] for c in cookies])
    json.dump(cookies, open("data/cookies.json", "w"))
    print("saved data/cookies.json")
    b.close()
