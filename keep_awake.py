from playwright.sync_api import sync_playwright
import os, sys

STREAMLIT_URL = os.environ.get("STREAMLIT_URL", "https://cbc253ecnemb4aggdm6ene.streamlit.app/")
VISIT_SECONDS = int(os.environ.get("VISIT_SECONDS", "60"))

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(STREAMLIT_URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(5000)

    wake_btn = page.get_by_role("button", name="Yes, get this app back up!")
    if wake_btn.count() > 0:
        print(f"WAKING UP {STREAMLIT_URL}")
        wake_btn.click()
        page.wait_for_timeout(60_000)
    else:
        print(f"ALREADY AWAKE {STREAMLIT_URL}")

    page.wait_for_timeout(VISIT_SECONDS * 1000)
    browser.close()   
