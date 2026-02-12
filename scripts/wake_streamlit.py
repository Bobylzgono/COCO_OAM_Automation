import os
import re
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://oamcoco-automation-boldsukh.streamlit.app/")

WAKE_BUTTON_RE = re.compile(r"(Yes,\s*get this app back up|get this app back up|wake up)", re.IGNORECASE)
SLEEP_TEXT_RE = re.compile(r"(This app has gone to sleep|gone to sleep|hibernat|zzzz|inactivity)", re.IGNORECASE)

def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(20_000)

        try:
            print(f"[wake] Opening: {APP_URL}")
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

            body = ""
            try:
                body = page.locator("body").inner_text(timeout=5_000)
            except Exception:
                pass

            wake_btn = page.get_by_role("button", name=WAKE_BUTTON_RE)
            is_sleep = bool(SLEEP_TEXT_RE.search(body)) or (wake_btn.count() > 0)

            if not is_sleep:
                print("[wake] App looks awake. (No action)")
                return 0

            if wake_btn.count() > 0:
                print("[wake] Sleep detected -> clicking wake button.")
                wake_btn.first.click(timeout=20_000)
                page.wait_for_timeout(15_000)  
            else:
                print("[wake] Sleep detected but wake button not found (UI changed?).")

            try:
                page.screenshot(path="keepawake_last.png", full_page=True)
            except Exception as e:
                print(f"[wake] Screenshot failed: {e}")

            return 0

        finally:
            browser.close()

if __name__ == "__main__":
    raise SystemExit(main())
