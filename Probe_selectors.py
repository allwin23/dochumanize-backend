#!/usr/bin/env python3
"""
probe_selectors.py — Probe humanizeai.pro to find the correct CSS selectors.

Run this ONCE before using humanize_doc.py to confirm which selectors work.
It submits a test sentence and prints all elements that changed after clicking Humanize.

Usage:
    python3 probe_selectors.py
"""

import asyncio
from playwright.async_api import async_playwright

SITE_URL = "https://www.humanizeai.pro/"
TEST_TEXT = (
    "The implementation of this methodology serves as a testament to the "
    "transformative potential of modern approaches, highlighting the crucial "
    "role that innovation plays in today's rapidly evolving technological landscape."
)

async def probe():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # visible so you can watch
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        print(f"Opening {SITE_URL}...")
        await page.goto(SITE_URL, wait_until="networkidle", timeout=30_000)
        await page.screenshot(path="/tmp/probe_01_loaded.png")

        # Dump all textareas and contenteditable divs
        print("\n── Inputs on page ──")
        inputs = await page.query_selector_all("textarea, [contenteditable='true']")
        for el in inputs:
            tag = await el.evaluate("e => e.tagName")
            cls = await el.evaluate("e => e.className")
            placeholder = await el.evaluate("e => e.placeholder || ''")
            print(f"  <{tag}> class='{cls}' placeholder='{placeholder}'")

        # Try to fill the first textarea
        try:
            await page.fill("textarea", TEST_TEXT)
            print(f"\nFilled textarea with test text.")
        except Exception as e:
            print(f"Could not fill textarea: {e}")

        # Dump all buttons
        print("\n── Buttons on page ──")
        buttons = await page.query_selector_all("button")
        for btn in buttons:
            txt = await btn.inner_text()
            cls = await btn.evaluate("e => e.className")
            print(f"  Button: '{txt.strip()[:60]}' class='{cls[:60]}'")

        # Capture snapshot of all text-containing divs before clicking
        before_texts = {}
        divs = await page.query_selector_all("div, p, span")
        for el in divs:
            try:
                txt = await el.inner_text()
                if txt and txt.strip():
                    sel = await el.evaluate("e => e.tagName + (e.id ? '#'+e.id : '') + (e.className ? '.'+e.className.split(' ')[0] : '')")
                    before_texts[sel] = txt.strip()[:100]
            except Exception:
                pass

        # Click Humanize
        print("\nLooking for Humanize button...")
        for btn_text in ["Humanize", "humanize", "HUMANIZE", "Submit", "Generate"]:
            try:
                btn = page.get_by_role("button", name=btn_text)
                if await btn.count() > 0:
                    print(f"Clicking '{btn_text}' button...")
                    await btn.click()
                    break
            except Exception:
                pass

        print("Waiting 10s for output...")
        await asyncio.sleep(10)
        await page.screenshot(path="/tmp/probe_02_after_click.png")

        # Find changed elements
        print("\n── Elements that changed (possible output selectors) ──")
        divs = await page.query_selector_all("div, p, span, textarea")
        for el in divs:
            try:
                txt = await el.inner_text()
                if txt and txt.strip() and len(txt.strip()) > 20:
                    sel = await el.evaluate("e => e.tagName + (e.id ? '#'+e.id : '') + (e.className ? '.'+e.className.split(' ')[0] : '')")
                    if before_texts.get(sel) != txt.strip()[:100]:
                        cls_full = await el.evaluate("e => e.className")
                        print(f"\n  Selector hint: '{cls_full[:80]}'")
                        print(f"  Text preview: {txt.strip()[:150]}")
            except Exception:
                pass

        print("\n\nScreenshots saved to /tmp/probe_01_loaded.png and /tmp/probe_02_after_click.png")
        print("Check them if selectors didn't match.\n")
        await browser.close()

asyncio.run(probe())