import asyncio
import json
from playwright.async_api import async_playwright


async def open_browser_and_wait():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        context = await browser.new_context(
            record_har_path="network_requests.har",  # Path to save the HAR file
            record_har_content="embed",  # Omit content to make the HAR file smaller
            # TODO record_har_url_filter="*",  # Optional URL filter
        )

        page = await context.new_page()

        print(
            "Browser is open. Press Enter in the terminal when you're ready to close the browser and save cookies..."
        )

        input("Press Enter to continue and close the browser...")

        cookies = await context.cookies()

        with open("cookies.json", "w") as f:
            json.dump(cookies, f, indent=4)

        await context.close()

        await browser.close()

asyncio.run(open_browser_and_wait())
