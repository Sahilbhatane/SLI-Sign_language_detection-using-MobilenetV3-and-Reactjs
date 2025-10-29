import asyncio
from playwright import async_api

async def run_test():
    pw = None
    browser = None
    context = None
    
    try:
        # Start a Playwright session in asynchronous mode
        pw = await async_api.async_playwright().start()
        
        # Launch a Chromium browser in headless mode with custom arguments
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--window-size=1280,720",         # Set the browser window size
                "--disable-dev-shm-usage",        # Avoid using /dev/shm which can cause issues in containers
                "--ipc=host",                     # Use host-level IPC for better stability
                "--single-process"                # Run the browser in a single process mode
            ],
        )
        
        # Create a new browser context (like an incognito window)
        context = await browser.new_context()
        context.set_default_timeout(5000)
        
        # Open a new page in the browser context
        page = await context.new_page()
        
        # Navigate to your target URL and wait until the network request is committed
        await page.goto("http://localhost:3000", wait_until="commit", timeout=10000)
        
        # Wait for the main page to reach DOMContentLoaded state (optional for stability)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except async_api.Error:
            pass
        
        # Iterate through all iframes and wait for them to load as well
        for frame in page.frames:
            try:
                await frame.wait_for_load_state("domcontentloaded", timeout=3000)
            except async_api.Error:
                pass
        
        # Interact with the page elements to simulate user flow
        # Click the 'Start Detection' button to attempt to start detection and check for results.
        frame = context.pages[-1]
        elem = frame.locator('xpath=html/body/div/div/div/main/div/div/div/div[2]/div/button').nth(0)
        await page.wait_for_timeout(3000); await elem.click(timeout=5000)
        

        # Try to capture a frame manually to trigger detection results and then extract detected phrases and confidence scores again.
        frame = context.pages[-1]
        elem = frame.locator('xpath=html/body/div/div/div/main/div/div/div/div[2]/div/button[2]').nth(0)
        await page.wait_for_timeout(3000); await elem.click(timeout=5000)
        

        # Stop detection as no valid detection results with confidence >= 85% appeared, then end the task.
        frame = context.pages[-1]
        elem = frame.locator('xpath=html/body/div/div/div/main/div/div/div/div[2]/div/button').nth(0)
        await page.wait_for_timeout(3000); await elem.click(timeout=5000)
        

        # Assertion: Check detected phrases and confidence scores >= 85%
        results_locator = frame.locator('xpath=//div[contains(@class, "detection-results")]//div[contains(@class, "result-item")]')
        count = await results_locator.count()
        assert count > 0, "No detection results found."
        
        previous_confidence = 100.0
        for i in range(count):
            phrase = await results_locator.nth(i).locator('.phrase').inner_text()
            confidence_text = await results_locator.nth(i).locator('.confidence').inner_text()
            confidence = float(confidence_text.strip('%'))
            assert confidence >= 85.0, f"Confidence {confidence}% for phrase '{phrase}' is below 85%."
            assert phrase != '', "Detected phrase should not be empty."
            # Check confidence ranking order
            assert confidence <= previous_confidence, f"Confidence scores are not in descending order at index {i}."
            previous_confidence = confidence
        await asyncio.sleep(5)
    
    finally:
        if context:
            await context.close()
        if browser:
            await browser.close()
        if pw:
            await pw.stop()
            
asyncio.run(run_test())
    