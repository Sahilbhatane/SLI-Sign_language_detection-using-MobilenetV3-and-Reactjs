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
        # Verify Home tab navigation button is active and clickable
        frame = context.pages[-1]
        elem = frame.locator('xpath=html/body/div/div/nav/div[2]/button').nth(0)
        await page.wait_for_timeout(3000); await elem.click(timeout=5000)
        

        # Assertion: Check that the heading 'Welcome to Sign Language Interpreter' is displayed
        heading_locator = frame.locator('text=Welcome to Sign Language Interpreter')
        assert await heading_locator.is_visible(), "Heading 'Welcome to Sign Language Interpreter' is not visible"
        
        # Assertion: Verify feature cards for Detect, Translate, and History are visible
        detect_card = frame.locator('text=Real-time Detection')
        translate_card = frame.locator('text=Multi-language')
        history_card = frame.locator('text=Track History')
        assert await detect_card.is_visible(), "Feature card 'Real-time Detection' is not visible"
        assert await translate_card.is_visible(), "Feature card 'Multi-language' is not visible"
        assert await history_card.is_visible(), "Feature card 'Track History' is not visible"
        
        # Assertion: Verify Home tab navigation button is active and clickable
        home_tab_button = frame.locator('xpath=html/body/div/div/nav/div[2]/button').nth(0)
        assert await home_tab_button.is_enabled(), "Home tab navigation button is not enabled"
        assert await home_tab_button.is_visible(), "Home tab navigation button is not visible"
        await asyncio.sleep(5)
    
    finally:
        if context:
            await context.close()
        if browser:
            await browser.close()
        if pw:
            await pw.stop()
            
asyncio.run(run_test())
    