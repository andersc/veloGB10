# tools/

`cdp_shot.py` — a tiny Chrome DevTools Protocol harness used to capture the product
screenshots and UI fragments in `assets/img/`. It exists because VeloBenchmark streams
live stats only over the WebSocket that sent the request, so "load page, wait, shoot"
tools cannot capture a live turn; this harness interacts and captures in ONE session.

Requires: `chromium-browser` on PATH and `pip install websockets pillow`.

Pattern (see /tmp recipes historically; write a new driver per need):

    import asyncio, sys
    sys.path.insert(0, "tools")
    from cdp_shot import run, OUT
    async def recipe(cdp):
        await cdp.navigate("http://127.0.0.1:13843/chat", 3)
        await cdp.evaluate("...js...")          # interact (Enter keydown sends)
        await cdp.shoot(f"{OUT}/name.png")       # viewport
        await cdp.shoot(f"{OUT}/name.png", full=True)  # tall-viewport full page
    run(recipe)

Gotchas learned the hard way:
- Send a chat message by dispatching a keydown Enter on the textarea (a synthetic
  .click() on Send can hit the button before Angular enables it).
- The Stop button's text is "⏹ Stop"; match with substring, not equality.
- For full-page captures of the app, override the device metrics to a tall viewport
  (the app scrolls an inner container, so captureBeyondViewport does nothing).
- UI fragments in assets/img/ui/ were cut with Page.captureScreenshot clip rects
  (element bounding boxes) or PIL crops of the 1680x1050 shots.
