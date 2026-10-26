#!/usr/bin/env python3
"""Minimal CDP harness: interact with the VeloBench SPA and capture screenshots
in ONE browser session (the app streams live stats only over the WS that sent
the request, so load-then-shoot tools cannot capture a live turn)."""
import asyncio, base64, json, os, shutil, subprocess
import websockets

CHROME = "/usr/bin/chromium-browser"
OUT = "$HOME/projects/VeloBench/website/assets/img/shots"

class CDP:
    def __init__(self, ws):
        self.ws = ws; self.id = 0; self.sid = None
        self.pending = {}; self.events = []; self.logs = []
        self._pump = None
    async def start_pump(self):
        async def pump():
            while True:
                try:
                    m = json.loads(await self.ws.recv())
                except Exception:
                    return
                mid = m.get("id")
                if mid and mid in self.pending:
                    self.pending[mid].set_result(m)
                    continue
                meth = m.get("method")
                if meth == "Page.javascriptDialogOpening":
                    self.logs.append("DIALOG: " + m["params"].get("message", "")[:200])
                    self.id += 1
                    await self.ws.send(json.dumps({"id": self.id, "method":
                        "Page.handleJavaScriptDialog", "params": {"accept": True},
                        "sessionId": self.sid}))
                elif meth == "Runtime.consoleAPICalled":
                    self.logs.append("CONSOLE: " + " ".join(
                        a.get("value", "") for a in m["params"].get("args", []))[:200])
                elif meth == "Runtime.exceptionThrown":
                    self.logs.append("EXC: " + str(m["params"].get("exceptionDetails", {})
                        .get("exception", {}).get("description", ""))[:200])
                else:
                    self.events.append(meth)
        self._pump = asyncio.create_task(pump())
    async def send(self, method, params=None, session=False, timeout=45):
        self.id += 1
        msg = {"id": self.id, "method": method, "params": params or {}}
        if session and self.sid: msg["sessionId"] = self.sid
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.id] = fut
        await self.ws.send(json.dumps(msg))
        try:
            r = await asyncio.wait_for(fut, timeout)
        finally:
            self.pending.pop(self.id, None)
        if "error" in r: raise RuntimeError(f"{method}: {r['error']}")
        return r.get("result", {})
    async def evaluate(self, expr, await_promise=False):
        r = await self.send("Runtime.evaluate", {
            "expression": expr, "returnByValue": True, "awaitPromise": await_promise}, session=True)
        if r.get("exceptionDetails"): raise RuntimeError(r["exceptionDetails"])
        return r.get("result", {}).get("value")
    async def shoot(self, path, full=False):
        r = await self.send("Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": full}, session=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f: f.write(base64.b64decode(r["data"]))
        print("saved", path, flush=True)
    async def navigate(self, url, wait=2.5):
        await self.send("Page.navigate", {"url": url}, session=True)
        await asyncio.sleep(wait)

async def main(recipe):
    os.makedirs(OUT, exist_ok=True)
    port = 9300 + (os.getpid() % 80)
    profile = f"/tmp/opencode/cdp-{os.getpid()}"
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
        "--hide-scrollbars", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        ws_url = None
        import time
        t0 = time.time()
        while time.time() - t0 < 30:
            line = proc.stderr.readline()
            if not line: break
            if "DevTools listening on" in line:
                ws_url = line.split("ws://", 1)[1].strip()
                break
        if not ws_url: raise RuntimeError("chrome did not publish DevTools url")
        async with websockets.connect("ws://" + ws_url, max_size=64*1024*1024) as ws:
            cdp = CDP(ws)
            await cdp.start_pump()
            t = await cdp.send("Target.createTarget", {"url": "about:blank"})
            cdp.sid = (await cdp.send("Target.attachToTarget",
                {"targetId": t["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable", session=True)
            await cdp.send("Runtime.enable", session=True)
            await cdp.send("Emulation.setDeviceMetricsOverride", {
                "width": 1680, "height": 1050, "deviceScaleFactor": 1, "mobile": False}, session=True)
            await recipe(cdp)
            if cdp.logs: print("PAGE LOGS:", *cdp.logs[:25], sep="\n  ", flush=True)
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)

def run(recipe): asyncio.run(main(recipe))
