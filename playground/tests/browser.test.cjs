const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");
const root = path.join(__dirname, "..");
const WEBUI = "https://cyth.18-118-248-231.sslip.io";
const SITE = "https://celesteisyy.github.io";
const PLAYGROUND = WEBUI + "/playground/";

async function mount(t, options = {}) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const context = await browser.newContext({ serviceWorkers: "block", viewport: { width: 1280, height: 850 } });
  const page = await context.newPage();
  const calls = [], errors = [];
  const state = { authenticated: options.authenticated !== false, status: options.status, expireInFrame: false };
  page.on("pageerror", error => errors.push(error.message));
  if (state.authenticated) {
    await context.addCookies([{ name: "token", value: "fixture-session", url: WEBUI, secure: true, sameSite: "Lax" }]);
  }
  await context.route("**/*", async route => {
    const request = route.request();
    const url = new URL(request.url());
    calls.push({ url: request.url(), headers: await request.allHeaders(), top: request.frame() === page.mainFrame() });
    if ([WEBUI, SITE].includes(url.origin) && url.pathname.startsWith("/playground/")) {
      const relative = url.pathname.slice("/playground/".length) || "index.html";
      if (!["index.html", "webui.js", "webui.css"].includes(relative)) return route.fulfill({ status: 404, body: "missing" });
      return route.fulfill({ contentType: relative.endsWith(".js") ? "application/javascript" : relative.endsWith(".css") ? "text/css" : "text/html", body: fs.readFileSync(path.join(root, relative)) });
    }
    if (url.origin === WEBUI && url.pathname === "/api/v1/auths/") {
      const authenticated = state.authenticated && request.headers().cookie?.includes("token=fixture-session");
      return route.fulfill({ status: state.status || (authenticated ? 200 : 401), contentType: "application/json", body: JSON.stringify(authenticated ? { id: "owner" } : { detail: "Unauthorized" }) });
    }
    if (url.origin === WEBUI && url.pathname === "/") {
      if (state.expireInFrame) return route.fulfill({ contentType: "text/html", body: '<script>location.replace("/auth")</script>' });
      return route.fulfill({ headers: { "x-frame-options": "SAMEORIGIN", "content-security-policy": "frame-ancestors 'self'" }, contentType: "text/html", body: "<h1>Open WebUI fixture</h1><button>New chat</button>" });
    }
    if (url.origin === WEBUI && url.pathname === "/auth") {
      return route.fulfill({ contentType: "text/html", body: "<h1>Sign in with Cognito</h1>" });
    }
    return route.fulfill({ status: 404, body: "blocked destination" });
  });
  return { context, page, calls, errors, state };
}

test("the site entry opens the fixed workspace without forwarding query strings, fragments or stored credentials", async t => {
  const { page, calls, errors } = await mount(t);
  await page.goto(SITE + "/playground/?code=private-code&redirect=https://evil.example/#token=private-token");
  await page.waitForURL(PLAYGROUND);
  await page.locator("#webui").waitFor({ state: "visible" });
  assert.ok(!calls.some(call => call.url.startsWith(WEBUI) && /private-|evil\.example/.test(call.url)));
  assert.ok(!calls.some(call => /\/(?:chat_stream|vision|auth\/config)/.test(call.url)));
  assert.deepEqual(errors, []);
});

test("signed-out visitors get a top-level login with a fixed return path and no embedded login request", async t => {
  const { page, calls, errors } = await mount(t, { authenticated: false });
  await page.goto(PLAYGROUND);
  await page.locator("#signInLink").waitFor({ state: "visible" });
  assert.equal(await page.locator("#webui").getAttribute("src"), null);
  await page.getByRole("link", { name: "Sign in", exact: true }).click();
  await page.waitForURL(WEBUI + "/auth?redirect=%2Fplayground%2F");
  assert.ok(calls.filter(call => call.url.includes("/auth?")).every(call => call.top));
  assert.deepEqual(errors, []);
});

test("same-origin embedding preserves Lax session cookies and navigation on desktop and mobile", async t => {
  const { page, calls, errors } = await mount(t);
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 850 });
    await page.goto(PLAYGROUND);
    await page.locator("#webui").waitFor({ state: "visible" });
    assert.equal(await page.frameLocator("#webui").getByRole("button", { name: "New chat" }).count(), 1);
    for (const [name, href] of [["Home", SITE + "/"], ["Blogs", SITE + "/blogs/"]]) {
      assert.equal(await page.getByRole("link", { name, exact: true }).getAttribute("href"), href);
    }
    const box = await page.locator("#webui").boundingBox();
    assert.ok(box.width > 300 && box.height > 500, JSON.stringify(box));
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  }
  assert.ok(calls.some(call => call.url === WEBUI + "/" && !call.top && call.headers.cookie?.includes("token=fixture-session")));
  assert.deepEqual(errors, []);
});

test("an embedded login page returns authentication to the top-level window", async t => {
  const { page, calls, state, errors } = await mount(t);
  state.expireInFrame = true;
  await page.goto(PLAYGROUND);
  await page.waitForURL(WEBUI + "/auth?redirect=%2Fplayground%2F");
  assert.ok(calls.some(call => call.url === WEBUI + "/auth?redirect=%2Fplayground%2F" && call.top));
  assert.deepEqual(errors, []);
});

test("service failures offer retry without claiming authentication expired", async t => {
  const { page, state } = await mount(t, { status: 503 });
  await page.goto(PLAYGROUND);
  await page.getByText("The workspace is unavailable. Please try again.").waitFor();
  assert.equal(await page.locator("#signInLink").isVisible(), false);
  state.status = 200;
  await page.getByRole("button", { name: "Try again" }).click();
  await page.locator("#webui").waitFor({ state: "visible" });
});

test("CSP blocks foreign frames, outbound fetches, and injected scripts", async t => {
  const { page, calls } = await mount(t);
  await page.goto(PLAYGROUND);
  await page.locator("#webui").waitFor({ state: "visible" });
  const result = await page.evaluate(async () => {
    const script = document.createElement("script");
    script.textContent = "window.__attack = true";
    document.body.appendChild(script);
    const frame = document.createElement("iframe");
    frame.src = "https://evil.example/steal";
    document.body.appendChild(frame);
    let blocked = false;
    try { await fetch("https://evil.example/steal"); } catch { blocked = true; }
    return { blocked, ran: !!window.__attack };
  });
  assert.deepEqual(result, { blocked: true, ran: false });
  assert.ok(!calls.some(call => call.url.startsWith("https://evil.example/")));
});
