const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { webcrypto, createHash } = require("node:crypto");

const root = path.join(__dirname, "..");
const API = "https://api.example.test";
const CALLBACK = "https://celesteisyy.github.io/playground/";
const ISSUER = "https://cognito-idp.us-east-2.amazonaws.com/us-east-2_example";
const HOSTED = "https://example.auth.us-east-2.amazoncognito.com";
const CONFIG = { mode: "cognito", authority: ISSUER, client_id: "client123", redirect_uri: CALLBACK, scope: "openid email" };

class Storage {
  map = new Map();
  get length() { return this.map.size; }
  key(index) { return [...this.map.keys()][index] ?? null; }
  getItem(key) { return this.map.get(key) ?? null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

function setup({ href = CALLBACK, storage = new Storage(), handler, clock = { now: Date.now() } } = {}) {
  const calls = [], redirects = [], replacements = [], errors = [];
  const events = new Map();
  const navigate = href => { redirects.push(href); queueMicrotask(() => events.get("pageshow")?.()); };
  const url = new URL(href);
  class ClockDate extends Date {
    static now() { return clock.now; }
  }
  const timer = (fn, ms) => { const value = setTimeout(fn, ms); value.unref(); return value; };
  const interval = (fn, ms) => { const value = setInterval(fn, ms); value.unref(); return value; };
  const window = {
    location: { href, origin: url.origin, protocol: url.protocol, assign: navigate, replace: navigate },
    history: { replaceState: (_, __, value) => replacements.push(value) },
    sessionStorage: storage,
    crypto: webcrypto,
    addEventListener(name, callback) { events.set(name, callback); }, removeEventListener() {}, stop() {},
    setTimeout: timer, clearTimeout, setInterval: interval, clearInterval,
    fetch: async (url, options = {}) => {
      calls.push({ url: String(url), options });
      if (handler) return handler(String(url), options);
      if (url === `${API}/auth/config`) return Response.json(CONFIG);
      if (url === `${ISSUER}/.well-known/openid-configuration`) return Response.json({
        issuer: ISSUER, authorization_endpoint: `${HOSTED}/oauth2/authorize`, token_endpoint: `${HOSTED}/oauth2/token`,
        revocation_endpoint: `${HOSTED}/oauth2/revoke`
      });
      return Response.json({ authenticated: true });
    }
  };
  window.self = window;
  window.top = window;
  window.parent = window;
  const context = vm.createContext({ window, console, URL, URLSearchParams, Headers, Request, Response, AbortSignal, AbortController,
    crypto: webcrypto, TextEncoder, TextDecoder, atob, btoa, navigator: {}, Date: ClockDate,
    setTimeout: timer, clearTimeout, setInterval: interval, clearInterval,
    fetch: (...args) => window.fetch(...args) });
  vm.runInContext(fs.readFileSync(path.join(root, "vendor/oidc-client-ts-3.5.0.js"), "utf8"), context);
  window.oidc = context.oidc;
  vm.runInContext(fs.readFileSync(path.join(root, "auth.js"), "utf8"), context);
  const auth = window.CyssieAuth.create(API, { onError: value => errors.push(value) });
  return { auth, window, calls, redirects, replacements, errors, storage, clock };
}

async function startLogin() {
  const app = setup();
  await app.auth.initialize();
  await app.auth.signIn();
  const url = new URL(app.redirects[0]);
  return { ...app, params: url.searchParams };
}

function tokenResponse(nonce, access = "access-token", expires = 300) {
  const encode = value => Buffer.from(JSON.stringify(value)).toString("base64url");
  // Simulate the TLS-protected token endpoint; API signature checks are covered in Python.
  const id = `${encode({ alg: "RS256" })}.${encode({ sub: "owner", nonce, iss: ISSUER, aud: "client123" })}.signature`;
  return { access_token: access, id_token: id, refresh_token: "refresh-token", token_type: "Bearer", expires_in: expires, scope: "openid email" };
}

async function finishLogin({ nonceChange, apiStatus = 200, expires = 300 } = {}) {
  const login = await startLogin();
  const metadata = { issuer: ISSUER, authorization_endpoint: `${HOSTED}/oauth2/authorize`, token_endpoint: `${HOSTED}/oauth2/token`, revocation_endpoint: `${HOSTED}/oauth2/revoke` };
  let exchange = null;
  const app = setup({
    storage: login.storage,
    clock: login.clock,
    href: `${CALLBACK}?code=example-code&state=${login.params.get("state")}`,
    handler: (url, options) => {
      if (url === `${API}/auth/config`) return Response.json(CONFIG);
      if (url.endsWith("/.well-known/openid-configuration")) return Response.json(metadata);
      if (url === `${HOSTED}/oauth2/token`) {
        exchange = new URLSearchParams(options.body);
        return Response.json(tokenResponse(nonceChange ?? login.params.get("nonce"), "access-token", expires));
      }
      if (url === `${HOSTED}/oauth2/revoke`) return new Response("", { status: 200 });
      return Response.json({ authenticated: true }, { status: apiStatus });
    }
  });
  await app.auth.initialize();
  return { ...app, login, get exchange() { return exchange; } };
}

test("only an old backend 404 enables legacy mode; token stays in memory", async () => {
  const app = setup({ handler: () => new Response("", { status: 404 }) });
  await app.auth.initialize();
  assert.equal(app.auth.mode, "legacy");
  app.auth.setLegacyToken(" secret ");
  assert.equal(await app.auth.getToken(), "secret");
  assert.equal(app.storage.length, 0);
  const next = setup({ storage: app.storage, handler: () => new Response("", { status: 404 }) });
  await next.auth.initialize();
  assert.equal(next.auth.hasSession(), false);
});

test("network failures never silently enable legacy sign-in", async () => {
  const app = setup({ handler: () => { throw new Error("offline"); } });
  await assert.rejects(app.auth.initialize());
  assert.equal(app.auth.mode, "loading");
  assert.throws(() => app.auth.setLegacyToken("secret"));
});

test("real OIDC library generates PKCE S256, state and nonce, without a client secret", async () => {
  const login = await startLogin();
  assert.equal(login.params.get("response_type"), "code");
  assert.equal(login.params.get("code_challenge_method"), "S256");
  assert.ok(login.params.get("state"));
  assert.equal(login.params.get("nonce").length, 64);
  assert.equal(login.params.has("client_secret"), false);
});

test("callback is scrubbed, PKCE code exchanged and access token used for API", async () => {
  const app = await finishLogin();
  assert.deepEqual(app.replacements, ["/playground/"]);
  assert.equal(app.auth.hasSession(), true);
  assert.equal(app.exchange.get("code"), "example-code");
  assert.equal(app.exchange.get("client_secret"), null);
  const challenge = createHash("sha256").update(app.exchange.get("code_verifier")).digest("base64url");
  assert.equal(challenge, app.login.params.get("code_challenge"));
  const request = app.calls.find(call => call.url === `${API}/auth/me`);
  assert.equal(request.options.headers.get("Authorization"), "Bearer access-token");
  assert.equal(request.options.redirect, "error");
  assert.equal(request.options.credentials, "omit");
});

test("unknown callback state is rejected without token exchange", async () => {
  const app = setup({ href: `${CALLBACK}?code=stolen&state=wrong` });
  await app.auth.initialize();
  assert.equal(app.auth.hasSession(), false);
  assert.equal(app.calls.some(call => call.url.endsWith("/oauth2/token")), false);
  assert.equal(app.errors.length, 1);
  assert.deepEqual(app.replacements, ["/playground/"]);
});

test("nonce mismatch rejects the returned session", async () => {
  const app = await finishLogin({ nonceChange: "wrong" });
  assert.equal(app.auth.hasSession(), false);
  assert.equal(app.calls.some(call => call.url === `${API}/auth/me`), false);
  assert.equal(app.errors.length, 1);
});

test("API rejects other accounts before the UI accepts sign-in", async () => {
  await assert.rejects(finishLogin({ apiStatus: 403 }), /does not have access/);
});

test("tokens cannot be sent to external links", async () => {
  const app = await finishLogin();
  const count = app.calls.length;
  await assert.rejects(app.auth.fetch("https://evil.example/download"), /own API/);
  assert.equal(app.calls.length, count);
});

test("401 clears session and does not replay a chat POST", async () => {
  const app = await finishLogin();
  const upstream = app.window.fetch;
  let posts = 0;
  app.window.fetch = (url, options) => {
    if (url === `${API}/chat_stream`) { posts++; return Promise.resolve(new Response("", { status: 401 })); }
    return upstream(url, options);
  };
  await assert.rejects(app.auth.fetch(`${API}/chat_stream`, { method: "POST", body: "hello" }), /sign in again/);
  assert.equal(app.auth.hasSession(), false);
  assert.equal(posts, 1);
});

test("concurrent near-expiry requests refresh once", async () => {
  const app = await finishLogin();
  app.clock.now += 250000;
  let refreshes = 0;
  const upstream = app.window.fetch;
  app.window.fetch = (url, options) => {
    if (url === `${HOSTED}/oauth2/token`) {
      refreshes++;
      assert.equal(new URLSearchParams(options.body).get("grant_type"), "refresh_token");
      return Promise.resolve(Response.json(tokenResponse(undefined, "renewed-token")));
    }
    return upstream(url, options);
  };
  const tokens = await Promise.all([app.auth.getToken(), app.auth.getToken()]);
  assert.deepEqual(tokens, ["renewed-token", "renewed-token"]);
  assert.equal(refreshes, 1);
});

test("refresh failure clears expired session", async () => {
  const app = await finishLogin();
  app.clock.now += 350000;
  const upstream = app.window.fetch;
  app.window.fetch = (url, options) => url === `${HOSTED}/oauth2/token`
    ? Promise.resolve(Response.json({ error: "invalid_grant" }, { status: 400 })) : upstream(url, options);
  await assert.rejects(app.auth.getToken(), /sign in again/);
  assert.equal(app.auth.hasSession(), false);
});

test("logout clears local session, revokes refresh token and ends hosted login", async () => {
  const app = await finishLogin();
  await app.auth.signOut();
  assert.equal(app.auth.hasSession(), false);
  const logout = new URL(app.redirects[0]);
  assert.equal(logout.origin, HOSTED);
  assert.equal(logout.pathname, "/logout");
  assert.equal(logout.searchParams.get("logout_uri"), CALLBACK);
  assert.equal(app.calls.filter(call => call.url === `${HOSTED}/oauth2/revoke`).length, 1);
  assert.equal(app.storage.length, 0);
});
