const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const root = path.join(__dirname, "..");
for (const name of ["index.html", "embed.html", "sign-in.html", "login-complete.html"]) {
  test(`${name}: exact script integrity, restricted frames, no external API access`, () => {
    const html = fs.readFileSync(path.join(root, name), "utf8");
    const match = html.match(/<meta http-equiv="Content-Security-Policy" content="([^"]+)"/);
    assert.ok(match && match.index < html.indexOf("<link "));
    const policy = new Map(match[1].split(";").map(part => {
      const [key, ...values] = part.trim().split(/\s+/); return [key, values];
    }));
    for (const key of ["default-src", "script-src-attr", "style-src-attr", "object-src", "base-uri", "form-action"]) {
      assert.deepEqual(policy.get(key), ["'none'"]);
    }
    const scripts = [...html.matchAll(/<script\s+([^>]*)><\/script>/g)];
    const hashes = scripts.map(([, attrs]) => {
      const src = attrs.match(/src="([^"]+)"/)[1].replace(/^\/playground\//, "").split("?")[0];
      const hash = "sha256-" + createHash("sha256").update(fs.readFileSync(path.join(root, src))).digest("base64");
      assert.ok(attrs.includes(`integrity="${hash}"`));
      return `'${hash}'`;
    });
    assert.deepEqual(policy.get("script-src"), hashes.length ? hashes : ["'none'"]);
    assert.deepEqual(policy.get("connect-src"), [name === "index.html" ? "'none'" : "'self'"]);
    assert.deepEqual(policy.get("frame-src"), [name === "index.html" ? "https://cyth.18-118-248-231.sslip.io" : name === "embed.html" ? "'self'" : "'none'"]);
    assert.ok(!/src="(?:auth|app|markdown|webui)\.js/.test(html));
  });
}
