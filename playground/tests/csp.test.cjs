const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

test("workspace CSP pins its script and confines frames and requests to the session origin", () => {
  const match = html.match(/<meta http-equiv="Content-Security-Policy" content="([^"]+)"/);
  assert.ok(match && match.index < html.indexOf("<link "));
  const policy = new Map(match[1].split(";").map(part => {
    const [name, ...values] = part.trim().split(/\s+/);
    return [name, values];
  }));
  for (const name of ["default-src", "script-src-attr", "style-src-attr", "object-src", "base-uri", "form-action"]) {
    assert.deepEqual(policy.get(name), ["'none'"]);
  }
  for (const name of ["frame-src", "connect-src", "style-src"]) {
    assert.deepEqual(policy.get(name), ["'self'"]);
  }
  const scripts = [...html.matchAll(/<script\s+([^>]*)><\/script>/g)];
  assert.equal(scripts.length, 1);
  const attrs = scripts[0][1];
  const src = attrs.match(/src="([^"]+)"/)[1];
  const hash = "sha256-" + createHash("sha256").update(fs.readFileSync(path.join(root, src.split("?")[0]))).digest("base64");
  assert.ok(attrs.includes(`integrity="${hash}"`));
  assert.deepEqual(policy.get("script-src"), [`'${hash}'`]);
  assert.ok(!/src="(?:auth|app|markdown)\.js/.test(html));
});
