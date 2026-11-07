const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

test("CSP precedes resources and permits only the exact integrity-pinned scripts", () => {
  const match = html.match(/<meta http-equiv="Content-Security-Policy" content="([^"]+)"/);
  assert.ok(match, "Missing CSP");
  assert.ok(match.index < html.indexOf('<link '));
  const policy = new Map(match[1].split(";").map(p => {
    const [name, ...values] = p.trim().split(/\s+/);
    return [name, values];
  }));
  for (const name of ["default-src", "script-src-attr", "style-src-attr", "object-src", "base-uri", "form-action"]) {
    assert.deepEqual(policy.get(name), ["'none'"]);
  }
  const scripts = [...html.matchAll(/<script\s+([^>]*)><\/script>/g)];
  assert.equal(scripts.length, 5);
  const hashes = scripts.map(([, attrs]) => {
    const src = attrs.match(/src="([^"]+)"/)[1];
    assert.ok(!src.startsWith("http"));
    const expected = "sha256-" + createHash("sha256")
      .update(fs.readFileSync(path.join(root, src.split("?")[0]))).digest("base64");
    assert.equal(attrs.match(/integrity="([^"]+)"/)[1], expected);
    assert.match(attrs, /crossorigin="anonymous"/);
    return `'${expected}'`;
  });
  assert.deepEqual(policy.get("script-src"), hashes);
  assert.ok(!match[1].includes("unsafe-"));
  assert.deepEqual(policy.get("connect-src"), [
    "https://18-118-248-231.sslip.io",
    "https://cognito-idp.us-east-2.amazonaws.com/us-east-2_0G8Lq71Sw/",
    "https://us-east-20g8lq71sw.auth.us-east-2.amazoncognito.com"
  ]);
});
