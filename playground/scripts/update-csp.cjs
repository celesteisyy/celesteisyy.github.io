// Run after editing a playground script.
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const root = path.join(__dirname, "..");
const index = path.join(root, "index.html");
let html = fs.readFileSync(index, "utf8");
const hashes = [];
html = html.replace(/<script\s+[^>]*src="([^"]+)"[^>]*><\/script>/g, (tag, src) => {
  const file = path.join(root, src.split("?")[0]);
  const hash = "sha256-" + createHash("sha256").update(fs.readFileSync(file)).digest("base64");
  hashes.push(`'${hash}'`);
  const clean = tag.replace(/\s+(?:integrity|crossorigin)="[^"]*"/g, "");
  return clean.replace(">", ` integrity="${hash}" crossorigin="anonymous">`);
});
const policy = [
  "default-src 'none'",
  `script-src ${hashes.join(" ")}`,
  "script-src-attr 'none'",
  "style-src 'self'",
  "style-src-attr 'none'",
  "connect-src 'self'",
  "frame-src 'self'",
  "img-src 'self'",
  "font-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'"
].join("; ");
const meta = `  <meta http-equiv="Content-Security-Policy" content="${policy}" />`;
if (/<meta http-equiv="Content-Security-Policy"[^>]*\/>/.test(html)) {
  html = html.replace(/  <meta http-equiv="Content-Security-Policy"[^>]*\/>/, meta);
} else {
  html = html.replace('  <meta charset="UTF-8" />', '  <meta charset="UTF-8" />\n' + meta);
}
fs.writeFileSync(index, html);
