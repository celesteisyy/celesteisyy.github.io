// Refresh the exact script hashes for every static workspace document.
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");
const root = path.join(__dirname, "..");
for (const name of ["index.html", "embed.html", "sign-in.html", "login-complete.html", "bootstrap.html"]) {
  const file = path.join(root, name);
  let html = fs.readFileSync(file, "utf8");
  const hashes = [];
  html = html.replace(/<script\s+[^>]*src="([^"]+)"[^>]*><\/script>/g, (tag, src) => {
    const relative = src.replace(/^\/playground\//, "").split("?")[0];
    const hash = "sha256-" + createHash("sha256").update(fs.readFileSync(path.join(root, relative))).digest("base64");
    hashes.push(`'${hash}'`);
    const clean = tag.replace(/\s+(?:integrity|crossorigin)="[^"]*"/g, "");
    return clean.replace(">", ` integrity="${hash}" crossorigin="anonymous">`);
  });
  const parent = name === "index.html";
  const policy = [
    "default-src 'none'",
    `script-src ${hashes.length ? hashes.join(" ") : "'none'"}`,
    "script-src-attr 'none'", "style-src 'self'", "style-src-attr 'none'",
    `connect-src ${parent ? "'none'" : "'self'"}`,
    `frame-src ${parent ? "https://cyth.18-118-248-231.sslip.io" : name === "embed.html" ? "'self'" : "'none'"}`,
    "img-src 'self'", "font-src 'self'", "object-src 'none'", "base-uri 'none'", "form-action 'none'"
  ].join("; ");
  html = html.replace(/<meta http-equiv="Content-Security-Policy"[^>]*\/>/, `<meta http-equiv="Content-Security-Policy" content="${policy}" />`);
  fs.writeFileSync(file, html);
}
