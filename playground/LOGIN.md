# Cyssie account login

The API's public `/auth/config` selects the mode. Cognito mode uses the
vendored `oidc-client-ts` library and managed login with authorization code,
PKCE S256, state, and nonce. The backend independently verifies the access
token and permits only its configured owner. The browser sends the access
token, never the ID token, in API requests.

Set the SPA's callback **and sign-out** URL to
`https://celesteisyy.github.io/playground/`, with `openid email` scopes.
The library discovers the managed login domain from the pool's OIDC metadata.
Backend configuration and deployment order are documented in the paired
`cyssie-api` change's `deploy/COGNITO.md`.

Publish this frontend before switching the backend to Cognito. A 404 from an
old backend's `/auth/config` enables the existing token dialog; network/server
errors do not silently fall back. The new API can explicitly select legacy
mode during rollout. Fixed legacy tokens remain only in memory.

Cognito sessions are stored in the current tab's `sessionStorage`, with refresh
before token expiry. They are not saved in `localStorage`. Conversations retain
their existing browser-local storage behavior. Sign out clears the local
session, attempts refresh-token revocation, and ends Cognito managed login.
The backend's offline token verifier can accept an already-issued access token
until expiry, so configure a short access-token lifetime.

Callback code/state are removed from the URL, and the page sets
`Referrer-Policy: no-referrer` via a meta tag. Authentication requests reject
external API origins and redirects. Generated files use authenticated fetch
and a temporary blob download, without tokens in links. PPTX uses
`/downloads/presentations/`, Draw.io uses `/downloads/diagrams/`; old static
links are translated to these endpoints. Deploy the paired backend update
and restart the API before deploying this download update. Ordinary file,
network and server errors keep the session and show a message in chat;
authentication failure opens sign-in.

## Vendored dependencies

The signed-out dialog blocks the chat workspace while the sticky public banner
remains available for Home, Blogs and LinkedIn navigation.

Math uses locally vendored KaTeX 0.19.0 for dollar and backslash delimiters.
Markdown is sanitized before parsed math is rendered with KaTeX's DOM API,
with trust disabled, per-formula macros and bounded expansion, size and input.
Invalid formulas fall back to text; code and escaped dollar signs stay literal.
The existing CSP still rejects model-supplied scripts and inline styles.
API-relative generated artifact links are resolved against the API for authenticated downloads.

The exact npm distributions, hashes, versions, and licenses are in
`vendor/manifest.json` and the adjacent license files. Serving OIDC, Markdown,
and sanitization scripts locally avoids runtime script CDN dependencies. To
update them, obtain the official npm package, retain its license and update
the manifest hash, script reference, and tests together. No build step is
required to serve this playground.

## Verification

The page includes a hash-based CSP before all resources, and every script has
matching Subresource Integrity. Inline scripts, event handlers, eval, frames,
objects and forms cannot execute/load; connections are limited to the current
API, user pool metadata path and exact managed-login domain. Markdown also
rejects styles, active elements and named-property clobbering. After changing
scripts or the API/pool/domain, run `node playground/scripts/update-csp.cjs`
and the checks below. CI also exercises the policy in Chromium. GitHub Pages
does not let this file set HTTP headers; a meta CSP cannot enforce
`frame-ancestors`, and does not isolate other pages on the same origin.
These measures reduce XSS risk; tokens remain JavaScript-accessible.

```bash
node --test playground/tests/auth.test.cjs playground/tests/csp.test.cjs
```

Tests execute the real OIDC library against simulated TLS endpoints. They cover
PKCE generation/exchange, callback cleanup, state/nonce rejection, backend
authorization rejection, refresh, expiry, logout, origin restrictions, and
legacy rollout behavior. Actual AWS settings and EC2/Nginx behavior still need
deployment verification.

