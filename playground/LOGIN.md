# Embedded Playground

The public page stays at `https://celesteisyy.github.io/playground/` with the
existing navigation. Its iframe opens a small sign-in helper on
`https://cyth.18-118-248-231.sslip.io/playground/embed/`. The authenticated
conversation inside that helper is Open WebUI, without a fork of its UI.
The GitHub page has no JavaScript and makes no direct authenticated API calls.

## Authentication and origins

- Sign in opens `/playground/sign-in` in a new browser window, then starts the
  existing `/oauth/oidc/login` flow. Open WebUI owns state, PKCE, Cognito claims,
  sessions and token validation. No shared service token is used.
- The sign-in document sets a five-minute, non-secret, host-only marker scoped
  to `/oauth/oidc/callback`. On that exact callback, Nginx rewrites only the
  successful fixed `/auth` redirect to `/playground/login-complete`. The marker
  is cleared there. Without the marker, ordinary WebUI login keeps `/auth`.
  Error/MFA/pending redirects and all OAuth state, credentials and Set-Cookie
  headers pass through unchanged. A forged marker grants no session or rights.
- The completion document checks its native WebUI session, then closes itself.
  It does not depend on the opener surviving Cognito navigation or on WebUI's
  client routing. Completion messages contain only `{type: "cyssie-signed-in"}`
  with exact origin and window checks. A bounded poll detects the authenticated
  embedded session even if the browser isolates the popup from its opener.
- Open WebUI v0.12.2 makes its auth cookie HttpOnly when `/api/v1/auths/` is
  queried. Its native `/auth` page cannot read that cookie in a fresh storage
  partition. The conversation frame therefore first loads the fixed cyth
  `/playground/bootstrap` document. That document calls WebUI's own authenticated
  session endpoint and writes its returned WebUI token to WebUI's own localStorage
  in the frame's partition, then navigates to `/`. The cookie remains HttpOnly.
  The outer gate reads only response statuses. Credentials and chat content
  never pass through postMessage, the GitHub parent, or URL parameters.
- Only auth cookies change to `SameSite=None; Secure` for cross-site embedding.
  The OAuth state/session cookie remains `SameSite=Lax; Secure`. Existing owner
  group restrictions remain enforced; password and OAuth registration stay off.
- Because cross-site cookies need explicit CSRF protection, the Nginx origin
  guard rejects every non-GET/HEAD/OPTIONS request and every Upgrade request
  unless `Origin` is exactly the cyth origin. Missing, null and foreign origins
  are rejected, including requests with an Authorization header. The application
  still authenticates every protected operation. Native external API clients
  making writes must provide this Origin as well as valid authentication.
- Every frame ancestor must be cyth itself or `https://celesteisyy.github.io`.
  CORS remains limited to the cyth origin: the GitHub parent does not need API
  access. No wildcard embedding, CORS, or origin rules are introduced.
- A browser that blocks embedded cookie/storage access may require the user to
  click Continue and grant storage access. Managed browser policies can refuse
  it. The UI then offers the full workspace; it never treats an inaccessible
  session as authenticated or disables authentication to make embedding work.

## Deploy before merging the GitHub entry

1. On EC2, update this branch and run:

   ```bash
   cd ~/cyssie-site
   git pull --ff-only
   sudo bash playground/deploy/install.sh
   sudo nginx -t && sudo systemctl reload nginx
   ```

   Keep exactly one `include /etc/nginx/snippets/cyth-playground.conf;` in the
   existing cyth HTTPS server. The installer puts the maps under Nginx's
   standard http-level `/etc/nginx/conf.d/` include. Retain TLS, proxy locations
   and existing authentication. The existing WebUI proxy must inherit the
   snippet's `proxy_redirect` (do not override it with `proxy_redirect off`).
   Remove a deployment-only IP allowlist only
   after the owner-only authentication settings have been verified.
2. Verify the guard **before enabling cross-site cookies**. These anonymous
   writes must return 403:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' -X POST \
     -H 'Origin: https://evil.example' \
     https://cyth.18-118-248-231.sslip.io/api/v1/auths/signout
   curl -s -o /dev/null -w '%{http_code}\n' -X POST \
     https://cyth.18-118-248-231.sslip.io/api/v1/auths/signout
   ```
3. Add the embed overlay **after both existing Compose files**:

   ```bash
   cd ~/cyssie-api/deploy/open-webui
   sudo docker compose --env-file .env \
     -f compose.yaml -f compose.small.yaml \
     -f ~/cyssie-site/playground/deploy/compose.embed.yaml \
     up -d --force-recreate
   ```

   Retain this third file on future recreations. It preserves the slim image,
   memory limits, existing project/volume, Cognito secrets and model settings.
   Existing cookies may still have Lax attributes: the helper's popup initiates
   a fresh Cognito flow so the callback issues the updated auth cookies.
4. Open `https://cyth.18-118-248-231.sslip.io/playground/` to check the deployed
   documents, then verify the exact GitHub iframe context before cutover. The
   real cross-site context matters; a same-origin preview alone is insufficient.
   Once EC2 deployment is ready, merge the GitHub entry and verify in the actual
   browser: sign in through the popup, chat, invoke a native tool, sign out,
   reload and use mobile navigation. Repeat with embedded cookies blocked to
   confirm the permission/fallback message. No Cognito callback changes needed.
5. Without cookies, `/playground/` and `/playground/embed/` return 200 while
   `/api/v1/auths/` and `/api/models` return 401 or 403. Inspect response headers
   to confirm the exact frame-ancestors list and `SameSite=None; Secure` auth
   cookies after signing in. Never paste those cookie values into an issue.

Only the named public static documents/assets are served by Nginx. The checkout,
secrets, uploads and arbitrary `/var/www` files are not published. The backend
API service and native tool service keep their loopback listeners and Cognito
authentication. Old browser-local chats are not imported or deleted.

## Updating and rollback

Run `node playground/scripts/update-csp.cjs` after script changes to refresh all
CSP/SRI hashes. The CI runs browser sign-in/embedding tests and actual Nginx
routing, unsafe-method and WebSocket origin checks.

For rollback, revert the frontend entry first. Recreate WebUI with the original
**two** Compose files to restore Lax auth cookies, then sign out/in to replace
existing None cookies. Only after those cookies are no longer active should the
origin guard be removed. Alternatively keep the guard as an additional control.
No data or volumes need deletion. Restore a saved Nginx configuration if needed,
run `nginx -t`, then reload. The direct Open WebUI root remains available.
