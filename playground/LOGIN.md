# Playground workspace

The public `/playground/` entry opens the workspace at
`https://cyth.18-118-248-231.sslip.io/playground/`.
The workspace keeps the site navigation and embeds Open WebUI on the **same
origin**. Home, Blogs and LinkedIn retain their existing destinations.

Open WebUI owns authentication, conversations, rendering, uploads and models.
The wrapper does not read, copy or forward session tokens or old chat storage.
Older browser-local chats are not imported or deleted. Cookies remain Secure
and SameSite=Lax. Cognito authentication runs in the top-level window through
`/auth?redirect=%2Fplayground%2F`, using Open WebUI's existing return-path flow.
No Cognito client changes are required.

## Deploy before merging the entry-point change

1. Check out this PR on EC2 (or the merged commit on later updates).
2. Run `sudo bash playground/deploy/install.sh`. This installs three static
   files and an Nginx snippet. It does not edit, reload or restart services.
3. Back up `/etc/nginx/sites-available/cyth`. In its **existing HTTPS server
   block**, add exactly one line:

   ```nginx
   include /etc/nginx/snippets/cyth-playground.conf;
   ```

   Retain existing IP restrictions, TLS configuration and proxy locations.
   Do not add a second server or place the include in the HTTP redirect block.
4. Check the upstream response headers:

   ```bash
   curl -sSI http://127.0.0.1:3000/ | grep -Ei '^(HTTP/|x-frame-options:|content-security-policy:)'
   ```

   Open WebUI must permit framing by its own origin: if configured, its
   `XFRAME_OPTIONS` must be `SAMEORIGIN` and its CSP `frame-ancestors` must
   include `'self'`. Do not remove other CSP directives or use a wildcard.
   Change these only if the live headers require it; preserve both Compose
   files and the existing data volume when recreating the container.
5. Run `sudo nginx -t && sudo systemctl reload nginx`.
6. Open `https://cyth.18-118-248-231.sslip.io/playground/`. Check signed-in chat,
   sign-out/sign-in, navigation, one native tool call and mobile layout.
   Authentication and WebUI API requests must stay on the same origin.
7. Merge only after this deployed route works. GitHub Pages then updates the
   old entry point without sending visitors to a route that is not ready.

The snippet serves only the three exact static asset paths. It does not expose
the checkout, uploads, secrets, or other files under `/var/www`. The original
API service and Open WebUI root continue to serve their existing routes.
The existing owner-only access restriction still applies.

## Update and rollback

Run the installer again from the selected commit and reload Nginx after its
configuration test. Run `node playground/scripts/update-csp.cjs` whenever
changing the wrapper script so SRI and CSP hashes stay in sync.

For rollback, restore the previous frontend commit. Remove the added Nginx
include (or restore its backup), run `nginx -t`, then reload. The direct
Open WebUI root continues to work; no data or volumes need deletion.

Legacy script files are no longer loaded by the entry point. They remain in
the repository for a separate cleanup; the wrapper never executes them.
