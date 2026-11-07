/* Login lives here; the API makes the authorization decision. */
(function () {
  "use strict";

  // Remove OAuth credentials from the address bar before any API requests.
  let callbackUrl = null;
  const current = new URL(window.location.href);
  if (current.searchParams.has("code") || current.searchParams.has("error")) {
    callbackUrl = current.href;
    for (const key of ["code", "state", "error", "error_description", "error_uri", "session_state"]) {
      current.searchParams.delete(key);
    }
    window.history.replaceState(null, "", current.pathname + current.search + current.hash);
  }

  class SignInError extends Error {}

  function create(apiBase, { onChange = () => {}, onError = () => {} } = {}) {
    const apiOrigin = new URL(apiBase).origin;
    let mode = "loading";
    let manager = null;
    let user = null;
    let legacyToken = "";
    let config = null;
    let renewing = null;
    let sessionVersion = 0;

    const changed = () => onChange();
    const hasSession = () => mode === "legacy" ? Boolean(legacyToken) : Boolean(user && (!user.expired || user.refresh_token));
    const forgetUser = async () => {
      sessionVersion++;
      user = null;
      legacyToken = "";
      if (manager) await manager.removeUser();
      changed();
    };

    async function getToken() {
      if (mode === "legacy" && legacyToken) return legacyToken;
      if (mode !== "cognito" || !user) throw new SignInError("Please sign in to Cyssie.");
      if (user.expires_in <= 60) {
        if (!user.refresh_token) {
          await forgetUser();
          throw new SignInError("Your session has expired. Please sign in again.");
        }
        if (!renewing) {
          const version = sessionVersion;
          renewing = manager.signinSilent().then(next => {
            if (version !== sessionVersion) throw new SignInError("Please sign in again.");
            if (!next || next.expired) throw new SignInError("Please sign in again.");
            user = next;
            return next.access_token;
          }).catch(async () => {
            await forgetUser();
            throw new SignInError("Your session has expired. Please sign in again.");
          }).finally(() => { renewing = null; });
        }
        return renewing;
      }
      return user.access_token;
    }

    async function authenticatedFetch(url, options = {}) {
      const target = new URL(url, apiBase);
      // Bearer tokens must never reach links produced by the model or redirects.
      if (target.origin !== apiOrigin || target.username || target.password) {
        throw new Error("Cyssie can only authenticate requests to its own API.");
      }
      const token = await getToken();
      const headers = new Headers(options.headers);
      headers.set("Authorization", `Bearer ${token}`);
      const response = await window.fetch(target.href, {
        ...options, headers, redirect: "error", credentials: "omit"
      });
      if (response.status === 401 || response.status === 403) {
        await forgetUser();
        throw new SignInError(response.status === 403
          ? "This account does not have access to Cyssie."
          : "Your session has expired. Please sign in again.");
      }
      return response;
    }

    async function initialize() {
      mode = "loading";
      changed();
      const response = await window.fetch(`${apiBase}/auth/config`, {
        cache: "no-store", credentials: "omit", redirect: "error",
        signal: AbortSignal.timeout(10000)
      });
      // Compatibility only with the existing backend, which has no auth/config.
      if (response.status === 404) {
        mode = "legacy";
        changed();
        return;
      }
      if (!response.ok) throw new Error("Cyssie could not load its sign-in settings. Please retry.");
      config = await response.json();
      if (config.mode === "legacy") {
        mode = "legacy";
        changed();
        return;
      }
      if (config.mode !== "cognito" || config.redirect_uri !== `${window.location.origin}/playground/`) {
        throw new Error("Cyssie's sign-in settings need to be checked.");
      }
      mode = "cognito";
      manager = new window.oidc.UserManager({
        authority: config.authority,
        client_id: config.client_id,
        redirect_uri: config.redirect_uri,
        response_type: "code",
        scope: config.scope,
        disablePKCE: false,
        loadUserInfo: false,
        automaticSilentRenew: false,
        monitorSession: false,
        requestTimeoutInSeconds: 10,
        userStore: new window.oidc.WebStorageStateStore({ store: window.sessionStorage, prefix: "cyssie.user." }),
        stateStore: new window.oidc.WebStorageStateStore({ store: window.sessionStorage, prefix: "cyssie.state." })
      });
      if (callbackUrl) {
        const callback = callbackUrl;
        callbackUrl = null;
        try {
          user = await manager.signinRedirectCallback(callback);
        } catch {
          await forgetUser();
          changed();
          onError("Please start sign-in from Cyssie's Sign in button.");
          return;
        }
      } else {
        user = await manager.getUser();
      }
      await manager.clearStaleState();
      if (user) {
        const verified = await authenticatedFetch(`${apiBase}/auth/me`, { signal: AbortSignal.timeout(10000) });
        if (!verified.ok) {
          await forgetUser();
          throw new Error("Cyssie could not verify your sign-in. Please retry.");
        }
      }
      changed();
    }

    async function signIn() {
      if (!manager) return initialize();
      // The library checks saved request state, PKCE, and this nonce.
      // The API independently verifies the access token signature and claims.
      const bytes = new Uint8Array(32);
      window.crypto.getRandomValues(bytes);
      const nonce = Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
      await manager.signinRedirect({ nonce, prompt: "login" });
    }

    async function signOut() {
      if (!manager) return forgetUser();
      // Clear the UI immediately and prevent an in-flight renewal restoring it.
      sessionVersion++;
      user = null;
      changed();
      let endpoint;
      try {
        endpoint = await manager.metadataService.getAuthorizationEndpoint();
        try { await manager.revokeTokens(["refresh_token"]); } catch { /* Clear local session even offline. */ }
      } finally {
        await forgetUser();
      }
      const logout = new URL("/logout", endpoint);
      logout.searchParams.set("client_id", config.client_id);
      logout.searchParams.set("logout_uri", config.redirect_uri);
      window.location.assign(logout.href);
    }

    return {
      initialize, signIn, signOut, getToken, hasSession,
      fetch: authenticatedFetch,
      get mode() { return mode; },
      setLegacyToken(token) {
        if (mode !== "legacy") throw new Error("Password sign-in is required.");
        legacyToken = token.trim();
        changed();
      },
      isSignInError(error) { return error instanceof SignInError; }
    };
  }

  window.CyssieAuth = { create };
})();
