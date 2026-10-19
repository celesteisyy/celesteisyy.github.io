"use strict";

(() => {
  const origin = "https://cyth.18-118-248-231.sslip.io";
  const signIn = origin + "/playground/sign-in";
  const startConversation = "/playground/bootstrap";
  const status = document.getElementById("workspaceStatus");
  if (window.location.origin !== origin) return;

  async function sessionStatus() {
    const response = await fetch("/api/v1/auths/", {
      credentials: "same-origin", cache: "no-store", redirect: "error",
      signal: AbortSignal.timeout(10000),
    });
    // The gate checks only the HTTP status; it never handles credentials.
    return response.status;
  }

  if (document.body.dataset.mode === "start") {
    try {
      // Remove the old helper's shared marker. Popup and frame must not race
      // over Open WebUI's global post-login return path.
      localStorage.removeItem("redirectPath");
      // This short-lived, non-secret marker chooses only a fixed return page.
      // Cognito state and every credential are still managed by Open WebUI.
      document.cookie = "cyth_embed_login=1; Path=/oauth/oidc/callback; Max-Age=300; Secure; SameSite=Lax";
      window.location.replace("/oauth/oidc/login");
    } catch {
      status.textContent = "Could not start sign-in. Please open the full workspace.";
    }
    return;
  }

  if (document.body.dataset.mode === "complete") {
    document.cookie = "cyth_embed_login=; Path=/oauth/oidc/callback; Max-Age=0; Secure; SameSite=Lax";
    sessionStatus().then(code => {
      if (code !== 200) {
        status.textContent = "Sign-in could not be confirmed. Please return and try again.";
        return;
      }
      status.textContent = "Signed in. You can return to Playground.";
      try {
        if (window.opener) window.opener.postMessage({ type: "cyssie-signed-in" }, origin);
      } catch {
        // The opener can be isolated or already closed. The gate also polls.
      }
      window.close();
    }).catch(() => {
      status.textContent = "Could not check sign-in. Please return to Playground.";
    });
    return;
  }

  if (document.body.dataset.mode === "bootstrap") {
    // This document runs only inside the cyth conversation frame. WebUI's
    // session endpoint validates its HttpOnly cookie and returns its own JWT,
    // exactly as the native app expects. Never send it to an ancestor/opener.
    if (window.parent === window) {
      status.textContent = "Open Playground to continue.";
      return;
    }
    (async () => {
      try {
        const response = await fetch("/api/v1/auths/", {
          credentials: "same-origin", cache: "no-store", redirect: "error",
          signal: AbortSignal.timeout(10000),
        });
        if (!response.ok) throw new Error("Session unavailable");
        const session = await response.json();
        if (typeof session.token !== "string" || !session.token || session.token.length > 16384) {
          throw new Error("Session unavailable");
        }
        localStorage.setItem("token", session.token);
        window.location.replace("/");
      } catch {
        document.body.dataset.failed = "true";
        status.textContent = "Could not open this session. Return to Playground and try again.";
      }
    })();
    return;
  }

  const frame = document.getElementById("conversation");
  const notice = document.getElementById("workspaceNotice");
  const login = document.getElementById("signInButton");
  const retry = document.getElementById("retryButton");
  const full = document.getElementById("fullWorkspaceLink");
  let popup = null;
  let poll = null;
  let monitor = null;
  let checking = false;
  let signInCompleted = false;
  let booting = false;
  let bootStarted = 0;

  function stopPolling() { clearInterval(poll); poll = null; }
  function showNotice(text) {
    clearInterval(monitor);
    monitor = null;
    frame.hidden = true;
    frame.removeAttribute("src");
    notice.hidden = false;
    status.textContent = text;
    login.hidden = false;
    booting = false;
  }

  function monitorConversation() {
    try {
      const url = new URL(frame.contentWindow.location.href);
      if (url.protocol !== "https:") return;
      if (url.origin !== origin || url.pathname.startsWith("/oauth/")) {
        showNotice("Sign in to wake Cyssie.");
        return;
      }
      if (url.pathname === "/auth" || url.pathname === startConversation) {
        const failed = url.pathname === startConversation && frame.contentDocument?.body?.dataset.failed === "true";
        if (booting && !failed && Date.now() - bootStarted < 20000) return;
        showNotice(booting
          ? "Your browser could not open sign-in here. Choose Continue, or open the full workspace."
          : "Sign in to wake Cyssie.");
        retry.hidden = false;
        full.hidden = false;
        return;
      }
      booting = false;
      notice.hidden = true;
      frame.hidden = false;
    } catch {
      showNotice("Sign in to wake Cyssie.");
    }
  }

  function openConversation() {
    stopPolling();
    if (booting || !frame.hidden) return;
    login.hidden = true;
    retry.hidden = true;
    status.textContent = "Opening your conversation…";
    booting = true;
    bootStarted = Date.now();
    // Bootstrap only inside the cyth frame's own storage partition. The
    // GitHub parent receives neither the response nor its credentials.
    frame.src = startConversation;
    clearInterval(monitor);
    monitor = setInterval(monitorConversation, 500);
  }

  async function checkSession(afterSignIn = false) {
    if (checking || booting || !frame.hidden) return;
    checking = true;
    try {
      const code = await sessionStatus();
      if (code === 200) {
        openConversation();
      } else if (code === 401 || code === 403) {
        const completed = afterSignIn || signInCompleted || !!popup?.closed;
        status.textContent = completed
          ? "Finish signing in, then choose Continue. Your browser may ask to allow sign-in here."
          : "Sign in to wake Cyssie.";
        login.hidden = false;
        if (completed) { retry.hidden = false; full.hidden = false; }
      } else {
        status.textContent = "The workspace is unavailable. Please try again.";
        retry.hidden = false;
        retry.textContent = "Try again";
      }
    } catch {
      status.textContent = "Could not reach the workspace. Please try again.";
      retry.hidden = false;
      retry.textContent = "Try again";
      full.hidden = false;
    } finally {
      checking = false;
    }
  }

  login.addEventListener("click", () => {
    signInCompleted = false;
    popup = window.open(signIn, "_blank", "popup,width=560,height=760");
    if (!popup) {
      status.textContent = "Allow the sign-in window, then try again.";
      full.hidden = false;
      return;
    }
    status.textContent = "Finish signing in in the new window, then return here.";
    retry.hidden = false;
    retry.textContent = "Continue";
    stopPolling();
    const started = Date.now();
    poll = setInterval(() => {
      if (Date.now() - started > 120000) { stopPolling(); return; }
      checkSession(false);
    }, 2000);
  });

  retry.addEventListener("click", async () => {
    // This permission request only happens in response to a user's click.
    if (popup && document.requestStorageAccess && document.hasStorageAccess) {
      try {
        if (!(await document.hasStorageAccess())) await document.requestStorageAccess();
      } catch {
        status.textContent = "This browser does not allow embedded sign-in. Open the full workspace to continue.";
        full.hidden = false;
        return;
      }
    }
    await checkSession(!!popup);
  });

  window.addEventListener("message", event => {
    if (event.origin !== origin || !popup || event.source !== popup) return;
    if (!event.data || event.data.type !== "cyssie-signed-in") return;
    signInCompleted = true;
    checkSession(true);
  });
  frame.addEventListener("load", monitorConversation);
  window.addEventListener("pagehide", () => { stopPolling(); clearInterval(monitor); });
  checkSession();
})();
