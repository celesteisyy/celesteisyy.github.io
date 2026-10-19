"use strict";

(() => {
  const origin = "https://cyth.18-118-248-231.sslip.io";
  const signIn = origin + "/playground/sign-in";
  const startConversation = "/auth?redirect=%2F&form=signin";
  const status = document.getElementById("workspaceStatus");
  if (window.location.origin !== origin) return;

  async function sessionStatus() {
    const response = await fetch("/api/v1/auths/", {
      credentials: "same-origin", cache: "no-store", redirect: "error",
      signal: AbortSignal.timeout(10000),
    });
    // Never read, copy, or message the response's credentials.
    return response.status;
  }

  if (document.body.dataset.mode === "start") {
    try {
      // This is Open WebUI's existing, non-secret post-login return-path key.
      localStorage.setItem("redirectPath", "/playground/login-complete");
      window.location.replace("/oauth/oidc/login");
    } catch {
      status.textContent = "Could not start sign-in. Please open the full workspace.";
    }
    return;
  }

  if (document.body.dataset.mode === "complete") {
    sessionStatus().then(code => {
      if (code !== 200) {
        status.textContent = "Sign-in could not be confirmed. Please return and try again.";
        return;
      }
      status.textContent = "Signed in. You can return to Playground.";
      if (window.opener) {
        window.opener.postMessage({ type: "cyssie-signed-in" }, origin);
        window.close();
      }
    }).catch(() => {
      status.textContent = "Could not check sign-in. Please return to Playground.";
    });
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
        showNotice("Sign in to continue your conversation.");
        return;
      }
      if (url.pathname === "/auth") {
        if (booting && Date.now() - bootStarted < 12000) return;
        showNotice(booting
          ? "Your browser could not open sign-in here. Choose Continue, or open the full workspace."
          : "Sign in to continue your conversation.");
        retry.hidden = false;
        full.hidden = false;
        return;
      }
      booting = false;
      notice.hidden = true;
      frame.hidden = false;
    } catch {
      showNotice("Sign in to continue your conversation.");
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
    // Open WebUI's own auth page copies its cookie into its own browser storage.
    // No custom token exchange and no credentials are exposed to the parent.
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
        const completed = afterSignIn || signInCompleted;
        status.textContent = completed
          ? "Finish signing in, then choose Continue. Your browser may ask to allow sign-in here."
          : "Sign in to open your workspace.";
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
