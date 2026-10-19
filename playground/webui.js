"use strict";

(() => {
  const origin = "https://cyth.18-118-248-231.sslip.io";
  const workspace = origin + "/playground/";
  const signIn = "/auth?redirect=%2Fplayground%2F";

  // Never forward URL parameters, credentials, or browser storage between sites.
  if (window.location.origin !== origin) {
    window.location.replace(workspace);
    return;
  }
  if (window.top !== window.self) {
    window.top.location.replace(workspace);
    return;
  }

  const frame = document.getElementById("webui");
  const notice = document.getElementById("workspaceNotice");
  const status = document.getElementById("workspaceStatus");
  const login = document.getElementById("signInLink");
  const retry = document.getElementById("retryButton");

  frame.addEventListener("load", () => {
    try {
      const url = new URL(frame.contentWindow.location.href);
      if (url.origin !== origin) return;
      // OAuth remains a top-level flow, including after a session expires.
      if (url.pathname === "/auth" || url.pathname.startsWith("/oauth/")) {
        window.location.assign(signIn);
        return;
      }
      if (url.protocol === "https:") {
        notice.hidden = true;
        frame.hidden = false;
      }
    } catch {
      status.textContent = "Open the full workspace to continue.";
    }
  });

  async function openWorkspace() {
    retry.disabled = true;
    login.hidden = true;
    status.textContent = "Opening your workspace…";
    try {
      const response = await fetch("/api/v1/auths/", {
        credentials: "same-origin",
        cache: "no-store",
        redirect: "error",
        signal: AbortSignal.timeout(10000),
      });
      if (response.status === 401 || response.status === 403) {
        status.textContent = "Sign in to open your workspace.";
        login.hidden = false;
      } else if (response.ok) {
        frame.src = "/";
        return;
      } else {
        status.textContent = "The workspace is unavailable. Please try again.";
      }
    } catch {
      status.textContent = "Could not reach the workspace. Please try again.";
    }
    retry.disabled = false;
  }

  retry.addEventListener("click", openWorkspace);
  openWorkspace();
})();
