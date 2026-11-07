"use strict";

(() => {
  const origin = "https://cyth.18-118-248-231.sslip.io";
  const frame = document.getElementById("webui");
  const navigation = document.getElementById("siteNavigation");
  const menu = navigation?.querySelector("details");
  if (!frame || !navigation) return;
  let handshake = null;

  function synchronize() {
    collapse(false);
    clearInterval(handshake);
    let attempts = 0;
    const ready = () => {
      if (++attempts > 8) { clearInterval(handshake); return; }
      frame.contentWindow?.postMessage({ type: "cyssie-navigation-ready" }, origin);
    };
    ready();
    handshake = setInterval(ready, 250);
  }

  function collapse(hidden) {
    // Keep keyboard navigation available while a link/menu has visible focus.
    if (hidden && navigation.querySelector(":focus-visible")) return;
    if (hidden && menu) menu.open = false;
    navigation.inert = hidden;
    document.body.classList.toggle("nav-collapsed", hidden);
  }

  window.addEventListener("message", event => {
    if (event.origin !== origin || event.source !== frame.contentWindow) return;
    const data = event.data;
    if (!data || data.type !== "cyssie-navigation" || typeof data.hidden !== "boolean") return;
    clearInterval(handshake);
    collapse(data.hidden);
  });
  window.addEventListener("keydown", event => {
    if (event.key === "Tab") collapse(false);
  }, true);
  frame.addEventListener("load", synchronize);
  window.addEventListener("pagehide", () => clearInterval(handshake));
  synchronize();
})();
