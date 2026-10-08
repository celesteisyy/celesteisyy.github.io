// Sanitize model HTML first, then render parsed math through KaTeX's DOM API.
// Never interpret tool JSON or model output as executable code.
(function () {
  "use strict";
  const MAX_FORMULA_CHARS = 4000;
  const MAX_FORMULAS = 128;

  function safeUrl(value) {
    try {
      return ["http:", "https:", "mailto:"].includes(new URL(value, location.href).protocol);
    } catch { return false; }
  }

  function render(container, content, { apiBase = "" } = {}) {
    const text = String(content || "");
    if (!window.marked || !window.DOMPurify) {
      container.textContent = text;
      return;
    }
    const formulas = [];
    const nonce = Array.from(crypto.getRandomValues(new Uint32Array(4)), n => n.toString(16)).join("-");
    function placeholder(token) {
      if (formulas.length >= MAX_FORMULAS || token.formula.length > MAX_FORMULA_CHARS) {
        return String(token.raw).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
      }
      const index = formulas.push(token) - 1;
      const tag = token.display ? "div" : "span";
      return '<' + tag + ' data-cyssie-math="' + nonce + '-' + index + '"></' + tag + '>';
    }
    const parser = new marked.Marked({
      breaks: true,
      gfm: true,
      extensions: [
        {
          name: "blockMath",
          level: "block",
          start(src) { return src.search(/^[ \t]{0,3}(?:\$\$|\\\[)/m); },
          tokenizer(src) {
            const match = /^(?: {0,3})?(?:\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\])(?:[ \t]*(?:\n|$))/.exec(src);
            if (match) return { type: "blockMath", raw: match[0], formula: match[1] ?? match[2], display: true };
          },
          renderer: placeholder
        },
        {
          name: "inlineMath",
          level: "inline",
          start(src) { return src.search(/\\\(|\$/); },
          tokenizer(src) {
            if (this.lexer.state.inRawBlock) return;
            const match = /^(?:\\\(([\s\S]*?)\\\)|\$\$([\s\S]*?)\$\$|\$(?![\s$])((?:\\.|[^\\$\n])*?[^\s\\])\$(?!\d))/.exec(src);
            if (match) return { type: "inlineMath", raw: match[0], formula: match[1] ?? match[2] ?? match[3], display: match[2] !== undefined };
          },
          renderer: placeholder
        }
      ]
    });
    container.innerHTML = DOMPurify.sanitize(parser.parse(text), {
      USE_PROFILES: { html: true },
      SANITIZE_NAMED_PROPS: true,
      FORBID_TAGS: [
        "style", "link", "meta", "base",
        "img", "picture", "video", "audio", "source", "track",
        "iframe", "object", "embed", "form", "input", "button",
        "svg", "math", "image", "use"
      ],
      FORBID_ATTR: ["style", "ping", "srcset", "poster", "formaction", "xlink:href"]
    });
    container.querySelectorAll("a").forEach(link => {
      const href = link.getAttribute("href") || "";
      if (!safeUrl(href)) {
        link.removeAttribute("href");
        link.removeAttribute("target");
        link.removeAttribute("rel");
      } else if (new URL(href, location.href).protocol !== "mailto:") {
        // Artifact runtimes may return an API-relative download path.
        if (apiBase && /^\/(?:downloads\/|static\/generated\/)/.test(href)) {
          link.href = new URL(href, apiBase).href;
        }
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
    });
    container.querySelectorAll("[data-cyssie-math]").forEach(marker => {
      const id = marker.getAttribute("data-cyssie-math");
      marker.removeAttribute("data-cyssie-math");
      if (!id.startsWith(nonce + "-")) return;
      const token = formulas[Number(id.slice(nonce.length + 1))];
      if (!token) return;
      marker.classList.add(token.display ? "math-display" : "math-inline");
      marker.setAttribute("role", "math");
      marker.setAttribute("aria-label", token.formula.trim());
      if (!window.katex) {
        marker.textContent = token.raw;
        return;
      }
      try {
        // render() sets trusted layout styles through DOM properties.
        // CSP still forbids model-supplied inline styles.
        katex.render(token.formula.trim(), marker, {
          displayMode: token.display,
          output: "html",
          trust: false,
          throwOnError: true,
          strict: "error",
          maxSize: 10,
          maxExpand: 1000,
          macros: {}
        });
      } catch {
        marker.textContent = token.raw;
        marker.classList.add("math-fallback");
      }
    });
  }
  window.CyssieMarkdown = Object.freeze({ render });
})();
