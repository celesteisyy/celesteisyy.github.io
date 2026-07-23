const API_BASE = "https://18-118-248-231.sslip.io";
const STORAGE_KEY = "cyssie.sessions.v1";
const ACTIVE_KEY = "cyssie.activeSessionId.v1";
const RECENT_HISTORY_TURNS = 16;
const COMPACTION_BATCH_TURNS = 4;
const COMPACTION_BATCH_CHARS = 60000;
const STREAM_RENDER_INTERVAL_MS = 80;
const STREAM_SAVE_INTERVAL_MS = 750;
const SIDEBAR_COLLAPSED_KEY = "cyssie.sidebarCollapsed.v1";
const MOBILE_SIDEBAR_BREAKPOINT = 860;

let accessToken = "";
let selectedImageFile = null;
let sessions = [];
let activeSessionId = null;

function createId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadSessions() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveSessions() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch (error) {
    console.warn(
      "Cyssie could not persist the latest local conversation state.",
      error
    );
  }
}

function getToken() {
  return accessToken;
}

document.addEventListener("DOMContentLoaded", () => {
  const chat = document.getElementById("chat");
  const sessionsEl = document.getElementById("sessions");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("messageInput");
  const sendButton = document.getElementById("sendButton");
  const newChatButton = document.getElementById("newChatButton");
  const clearChatButton = document.getElementById("clearChatButton");
  const deleteChatButton = document.getElementById("deleteChatButton");
  const tokenButton = document.getElementById("tokenButton");
  const tokenModal = document.getElementById("tokenModal");
  const tokenInput = document.getElementById("tokenInput");
  const saveTokenButton = document.getElementById("saveTokenButton");
  const activeTitle = document.getElementById("activeTitle");
  const workspace = document.querySelector(".workspace");
  const sidebarToggle = document.getElementById("sidebarToggle");

  const requiredElements = {
    chat,
    sessionsEl,
    form,
    input,
    sendButton,
    newChatButton,
    clearChatButton,
    deleteChatButton,
    tokenButton,
    tokenModal,
    tokenInput,
    saveTokenButton,
    activeTitle,
    workspace,
    sidebarToggle
  };

  const missing = Object.entries(requiredElements)
    .filter(([, value]) => !value)
    .map(([name]) => name);

  if (missing.length) {
    console.error("Cyssie UI could not initialize. Missing DOM elements:", missing);
    return;
  }

  const imageInput = document.createElement("input");
  imageInput.type = "file";
  imageInput.accept = "image/jpeg,image/jpg,image/png,image/webp";
  imageInput.style.display = "none";

  const imageButton = document.createElement("button");
  imageButton.type = "button";
  imageButton.className = "image-button";
  imageButton.textContent = "Image";
  imageButton.title = "Upload an image";

  form.insertBefore(imageInput, sendButton);
  form.insertBefore(imageButton, sendButton);

  sessions = loadSessions();
  activeSessionId = localStorage.getItem(ACTIVE_KEY);

  if (!sessions.length) {
    createSession();
  }

  if (!sessions.some((session) => session.id === activeSessionId)) {
    activeSessionId = sessions[0].id;
    localStorage.setItem(ACTIVE_KEY, activeSessionId);
  }

  const savedSidebarState = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
  const initialSidebarCollapsed = savedSidebarState === null
    ? window.matchMedia(`(max-width: ${MOBILE_SIDEBAR_BREAKPOINT}px)`).matches
    : savedSidebarState === "true";

  setSidebarCollapsed(initialSidebarCollapsed, false);
  autoResizeInput();
  render();

  if (!getToken()) {
    openTokenModal();
  }

  function setSidebarCollapsed(collapsed, persist = true) {
    workspace.classList.toggle("sidebar-collapsed", collapsed);
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
    sidebarToggle.textContent = collapsed ? "Show sessions" : "Hide sessions";

    if (persist) {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(collapsed));
    }
  }

  function autoResizeInput() {
    input.style.height = "auto";
    const maxHeight = Number.parseFloat(getComputedStyle(input).maxHeight) || 160;
    const nextHeight = Math.min(input.scrollHeight, maxHeight);
    input.style.height = `${nextHeight}px`;
    input.style.overflowY = input.scrollHeight > maxHeight ? "auto" : "hidden";
  }

  function getActiveSession() {
    return sessions.find((session) => session.id === activeSessionId);
  }

  function createSession(message = "Cyssie is awake. Start a new conversation whenever you are ready.") {
    const session = {
      id: createId(),
      title: "New chat",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      contextSummary: "",
      summarizedMessageCount: 0,
      messages: [
        {
          role: "system",
          content: message
        }
      ]
    };

    sessions.unshift(session);
    activeSessionId = session.id;
    localStorage.setItem(ACTIVE_KEY, activeSessionId);
    saveSessions();

    return session;
  }

  function updateSessionTitle(session, firstUserMessage) {
    if (session.title !== "New chat") return;

    const compact = firstUserMessage.replace(/\s+/g, " ").trim();
    session.title = compact.length > 38 ? `${compact.slice(0, 38)}...` : compact || "New chat";
  }

  function addMessageToSession(role, content) {
    const session = getActiveSession();
    if (!session) return;

    session.messages.push({ role, content });
    session.updatedAt = Date.now();

    if (role === "user") {
      updateSessionTitle(session, content);
    }

    sessions = [
      session,
      ...sessions.filter((item) => item.id !== session.id)
    ];

    saveSessions();
  }

  function render() {
    renderSessions();
    renderMessages();
  }

  function renderSessions() {
    sessionsEl.innerHTML = "";

    sessions.forEach((session) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `session-item ${session.id === activeSessionId ? "active" : ""}`;

      const title = document.createElement("span");
      title.className = "session-title";
      title.textContent = session.title;

      const meta = document.createElement("span");
      meta.className = "session-meta";
      meta.textContent = formatTime(session.updatedAt);

      button.appendChild(title);
      button.appendChild(meta);

      button.addEventListener("click", () => {
        activeSessionId = session.id;
        localStorage.setItem(ACTIVE_KEY, activeSessionId);
        render();
      });

      sessionsEl.appendChild(button);
    });
  }

  function formatTime(timestamp) {
    return new Date(timestamp).toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  }

  function escapeHtml(text) {
    return String(text || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function isSafeRenderedUrl(value) {
    try {
      const url = new URL(value, window.location.href);
      return ["http:", "https:", "mailto:"].includes(url.protocol);
    } catch {
      return false;
    }
  }

  function renderMarkdown(content) {
    const text = String(content || "");

    if (window.marked && window.DOMPurify) {
      marked.setOptions({
        breaks: true,
        gfm: true
      });

      const sanitized = DOMPurify.sanitize(marked.parse(text), {
        FORBID_TAGS: [
          "img", "picture", "video", "audio", "source", "track",
          "iframe", "object", "embed", "form", "input", "button",
          "svg", "math", "image", "use"
        ],
        FORBID_ATTR: [
          "style", "ping", "srcset", "poster", "formaction", "xlink:href"
        ]
      });
      const template = document.createElement("template");
      template.innerHTML = sanitized;

      template.content.querySelectorAll("a").forEach((link) => {
        const href = link.getAttribute("href") || "";
        if (!isSafeRenderedUrl(href)) {
          link.removeAttribute("href");
          link.removeAttribute("target");
          link.removeAttribute("rel");
          return;
        }
        if (new URL(href, window.location.href).protocol !== "mailto:") {
          link.target = "_blank";
          link.rel = "noopener noreferrer";
        }
      });

      return template.innerHTML;
    }

    return escapeHtml(text).replaceAll("\n", "<br>");
  }

  function renderMessages() {
    const session = getActiveSession();
    if (!session) return;

    chat.innerHTML = "";
    activeTitle.textContent = `${session.title}.md`;

    session.messages.forEach((message) => {
      const div = document.createElement("div");
      div.className = `msg ${message.role}`;

      if (message.role === "bot") {
        div.classList.add("markdown-body");
        div.innerHTML = renderMarkdown(message.content);
      } else {
        div.textContent = message.content;
      }

      chat.appendChild(div);
    });

    chat.scrollTop = chat.scrollHeight;
  }

  function openTokenModal() {
    tokenInput.value = getToken();
    tokenModal.classList.add("visible");
    setTimeout(() => tokenInput.focus(), 50);
  }

  function closeTokenModal() {
    tokenModal.classList.remove("visible");
  }

  function buildHistoryForApi(session) {
    if (!session) return [];

    return session.messages
      .filter((message) => message.role === "user" || message.role === "bot")
      .filter((message) => message.content && message.content !== "Cyssie is thinking...")
      .map((message) => ({
        role: message.role === "bot" ? "assistant" : "user",
        content: message.content
      }));
  }

  function getRecentHistoryStart(history, recentTurns = RECENT_HISTORY_TURNS) {
    const userIndexes = history
      .map((message, index) => message.role === "user" ? index : -1)
      .filter((index) => index >= 0);

    if (userIndexes.length <= recentTurns) return 0;
    return userIndexes[userIndexes.length - recentTurns];
  }

  function countUserTurns(history) {
    return history.filter((message) => message.role === "user").length;
  }

  function historyCharacterCount(history) {
    return history.reduce(
      (total, message) => total + String(message.content || "").length,
      0
    );
  }

  async function prepareConversationContext(session, history, token) {
    const recentStart = getRecentHistoryStart(history);
    const savedCount = Number.isInteger(session.summarizedMessageCount)
      ? session.summarizedMessageCount
      : 0;
    const summaryCursor = Math.max(0, Math.min(savedCount, recentStart));
    const messagesToCompact = history.slice(summaryCursor, recentStart);
    const shouldCompact = (
      countUserTurns(messagesToCompact) >= COMPACTION_BATCH_TURNS
      || historyCharacterCount(messagesToCompact) >= COMPACTION_BATCH_CHARS
    );

    session.contextSummary = String(session.contextSummary || "");
    session.summarizedMessageCount = summaryCursor;

    if (shouldCompact) {
      try {
        const compactResponse = await fetch(`${API_BASE}/context/compact`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({
            history: messagesToCompact,
            conversation_summary: session.contextSummary
          })
        });

        if (!compactResponse.ok) {
          throw new Error(`Context compaction failed with HTTP ${compactResponse.status}`);
        }

        const compacted = await compactResponse.json();
        session.contextSummary = String(compacted.conversation_summary || "");
        session.summarizedMessageCount = recentStart;
        saveSessions();
      } catch (error) {
        console.warn("Cyssie could not compact older history; using recent turns.", error);

        return {
          conversationSummary: session.contextSummary,
          history: history.slice(recentStart)
        };
      }
    }

    return {
      conversationSummary: session.contextSummary,
      history: history.slice(session.summarizedMessageCount)
    };
  }

  function getImageLabel(imageFile) {
    if (!imageFile) return "";

    return `[Image uploaded: ${imageFile.name}]`;
  }

  function clearSelectedImage() {
    selectedImageFile = null;
    imageInput.value = "";
    imageButton.textContent = "Image";
    imageButton.classList.remove("has-image");
  }

  async function readStreamingResponse(response, loadingMessage, session) {
    if (!response.body) {
      loadingMessage.content = await response.text();
      saveSessions();
      render();
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";
    let lastRenderAt = 0;
    let lastSaveAt = 0;

    function updateStreamingView(force = false) {
      const now = performance.now();

      if (force || now - lastSaveAt >= STREAM_SAVE_INTERVAL_MS) {
        saveSessions();
        lastSaveAt = now;
      }

      if (!force && now - lastRenderAt < STREAM_RENDER_INTERVAL_MS) {
        return;
      }

      lastRenderAt = now;

      if (activeSessionId !== session.id) {
        return;
      }

      const messageIndex = session.messages.indexOf(loadingMessage);
      const messageElement = messageIndex >= 0 ? chat.children[messageIndex] : null;

      if (!messageElement) {
        renderMessages();
        return;
      }

      if (force) {
        messageElement.style.whiteSpace = "";
        messageElement.innerHTML = renderMarkdown(loadingMessage.content);
      } else {
        messageElement.style.whiteSpace = "pre-wrap";
        messageElement.textContent = loadingMessage.content;
      }
      chat.scrollTop = chat.scrollHeight;
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      fullText += decoder.decode(value, { stream: true });
      loadingMessage.content = fullText || "Cyssie is thinking...";
      session.updatedAt = Date.now();
      updateStreamingView();
    }

    const tail = decoder.decode();
    if (tail) {
      fullText += tail;
    }

    loadingMessage.content = fullText || "(No response)";
    session.updatedAt = Date.now();
    updateStreamingView(true);
  }

  async function sendMessage(message, imageFile = null) {
    if (!getToken()) {
      openTokenModal();
      return;
    }

    const historyBeforeSend = buildHistoryForApi(getActiveSession());
    const userDisplay = imageFile ? `${message || ""}${message ? "\n" : ""}${getImageLabel(imageFile)}` : message;
    addMessageToSession("user", userDisplay);
    addMessageToSession("bot", "Cyssie is thinking...");
    render();

    const session = getActiveSession();
    const loadingMessage = session.messages[session.messages.length - 1];

    sendButton.disabled = true;
    imageButton.disabled = true;

    try {
      const token = getToken();
      let response;

      if (imageFile) {
        const formData = new FormData();
        formData.append("message", message || "What is in this image?");
        formData.append("image", imageFile);

        response = await fetch(`${API_BASE}/vision`, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${token}`
          },
          body: formData
        });
      } else {
        const preparedContext = await prepareConversationContext(
          session,
          historyBeforeSend,
          token
        );

        response = await fetch(`${API_BASE}/chat_stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({
            message,
            history: preparedContext.history,
            conversation_summary: preparedContext.conversationSummary
          })
        });
      }

      if (response.status === 401) {
        openTokenModal();
        throw new Error("Unauthorized. Please check your access token.");
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      if (imageFile) {
        const data = await response.json();
        loadingMessage.content = data.response || "(No response)";
      } else {
        await readStreamingResponse(response, loadingMessage, session);
      }
    } catch (error) {
      console.error(error);
      const partialText = (
        loadingMessage.content
        && loadingMessage.content !== "Cyssie is thinking..."
      ) ? loadingMessage.content : "";

      loadingMessage.content = partialText
        ? `${partialText}\n\n[Cyssie's connection was interrupted before this answer finished.]`
        : "Cyssie's off to grab some coffee — come find her a little later!";
    } finally {
      session.updatedAt = Date.now();
      saveSessions();
      render();
      sendButton.disabled = false;
      imageButton.disabled = false;
      input.focus();
    }
  }

  sidebarToggle.addEventListener("click", () => {
    setSidebarCollapsed(!workspace.classList.contains("sidebar-collapsed"));
  });

  input.addEventListener("input", autoResizeInput);

  imageButton.addEventListener("click", () => {
    imageInput.click();
  });

  imageInput.addEventListener("change", () => {
    selectedImageFile = imageInput.files && imageInput.files[0] ? imageInput.files[0] : null;

    if (selectedImageFile) {
      imageButton.textContent = `Image: ${selectedImageFile.name}`;
      imageButton.classList.add("has-image");
    } else {
      clearSelectedImage();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = input.value.trim();
    const imageFile = selectedImageFile;

    if (!message && !imageFile) return;

    if (!getToken()) {
      openTokenModal();
      return;
    }

    input.value = "";
    autoResizeInput();
    clearSelectedImage();

    await sendMessage(message, imageFile);
  });

  newChatButton.addEventListener("click", (event) => {
    event.preventDefault();
    createSession();
    render();
    input.focus();
  });

  clearChatButton.addEventListener("click", async (event) => {
    event.preventDefault();

    const session = getActiveSession();

    session.messages = [
      {
        role: "system",
        content: "Session cleared. Cyssie is ready again."
      }
    ];
    session.contextSummary = "";
    session.summarizedMessageCount = 0;

    session.updatedAt = Date.now();
    saveSessions();
    render();

    const token = getToken();

    if (token) {
      try {
        await fetch(`${API_BASE}/reset`, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${token}`
          }
        });
      } catch {
        // Reset failure should not block local session clearing.
      }
    }
  });

  deleteChatButton.addEventListener("click", (event) => {
    event.preventDefault();

    if (sessions.length <= 1) {
      sessions = [];
      createSession("Conversation deleted. Cyssie is ready for a fresh start.");
    } else {
      sessions = sessions.filter((session) => session.id !== activeSessionId);
      activeSessionId = sessions[0].id;
      localStorage.setItem(ACTIVE_KEY, activeSessionId);
    }

    saveSessions();
    render();
    input.focus();
  });

  tokenButton.addEventListener("click", (event) => {
    event.preventDefault();
    openTokenModal();
  });

  saveTokenButton.addEventListener("click", (event) => {
    event.preventDefault();

    const token = tokenInput.value.trim();

    if (token) {
      accessToken = token;
      tokenInput.value = "";
    }

    closeTokenModal();
    input.focus();
  });

  tokenModal.addEventListener("click", (event) => {
    if (event.target === tokenModal && getToken()) {
      closeTokenModal();
    }
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
});
