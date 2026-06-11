const API_BASE = "https://18-118-248-231.sslip.io";

const STORAGE_KEY = "cyssie.sessions.v1";
const ACTIVE_KEY = "cyssie.activeSessionId.v1";
const TOKEN_KEY = "cyssie.accessToken.v1";

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

let sessions = loadSessions();
let activeSessionId = localStorage.getItem(ACTIVE_KEY);

if (!sessions.length) {
  createSession();
}

if (!sessions.some((session) => session.id === activeSessionId)) {
  activeSessionId = sessions[0].id;
  localStorage.setItem(ACTIVE_KEY, activeSessionId);
}

render();

if (!getToken()) {
  openTokenModal();
}

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
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function getActiveSession() {
  return sessions.find((session) => session.id === activeSessionId);
}

function createSession() {
  const session = {
    id: createId(),
    title: "New chat",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [
      {
        role: "system",
        content: "Cyssie is awake. Start a new conversation whenever you are ready."
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

function renderMessages() {
  const session = getActiveSession();

  chat.innerHTML = "";
  activeTitle.textContent = `${session.title}.md`;

  session.messages.forEach((message) => {
    const div = document.createElement("div");
    div.className = `msg ${message.role}`;
    div.textContent = message.content;
    chat.appendChild(div);
  });

  chat.scrollTop = chat.scrollHeight;
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
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
  return session.messages
    .filter((message) => message.role === "user" || message.role === "bot")
    .map((message) => ({
      role: message.role === "bot" ? "assistant" : "user",
      content: message.content
    }));
}

async function sendMessage(message) {
  addMessageToSession("user", message);
  addMessageToSession("bot", "Cyssie is thinking...");
  render();

  const session = getActiveSession();
  const loadingMessage = session.messages[session.messages.length - 1];

  sendButton.disabled = true;

  try {
    const token = getToken();

    if (!token) {
      openTokenModal();
      throw new Error("Missing access token");
    }

    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify({
        message,
        history: buildHistoryForApi(session)
      })
    });

    if (response.status === 401) {
      openTokenModal();
      throw new Error("Unauthorized. Please check your access token.");
    }

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    loadingMessage.content = data.response || "(No response)";
  } catch (error) {
    loadingMessage.content = `Cyssie could not respond: ${error.message}`;
  } finally {
    session.updatedAt = Date.now();
    saveSessions();
    render();
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  await sendMessage(message);
});

newChatButton.addEventListener("click", () => {
  createSession();
  render();
  input.focus();
});

clearChatButton.addEventListener("click", async () => {
  const session = getActiveSession();

  session.messages = [
    {
      role: "system",
      content: "Session cleared. Cyssie is ready again."
    }
  ];

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

deleteChatButton.addEventListener("click", () => {
  if (sessions.length === 1) {
    sessions = [];
    createSession();
  } else {
    sessions = sessions.filter((session) => session.id !== activeSessionId);
    activeSessionId = sessions[0].id;
    localStorage.setItem(ACTIVE_KEY, activeSessionId);
  }

  saveSessions();
  render();
});

tokenButton.addEventListener("click", openTokenModal);

saveTokenButton.addEventListener("click", () => {
  const token = tokenInput.value.trim();

  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
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