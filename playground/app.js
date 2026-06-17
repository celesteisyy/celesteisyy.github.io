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

const imageInput = document.createElement("input");
imageInput.type = "file";
imageInput.accept = "image/jpeg,image/jpg,image/png,image/webp";
imageInput.style.display = "none";

const imageButton = document.createElement("button");
imageButton.type = "button";
imageButton.className = "image-button";
imageButton.textContent = "Image";
imageButton.title = "Upload an image";

let selectedImageFile = null;

if (form && sendButton) {
  form.insertBefore(imageInput, sendButton);
  form.insertBefore(imageButton, sendButton);
}

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

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(content) {
  const text = String(content || "");

  if (window.marked && window.DOMPurify) {
    marked.setOptions({
      breaks: true,
      gfm: true
    });

    return DOMPurify.sanitize(marked.parse(text));
  }

  return escapeHtml(text).replaceAll("\n", "<br>");
}

function renderMessages() {
  const session = getActiveSession();

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

async function sendMessage(message, imageFile = null) {
  const userDisplay = imageFile
    ? `${message || ""}${message ? "\n" : ""}${getImageLabel(imageFile)}`
    : message;

  addMessageToSession("user", userDisplay);
  addMessageToSession("bot", "Cyssie is thinking...");
  render();

  const session = getActiveSession();
  const loadingMessage = session.messages[session.messages.length - 1];

  sendButton.disabled = true;
  imageButton.disabled = true;

  try {
    const token = getToken();

    if (!token) {
      openTokenModal();
      throw new Error("Missing access token");
    }

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
      response = await fetch(`${API_BASE}/chat`, {
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
    }

    if (response.status === 401) {
      openTokenModal();
      throw new Error("Unauthorized. Please check your access token.");
    }

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const data = await response.json();
    loadingMessage.content = data.response || "(No response)";
  } catch (error) {
    console.error(error);
    loadingMessage.content = "Cyssie's off to grab some coffee — come find her a little later!";
  } finally {
    session.updatedAt = Date.now();
    saveSessions();
    render();
    sendButton.disabled = false;
    imageButton.disabled = false;
    input.focus();
  }
}

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

  input.value = "";
  clearSelectedImage();

  await sendMessage(message, imageFile);
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