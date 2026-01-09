// LLM Chat UI Frontend

// State
let currentChatId = null;
let currentBackend = 'vllm';
let models = { vllm: [], llamacpp: [] };
let isStreaming = false;

// Settings (stored in localStorage)
const defaultSettings = {
    temperature: 0.7,
    maxTokens: 2048,
    topP: 0.9
};

function getSettings() {
    const stored = localStorage.getItem('llmChatSettings');
    return stored ? JSON.parse(stored) : { ...defaultSettings };
}

function saveSettings(settings) {
    localStorage.setItem('llmChatSettings', JSON.stringify(settings));
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    loadSettings();
    await loadModels();
    await loadChats();
    await checkBackendStatus();
});

// Settings
function loadSettings() {
    const settings = getSettings();
    document.getElementById('temperature').value = settings.temperature;
    document.getElementById('temperatureValue').textContent = settings.temperature;
    document.getElementById('maxTokens').value = settings.maxTokens;
    document.getElementById('maxTokensValue').textContent = settings.maxTokens;
    document.getElementById('topP').value = settings.topP;
    document.getElementById('topPValue').textContent = settings.topP;
}

function updateSettingValue(setting) {
    const input = document.getElementById(setting);
    const valueSpan = document.getElementById(setting + 'Value');
    valueSpan.textContent = input.value;

    const settings = getSettings();
    settings[setting] = parseFloat(input.value);
    saveSettings(settings);
}

function resetSettings() {
    saveSettings({ ...defaultSettings });
    loadSettings();
}

function toggleSettings() {
    document.getElementById('settingsPanel').classList.toggle('open');
}

// Backend status
async function checkBackendStatus() {
    try {
        const response = await fetch('/api/backends/status');
        const status = await response.json();
        updateStatusIndicator(status.status);

        if (status.status === 'running') {
            currentBackend = status.backend;
            document.getElementById('backendSelect').value = currentBackend;
            // Update model select to show current model
            const modelSelect = document.getElementById('modelSelect');
            for (let option of modelSelect.options) {
                if (option.value === status.model) {
                    option.selected = true;
                    break;
                }
            }
        }
    } catch (error) {
        console.error('Error checking backend status:', error);
    }
}

function updateStatusIndicator(status) {
    const indicator = document.getElementById('statusIndicator');
    indicator.className = 'status-indicator';
    if (status === 'running') {
        indicator.classList.add('running');
        indicator.title = 'Backend running';
    } else if (status === 'starting') {
        indicator.classList.add('starting');
        indicator.title = 'Backend starting...';
    } else {
        indicator.title = 'Backend stopped';
    }
}

// Models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        models = await response.json();
        updateModelSelect();
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function updateModelSelect() {
    const modelSelect = document.getElementById('modelSelect');
    const backendModels = models[currentBackend] || [];

    modelSelect.innerHTML = '<option value="">Select a model...</option>';

    for (const model of backendModels) {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = model.name;
        modelSelect.appendChild(option);
    }
}

async function onBackendChange() {
    currentBackend = document.getElementById('backendSelect').value;
    updateModelSelect();
}

async function onModelChange() {
    const modelPath = document.getElementById('modelSelect').value;
    if (!modelPath) return;

    // Check if this model is already running
    const statusResponse = await fetch('/api/backends/status');
    const status = await statusResponse.json();

    if (status.status === 'running' &&
        status.backend === currentBackend &&
        status.model === modelPath) {
        // Already running this model
        return;
    }

    // Start the backend
    await startBackend(currentBackend, modelPath);
}

async function startBackend(backend, modelPath) {
    const modal = document.getElementById('loadingModal');
    const logsDiv = document.getElementById('loadingLogs');
    const titleEl = document.getElementById('loadingTitle');

    modal.classList.add('open');
    titleEl.textContent = `Starting ${backend}...`;
    logsDiv.innerHTML = '';
    updateStatusIndicator('starting');

    try {
        const response = await fetch('/api/backends/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend, model_path: modelPath })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.message) {
                            const p = document.createElement('p');
                            p.textContent = data.message;
                            logsDiv.appendChild(p);
                            logsDiv.scrollTop = logsDiv.scrollHeight;
                        }

                        if (data.done) {
                            if (data.success) {
                                updateStatusIndicator('running');
                                hideWelcomeMessage();
                            } else {
                                updateStatusIndicator('stopped');
                            }
                            setTimeout(() => modal.classList.remove('open'), 1000);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE:', e);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error starting backend:', error);
        const p = document.createElement('p');
        p.textContent = `Error: ${error.message}`;
        p.style.color = 'var(--error)';
        logsDiv.appendChild(p);
        updateStatusIndicator('stopped');
        setTimeout(() => modal.classList.remove('open'), 2000);
    }
}

// Chat list
async function loadChats() {
    try {
        const response = await fetch('/api/chats');
        const data = await response.json();
        renderChatList(data.chats);
    } catch (error) {
        console.error('Error loading chats:', error);
    }
}

function renderChatList(chats) {
    const chatList = document.getElementById('chatList');
    chatList.innerHTML = '';

    for (const chat of chats) {
        const div = document.createElement('div');
        div.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
        div.onclick = () => loadChat(chat.id);
        div.innerHTML = `
            <span class="chat-item-title">${escapeHtml(chat.title)}</span>
            <button class="chat-item-delete" onclick="event.stopPropagation(); deleteChat('${chat.id}')">&times;</button>
        `;
        chatList.appendChild(div);
    }
}

async function createNewChat() {
    try {
        const response = await fetch('/api/chats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                backend: currentBackend,
                model: document.getElementById('modelSelect').value
            })
        });
        const chat = await response.json();
        currentChatId = chat.id;
        await loadChats();
        clearMessages();
        hideWelcomeMessage();
    } catch (error) {
        console.error('Error creating chat:', error);
    }
}

async function loadChat(chatId) {
    try {
        const response = await fetch(`/api/chats/${chatId}`);
        const chat = await response.json();

        currentChatId = chatId;
        await loadChats(); // Refresh list to update active state

        renderMessages(chat.messages);
        hideWelcomeMessage();
    } catch (error) {
        console.error('Error loading chat:', error);
    }
}

async function deleteChat(chatId) {
    if (!confirm('Delete this chat?')) return;

    try {
        await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
        if (currentChatId === chatId) {
            currentChatId = null;
            clearMessages();
            showWelcomeMessage();
        }
        await loadChats();
    } catch (error) {
        console.error('Error deleting chat:', error);
    }
}

// Messages
function clearMessages() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = `
        <div class="welcome-message" id="welcomeMessage" style="display: none;">
            <h2>Welcome to LLM Chat</h2>
            <p>Select a model to start chatting</p>
        </div>
    `;
}

function showWelcomeMessage() {
    document.getElementById('welcomeMessage').style.display = 'flex';
}

function hideWelcomeMessage() {
    const welcome = document.getElementById('welcomeMessage');
    if (welcome) welcome.style.display = 'none';
}

function renderMessages(messages) {
    clearMessages();
    hideWelcomeMessage();

    const messagesDiv = document.getElementById('messages');

    for (const msg of messages) {
        appendMessage(msg.role, msg.content, msg.reasoning, false);
    }

    scrollToBottom();
}

function appendMessage(role, content, reasoning = '', streaming = false) {
    const messagesDiv = document.getElementById('messages');

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let html = `<div class="message-role">${role}</div>`;

    if (reasoning) {
        html += `
            <div class="thinking-block">
                <div class="thinking-header" onclick="toggleThinking(this)">
                    <span class="thinking-icon">&#128173;</span>
                    <span>Thinking</span>
                    <span class="thinking-toggle">&#9660;</span>
                </div>
                <div class="thinking-content${streaming ? '' : ' collapsed'}">${escapeHtml(reasoning)}</div>
            </div>
        `;
    }

    html += `<div class="message-content">${escapeHtml(content)}${streaming ? '<span class="streaming-cursor"></span>' : ''}</div>`;

    messageDiv.innerHTML = html;
    messagesDiv.appendChild(messageDiv);

    return messageDiv;
}

function toggleThinking(header) {
    const content = header.nextElementSibling;
    const toggle = header.querySelector('.thinking-toggle');
    content.classList.toggle('collapsed');
    toggle.textContent = content.classList.contains('collapsed') ? '\u25BC' : '\u25B2';
}

function scrollToBottom() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Send message
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const content = input.value.trim();

    if (!content || isStreaming) return;

    // Check backend status
    const statusResponse = await fetch('/api/backends/status');
    const status = await statusResponse.json();

    if (status.status !== 'running') {
        alert('Please select a model first');
        return;
    }

    // Create chat if needed
    if (!currentChatId) {
        await createNewChat();
    }

    // Add user message
    appendMessage('user', content);
    input.value = '';
    autoResize(input);

    // Save user message
    await fetch(`/api/chats/${currentChatId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: 'user', content })
    });

    // Get all messages for context
    const chatResponse = await fetch(`/api/chats/${currentChatId}`);
    const chat = await chatResponse.json();

    // Stream response
    isStreaming = true;
    document.getElementById('sendBtn').disabled = true;

    const settings = getSettings();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: chat.messages,
                temperature: settings.temperature,
                max_tokens: settings.maxTokens,
                top_p: settings.topP
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let assistantContent = '';
        let assistantReasoning = '';
        let messageDiv = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'content') {
                            assistantContent += data.text;
                            updateStreamingMessage(messageDiv, assistantContent, assistantReasoning);
                        } else if (data.type === 'reasoning') {
                            assistantReasoning += data.text;
                            updateStreamingMessage(messageDiv, assistantContent, assistantReasoning);
                        } else if (data.type === 'error') {
                            assistantContent += `\n\nError: ${data.message}`;
                            updateStreamingMessage(messageDiv, assistantContent, assistantReasoning);
                        } else if (data.type === 'done') {
                            // Finalize message
                            if (messageDiv) {
                                const cursor = messageDiv.querySelector('.streaming-cursor');
                                if (cursor) cursor.remove();

                                // Collapse thinking block
                                const thinkingContent = messageDiv.querySelector('.thinking-content');
                                if (thinkingContent) {
                                    thinkingContent.classList.add('collapsed');
                                    const toggle = messageDiv.querySelector('.thinking-toggle');
                                    if (toggle) toggle.textContent = '\u25BC';
                                }
                            }

                            // Save assistant message
                            await fetch(`/api/chats/${currentChatId}/messages`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    role: 'assistant',
                                    content: data.full_content,
                                    reasoning: data.full_reasoning
                                })
                            });

                            await loadChats(); // Refresh to update title
                        }

                        // Create message div on first content
                        if (!messageDiv && (assistantContent || assistantReasoning)) {
                            messageDiv = appendMessage('assistant', '', '', true);
                        }

                        scrollToBottom();

                    } catch (e) {
                        console.error('Error parsing SSE:', e);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error sending message:', error);
        appendMessage('assistant', `Error: ${error.message}`);
    } finally {
        isStreaming = false;
        document.getElementById('sendBtn').disabled = false;
    }
}

function updateStreamingMessage(messageDiv, content, reasoning) {
    if (!messageDiv) return;

    const contentDiv = messageDiv.querySelector('.message-content');
    if (contentDiv) {
        contentDiv.innerHTML = escapeHtml(content) + '<span class="streaming-cursor"></span>';
    }

    // Update or create reasoning block
    if (reasoning) {
        let thinkingBlock = messageDiv.querySelector('.thinking-block');
        if (!thinkingBlock) {
            const roleDiv = messageDiv.querySelector('.message-role');
            thinkingBlock = document.createElement('div');
            thinkingBlock.className = 'thinking-block';
            thinkingBlock.innerHTML = `
                <div class="thinking-header" onclick="toggleThinking(this)">
                    <span class="thinking-icon">&#128173;</span>
                    <span>Thinking</span>
                    <span class="thinking-toggle">&#9650;</span>
                </div>
                <div class="thinking-content"></div>
            `;
            roleDiv.after(thinkingBlock);
        }

        const thinkingContent = thinkingBlock.querySelector('.thinking-content');
        if (thinkingContent) {
            thinkingContent.textContent = reasoning;
        }
    }
}

// Input handling
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

// Utility
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
