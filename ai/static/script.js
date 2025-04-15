async function sendMessage() {
    const input = document.getElementById("input-box").value;
    if (!input.trim()) return;
    
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<p><b>Sen:</b> ${input}</p>`;
    
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: input, use_speech: false })
        });
        const data = await response.json();
        chatBox.innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
        saveChatHistory(input, data.response);
    } catch (error) {
        chatBox.innerHTML += `<p><b>Hata:</b> Bir sorun oluştu!</p>`;
    }
    
    chatBox.scrollTop = chatBox.scrollHeight;
    document.getElementById("input-box").value = "";
}

function toggleTheme() {
    const themeLink = document.getElementById("theme");
    const currentTheme = themeLink.getAttribute("href");
    themeLink.setAttribute("href", currentTheme.includes("dark") ? "/static/themes/light.css" : "/static/themes/dark.css");
}

function toggleSpeech() {
    fetch("/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    }).then(response => response.json()).then(data => {
        const chatBox = document.getElementById("chat-box");
        chatBox.innerHTML += `<p><b>Sen (Ses):</b> ${data.response}</p>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}

function saveChatHistory(input, response) {
    let history = JSON.parse(localStorage.getItem("chat_history") || "[]");
    history.push({ input, response, timestamp: new Date().toISOString() });
    localStorage.setItem("chat_history", JSON.stringify(history));
}

document.getElementById("input-box").addEventListener("keypress", function(e) {
    if (e.key === "Enter") sendMessage();
});