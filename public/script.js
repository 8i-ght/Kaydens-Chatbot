// Creat a chat inferface with button logic
document.getElementById('send-button').addEventListener('click', async () => {
    const userInput = document.getElementById('user-input').value.trim();
    const outputContainer = document.getElementById('output-container');

    if (userInput) {
        // Display user question
        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = `You: ${userInput}`;
        outputContainer.appendChild(userMessage);

        // Send the question to backend
        try {
            const response = await fetch('http://127.0.0.1:5000/ask', {
                method: 'POST', 
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: userInput }),
            });

            const data = await response.json();

            // Display the response
            const chatbotMessage = document.createElement('div');
            chatbotMessage.className = 'chatbot-message';
            chatbotMessage.textContent = `Chatbot: ${data.answer}`;
            outputContainer.appendChild(chatbotMessage);
        } catch (error) {
            console.error('Error:', error);
        }
        // Clear the input field
        document.getElementById('user-input').value = '';
    }
});

// Add event listener for the enter key
document.getElementById('user-input').addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        document.getElementById('send-button').click();
    }
});