import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, ShieldAlert, KeyRound } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hola, soy Providencia. Para empezar a consultar los documentos, por favor ingresa tu API Key de Gemini arriba.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [isKeySaved, setIsKeySaved] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Cargar la API key guardada si existe
    const savedKey = localStorage.getItem('providencia_gemini_key');
    if (savedKey) {
      setApiKey(savedKey);
      setIsKeySaved(true);
    }
    scrollToBottom();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const saveKey = () => {
    if (apiKey.trim()) {
      localStorage.setItem('providencia_gemini_key', apiKey.trim());
      setIsKeySaved(true);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    if (!isKeySaved || !apiKey.trim()) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Por favor, guarda tu API Key de Gemini antes de enviar un mensaje.', isError: true }]);
      return;
    }

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
    setInput('');
    setIsLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: userMessage,
          gemini_api_key: apiKey.trim() // Enviar la llave al backend
        }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Error en el servidor');
      }

      setMessages(prev => [...prev, { role: 'assistant', text: data.reply }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', text: `Error: ${error.message}`, isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col w-full h-screen bg-providencia-darker font-sans">
      
      {/* Header con BYOK Input */}
      <header className="flex flex-col md:flex-row items-center justify-between px-6 py-4 bg-providencia-dark border-b border-providencia-gray shadow-md gap-4">
        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="w-10 h-10 rounded-lg bg-providencia-blue flex items-center justify-center shadow-[0_0_15px_rgba(0,73,135,0.5)]">
            <Bot size={24} className="text-white" />
          </div>
          <h1 className="text-xl font-semibold tracking-wide text-white">PROVIDENCIA</h1>
        </div>

        {/* Input de API Key */}
        <div className="flex items-center gap-2 w-full md:w-auto">
          <div className="relative w-full md:w-64">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
              <KeyRound size={16} className="text-gray-400" />
            </div>
            <input
              type="password"
              placeholder="Tu Gemini API Key..."
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setIsKeySaved(false);
              }}
              className="w-full bg-providencia-darker border border-providencia-gray text-white text-sm rounded-lg focus:ring-providencia-blue focus:border-providencia-blue block pl-10 p-2 outline-none"
            />
          </div>
          <button 
            onClick={saveKey}
            className={`px-4 py-2 text-sm font-medium text-white rounded-lg transition-colors ${isKeySaved ? 'bg-providencia-green hover:bg-green-600' : 'bg-providencia-blue hover:bg-blue-700'}`}
          >
            {isKeySaved ? 'Guardada' : 'Guardar'}
          </button>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 md:p-8 flex flex-col gap-6 w-full max-w-4xl mx-auto">
        {messages.map((msg, index) => (
          <div 
            key={index} 
            className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex gap-4 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
              
              <div className="flex-shrink-0 mt-1">
                {msg.role === 'user' ? (
                  <div className="w-8 h-8 rounded-full bg-providencia-gray flex items-center justify-center">
                    <User size={18} className="text-gray-300" />
                  </div>
                ) : (
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${msg.isError ? 'bg-red-900/50 text-red-500' : 'bg-providencia-blue/20 text-providencia-blue border border-providencia-blue/30'}`}>
                    {msg.isError ? <ShieldAlert size={18} /> : <Bot size={18} />}
                  </div>
                )}
              </div>

              <div 
                className={`p-4 rounded-2xl whitespace-pre-wrap leading-relaxed ${
                  msg.role === 'user' 
                    ? 'bg-providencia-blue text-white rounded-tr-sm shadow-md' 
                    : msg.isError 
                      ? 'bg-red-950/30 text-red-200 border border-red-900/50 rounded-tl-sm'
                      : 'bg-providencia-gray text-providencia-light rounded-tl-sm shadow-sm'
                }`}
              >
                {msg.text}
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex w-full justify-start">
            <div className="flex gap-4 max-w-[85%] flex-row">
              <div className="flex-shrink-0 mt-1">
                <div className="w-8 h-8 rounded-lg bg-providencia-blue/20 text-providencia-blue border border-providencia-blue/30 flex items-center justify-center">
                  <Bot size={18} />
                </div>
              </div>
              <div className="p-4 rounded-2xl bg-providencia-gray text-providencia-light rounded-tl-sm flex items-center gap-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* Input Area */}
      <footer className="p-4 md:p-6 bg-providencia-dark border-t border-providencia-gray">
        <div className="max-w-4xl mx-auto">
          <form 
            onSubmit={handleSend}
            className={`relative flex items-center w-full bg-providencia-darker border rounded-xl overflow-hidden transition-colors shadow-inner ${isKeySaved ? 'border-providencia-gray focus-within:border-providencia-blue' : 'border-red-900/50 opacity-75'}`}
          >
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isKeySaved ? "Pregunta algo sobre los documentos..." : "Guarda tu API Key de Gemini arriba para preguntar..."}
              className="flex-1 bg-transparent p-4 text-white outline-none placeholder-gray-500 w-full"
              disabled={isLoading || !isKeySaved}
            />
            <button 
              type="submit" 
              disabled={!input.trim() || isLoading || !isKeySaved}
              className="p-3 m-2 bg-providencia-blue hover:bg-blue-700 disabled:bg-providencia-gray disabled:text-gray-500 text-white rounded-lg transition-colors flex items-center justify-center cursor-pointer shadow-md"
            >
              <Send size={20} className={isLoading ? 'opacity-50' : ''} />
            </button>
          </form>
        </div>
      </footer>
    </div>
  );
}

export default App;
