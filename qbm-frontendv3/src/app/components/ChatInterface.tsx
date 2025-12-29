"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Bot, User } from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { MetricsPanel, isMetricIntentQuery } from "./MetricsPanel";

interface Message {
  role: "user" | "assistant";
  content: string;
  // If true, render MetricsPanel instead of text content
  isMetricsResponse?: boolean;
}

interface ChatInterfaceProps {
  apiUrl: string;
  agentName?: string;
  placeholder?: string;
  initialQuery?: string | null;
}

export function ChatInterface({ apiUrl, agentName = "Assistant", placeholder = "Type your message...", initialQuery }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [processedQuery, setProcessedQuery] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Generate unique IDs
  const generateId = () => Math.random().toString(36).substring(2, 15);
  const [threadId] = useState(() => generateId());

  // Send message function - uses C1Chat compatible format
  const sendMessage = async (userMessage: string, currentMessages: Message[]) => {
    if (isLoading) return;
    
    setMessages([...currentMessages, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      // CRITICAL: Detect metric-intent queries and render MetricsPanel directly
      // Do NOT send to LLM - numbers must come from deterministic endpoint only
      if (isMetricIntentQuery(userMessage)) {
        // Render MetricsPanel component instead of LLM response
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "", // Content is ignored when isMetricsResponse is true
            isMetricsResponse: true,
          },
        ]);
        setIsLoading(false);
        return;
      }

      // Non-metric queries go to LLM
      const responseId = generateId();
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: { role: "user", content: userMessage },
          threadId,
          responseId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = "";

      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        assistantMessage += chunk;

        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            role: "assistant",
            content: assistantMessage,
          };
          return newMessages;
        });
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, an error occurred. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle initial query - only process if it's different from the last processed query
  useEffect(() => {
    if (initialQuery && initialQuery !== processedQuery && !isLoading) {
      setProcessedQuery(initialQuery);
      sendMessage(initialQuery, messages);
    }
  }, [initialQuery]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    sendMessage(userMessage, messages);
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <Bot className="w-12 h-12 mx-auto mb-4 text-emerald-500" />
              <p className="text-lg font-medium text-gray-600">{agentName}</p>
              <p className="text-sm">Ask me anything about Quranic behaviors</p>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          message.isMetricsResponse ? (
            // Render MetricsPanel for metric-intent responses - numbers from deterministic endpoint
            <div key={index} className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-emerald-600" />
              </div>
              <div className="flex-1 max-w-[90%]">
                <MetricsPanel backendUrl={process.env.NEXT_PUBLIC_QBM_BACKEND_URL || "http://localhost:8000"} />
              </div>
            </div>
          ) : (
            <ChatMessage key={index} role={message.role} content={message.content} />
          )
        ))}

        {isLoading && messages[messages.length - 1]?.role === "user" && (
          <div className="flex gap-3 justify-start">
            <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
              <Bot className="w-5 h-5 text-emerald-600" />
            </div>
            <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
              <Loader2 className="w-5 h-5 animate-spin text-emerald-600" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-200 bg-white p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={placeholder}
              disabled={isLoading}
              className="flex-1 px-4 py-3 rounded-xl border border-gray-300 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-200 transition-all disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-6 py-3 bg-emerald-600 text-white rounded-xl hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
