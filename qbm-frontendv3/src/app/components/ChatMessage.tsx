"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User } from "lucide-react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  if (role === "user") {
    return (
      <div className="flex gap-3 justify-end">
        <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-emerald-600 text-white">
          <div className="whitespace-pre-wrap">{content}</div>
        </div>
        <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-emerald-600" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 justify-start">
      <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center flex-shrink-0">
        <Bot className="w-5 h-5 text-emerald-600" />
      </div>
      <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-white border border-gray-200 text-gray-800">
        <div className="prose prose-sm max-w-none prose-emerald">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children, ...props }) => (
                <div className="overflow-x-auto my-4">
                  <table className="min-w-full border-collapse border border-gray-300" {...props}>
                    {children}
                  </table>
                </div>
              ),
              thead: ({ children, ...props }) => (
                <thead className="bg-emerald-50" {...props}>
                  {children}
                </thead>
              ),
              th: ({ children, ...props }) => (
                <th className="border border-gray-300 px-3 py-2 text-left font-semibold text-emerald-800" {...props}>
                  {children}
                </th>
              ),
              td: ({ children, ...props }) => (
                <td className="border border-gray-300 px-3 py-2" {...props}>
                  {children}
                </td>
              ),
              h1: ({ children, ...props }) => (
                <h1 className="text-xl font-bold text-emerald-800 mt-4 mb-2" {...props}>
                  {children}
                </h1>
              ),
              h2: ({ children, ...props }) => (
                <h2 className="text-lg font-bold text-emerald-700 mt-3 mb-2" {...props}>
                  {children}
                </h2>
              ),
              h3: ({ children, ...props }) => (
                <h3 className="text-base font-semibold text-emerald-600 mt-2 mb-1" {...props}>
                  {children}
                </h3>
              ),
              ul: ({ children, ...props }) => (
                <ul className="list-disc list-inside my-2 space-y-1" {...props}>
                  {children}
                </ul>
              ),
              ol: ({ children, ...props }) => (
                <ol className="list-decimal list-inside my-2 space-y-1" {...props}>
                  {children}
                </ol>
              ),
              li: ({ children, ...props }) => (
                <li className="text-gray-700" {...props}>
                  {children}
                </li>
              ),
              p: ({ children, ...props }) => (
                <p className="my-2 leading-relaxed" {...props}>
                  {children}
                </p>
              ),
              strong: ({ children, ...props }) => (
                <strong className="font-semibold text-emerald-700" {...props}>
                  {children}
                </strong>
              ),
              code: ({ children, className, ...props }) => {
                const isInline = !className;
                if (isInline) {
                  return (
                    <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono text-emerald-700" {...props}>
                      {children}
                    </code>
                  );
                }
                return (
                  <code className={`block bg-gray-100 p-3 rounded-lg overflow-x-auto text-sm font-mono ${className}`} {...props}>
                    {children}
                  </code>
                );
              },
              blockquote: ({ children, ...props }) => (
                <blockquote className="border-l-4 border-emerald-300 pl-4 my-2 italic text-gray-600" {...props}>
                  {children}
                </blockquote>
              ),
            }}
          >
            {content || "..."}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
