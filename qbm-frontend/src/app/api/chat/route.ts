import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { QBM_SYSTEM_PROMPT, tools, executeTools } from "@/lib/qbm-tools";

// Initialize C1/Thesys client
const client = new OpenAI({
  apiKey: process.env.THESYS_API_KEY,
  baseURL: "https://api.thesys.dev/v1/embed",
});

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();

    // Build conversation with system prompt
    const fullMessages = [
      { role: "system" as const, content: QBM_SYSTEM_PROMPT },
      ...messages,
    ];

    // Initial call - may trigger tool use
    let completion = await client.chat.completions.create({
      model: process.env.C1_MODEL || "c1-nightly",
      messages: fullMessages,
      tools,
      stream: false,
    });

    let assistantMessage = completion.choices[0].message;
    const allMessages = [...fullMessages];

    // Handle tool calls in a loop
    while (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      // Add assistant's tool call message
      allMessages.push(assistantMessage);

      // Execute all tools and get results
      const toolResults = await executeTools(assistantMessage.tool_calls);
      
      // Add tool results to conversation
      for (const result of toolResults) {
        allMessages.push(result);
      }

      // Call again with tool results
      completion = await client.chat.completions.create({
        model: process.env.C1_MODEL || "c1-nightly",
        messages: allMessages,
        tools,
        stream: false,
      });

      assistantMessage = completion.choices[0].message;
    }

    // Final streaming response
    const streamResponse = await client.chat.completions.create({
      model: process.env.C1_MODEL || "c1-nightly",
      messages: [...allMessages, assistantMessage],
      stream: true,
    });

    // Create streaming response
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of streamResponse) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              controller.enqueue(encoder.encode(content));
            }
          }
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return new NextResponse(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Chat API error:", error);
    return NextResponse.json(
      { error: "Failed to process request" },
      { status: 500 }
    );
  }
}
