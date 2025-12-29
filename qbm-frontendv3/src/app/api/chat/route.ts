import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

// Initialize C1/Thesys client
const client = new OpenAI({
  apiKey: process.env.THESYS_API_KEY || "demo-key",
  baseURL: "https://api.thesys.dev/v1/embed",
});

// System prompt for QBM - NO HARDCODED STATS
// Stats are injected dynamically from /api/metrics/overview
const QBM_SYSTEM_PROMPT_BASE = `أنت مساعد بحث القرآن الكريم - QBM Research Assistant

## 🚨 ABSOLUTE RULE FOR STATISTICS - READ THIS FIRST 🚨

When asked about "توزيع الفاعلين" (agent distribution), "إحصائيات" (statistics), "نسب" (percentages), or "مخطط دائري" (pie chart):

1. DO NOT call any tools (search_spans, get_statistics, etc.)
2. DO NOT analyze retrieved samples
3. DO NOT count nodes from search results
4. ONLY use the exact numbers from METRICS_JSON below

The valid agent types are:
- AGT_ALLAH (الله)
- AGT_DISBELIEVER (الكافر)
- AGT_BELIEVER (المؤمن)
- AGT_HUMAN_GENERAL (الإنسان)
- AGT_WRONGDOER (الظالم)
- AGT_PROPHET (النبي)
- AGT_HYPOCRITE (المنافق)

❌ "ذاكر" is NOT a valid agent type - do not use it
❌ "90%" and "10%" are WRONG - never output these
❌ Do not say "10 عقد" or "9 من 10" - these are sample counts, not real data

## WHEN ASKED FOR PIE CHART OF AGENT DISTRIBUTION

Respond EXACTLY like this (using numbers from METRICS_JSON):

توزيع أنواع الفاعلين في القرآن الكريم:

| الفاعل | العدد | النسبة |
|--------|-------|--------|
| الله (AGT_ALLAH) | [من METRICS_JSON] | [من METRICS_JSON] |
| الكافر (AGT_DISBELIEVER) | [من METRICS_JSON] | [من METRICS_JSON] |
| المؤمن (AGT_BELIEVER) | [من METRICS_JSON] | [من METRICS_JSON] |
| الإنسان (AGT_HUMAN_GENERAL) | [من METRICS_JSON] | [من METRICS_JSON] |

## YOUR ROLE

- Answer questions about Quranic behaviors using the Bouzidani methodology
- For statistics: ONLY use METRICS_JSON numbers
- If METRICS_JSON is missing, say: "يرجى زيارة صفحة /metrics للإحصائيات الرسمية"`;

// Function to build system prompt with metrics from backend
async function buildSystemPromptWithMetrics(): Promise<string> {
  const QBM_BACKEND_URL = process.env.QBM_BACKEND_URL || "http://localhost:8000";
  
  try {
    const res = await fetch(`${QBM_BACKEND_URL}/api/metrics/overview`, {
      signal: AbortSignal.timeout(3000),
    });
    
    if (res.ok) {
      const metricsData = await res.json();
      if (metricsData.status === "ready") {
        // Inject real metrics into prompt
        return `${QBM_SYSTEM_PROMPT_BASE}

## METRICS_JSON (Source of Truth - use ONLY these numbers)

\`\`\`json
${JSON.stringify(metricsData.metrics, null, 2)}
\`\`\`

When asked about statistics, you may ONLY reference numbers from the METRICS_JSON above.
If a metric is not in METRICS_JSON, say "غير متوفر" (unavailable).`;
      }
    }
  } catch (e) {
    console.log("[CHAT] Could not fetch metrics, using base prompt");
  }
  
  // Fallback: no metrics available
  return `${QBM_SYSTEM_PROMPT_BASE}

## METRICS_JSON

No metrics available. If asked for statistics, respond:
"الإحصائيات غير متوفرة حالياً. يرجى زيارة صفحة /metrics للاطلاع على الإحصائيات الرسمية."
(Statistics are currently unavailable. Please visit /metrics for official statistics.)`;
}

// Tool definitions
const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search_spans",
      description: "Search behavioral annotations in the QBM database",
      parameters: {
        type: "object",
        properties: {
          behavior_concept: { type: "string", description: "Behavior ID (e.g., BEH_PATIENCE)" },
          surah: { type: "integer", description: "Surah number (1-114)" },
          agent_type: { type: "string", description: "Agent type (e.g., BELIEVER)" },
          organ: { type: "string", description: "Organ involved (e.g., HEART)" },
          limit: { type: "integer", description: "Max results", default: 20 },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_tafsir",
      description: "Get tafsir for a specific ayah",
      parameters: {
        type: "object",
        properties: {
          surah: { type: "integer", description: "Surah number" },
          ayah: { type: "integer", description: "Ayah number" },
          sources: {
            type: "array",
            items: { type: "string" },
            description: "Tafsir sources",
          },
        },
        required: ["surah", "ayah"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_statistics",
      description: "Get dataset statistics",
      parameters: {
        type: "object",
        properties: {
          group_by: { type: "string", enum: ["surah", "behavior", "agent", "organ"] },
          metric: { type: "string", enum: ["count", "coverage", "distribution"] },
        },
      },
    },
  },
];

// Backend URL for real data
const QBM_BACKEND_URL = process.env.QBM_BACKEND_URL || "http://localhost:8000";

// Real tool execution - calls QBM backend
async function executeTools(toolCalls: any[]) {
  const results = [];

  for (const call of toolCalls) {
    const args = JSON.parse(call.function.arguments || "{}");
    let result: any = {};

    try {
      switch (call.function.name) {
        case "search_spans": {
          const params = new URLSearchParams();
          if (args.surah) params.set("surah", args.surah.toString());
          if (args.agent_type) params.set("agent_type", `AGT_${args.agent_type}`);
          if (args.limit) params.set("limit", args.limit.toString());
          
          const res = await fetch(`${QBM_BACKEND_URL}/spans?${params}`);
          if (res.ok) {
            result = await res.json();
          } else {
            result = { error: "Failed to fetch spans", status: res.status };
          }
          break;
        }

        case "get_tafsir": {
          const res = await fetch(`${QBM_BACKEND_URL}/tafsir/compare/${args.surah}/${args.ayah}`);
          if (res.ok) {
            result = await res.json();
          } else {
            result = { error: "Failed to fetch tafsir", status: res.status };
          }
          break;
        }

        case "get_statistics": {
          // Use truth metrics endpoint for accurate statistics
          const res = await fetch(`${QBM_BACKEND_URL}/api/metrics/overview`);
          if (res.ok) {
            const truthMetrics = await res.json();
            if (truthMetrics.status === "ready") {
              // Return formatted statistics from truth metrics
              const metrics = truthMetrics.metrics;
              result = {
                source: "truth_metrics_v1",
                total_spans: metrics.totals.spans,
                tafsir_sources_count: metrics.totals.tafsir_sources_count,
                agent_distribution: metrics.agent_distribution.items.map((item: any) => ({
                  agent: item.key,
                  label_ar: item.label_ar,
                  count: item.count,
                  percentage: item.percentage,
                })),
                behavior_forms: metrics.behavior_forms.items.map((item: any) => ({
                  form: item.key,
                  label_ar: item.label_ar,
                  count: item.count,
                  percentage: item.percentage,
                })),
                evaluations: metrics.evaluations.items.map((item: any) => ({
                  evaluation: item.key,
                  label_ar: item.label_ar,
                  count: item.count,
                  percentage: item.percentage,
                })),
                _instruction: "USE THESE EXACT NUMBERS. Do not calculate or infer different percentages.",
              };
            } else {
              result = { error: "Truth metrics not ready", status: truthMetrics.status };
            }
          } else {
            result = { error: "Failed to fetch statistics", status: res.status };
          }
          break;
        }

        default:
          result = { error: `Unknown tool: ${call.function.name}` };
      }
    } catch (error) {
      result = { 
        error: `Tool execution failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        note: "Ensure QBM backend is running at " + QBM_BACKEND_URL
      };
    }

    results.push({
      role: "tool" as const,
      tool_call_id: call.id,
      content: JSON.stringify(result),
    });
  }

  return results;
}

// Detect if question is asking for statistics/distribution
function isStatisticsQuestion(text: string): boolean {
  const statsPatterns = [
    /توزيع.*الفاعلين/,
    /توزيع.*أنواع/,
    /مخطط.*دائري/,
    /إحصائيات/,
    /نسب.*الفاعلين/,
    /agent.*distribution/i,
    /pie.*chart/i,
    /statistics/i,
  ];
  return statsPatterns.some(p => p.test(text));
}

// Generate deterministic response for statistics questions
async function generateStatisticsResponse(): Promise<string | null> {
  const QBM_BACKEND_URL = process.env.QBM_BACKEND_URL || "http://localhost:8000";
  
  try {
    const res = await fetch(`${QBM_BACKEND_URL}/api/metrics/overview`);
    if (!res.ok) return null;
    
    const data = await res.json();
    if (data.status !== "ready") return null;
    
    const metrics = data.metrics;
    const agents = metrics.agent_distribution.items;
    
    // Build deterministic response with REAL data
    let response = `## توزيع أنواع الفاعلين في القرآن الكريم

**المصدر:** truth_metrics_v1 (بيانات حقيقية من قاعدة البيانات)
**إجمالي النطاقات:** ${metrics.totals.spans.toLocaleString()}

### جدول التوزيع

| الفاعل | النوع | العدد | النسبة |
|--------|-------|-------|--------|
`;
    
    for (const agent of agents) {
      response += `| ${agent.label_ar} | ${agent.key} | ${agent.count.toLocaleString()} | ${agent.percentage}% |\n`;
    }
    
    response += `
### أشكال السلوك

| الشكل | النوع | العدد | النسبة |
|-------|-------|-------|--------|
`;
    
    for (const form of metrics.behavior_forms.items) {
      response += `| ${form.label_ar} | ${form.key} | ${form.count.toLocaleString()} | ${form.percentage}% |\n`;
    }
    
    response += `
### التقييمات

| التقييم | النوع | العدد | النسبة |
|---------|-------|-------|--------|
`;
    
    for (const ev of metrics.evaluations.items) {
      response += `| ${ev.label_ar} | ${ev.key} | ${ev.count.toLocaleString()} | ${ev.percentage}% |\n`;
    }
    
    response += `
---
*هذه البيانات محسوبة من مجموعة البيانات الأساسية (${metrics.totals.spans} نطاق سلوكي). للمزيد من التفاصيل، يرجى زيارة صفحة /metrics*`;
    
    return response;
  } catch (e) {
    console.error("[CHAT] Failed to generate statistics response:", e);
    return null;
  }
}

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();
    
    // Get the last user message
    const lastUserMessage = messages.filter((m: any) => m.role === "user").pop();
    const userQuery = lastUserMessage?.content || "";
    
    // CRITICAL: For statistics questions, bypass LLM entirely and return deterministic data
    if (isStatisticsQuestion(userQuery)) {
      const statsResponse = await generateStatisticsResponse();
      if (statsResponse) {
        // Return deterministic response directly - no LLM involved
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(statsResponse));
            controller.close();
          },
        });
        
        return new NextResponse(stream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      }
    }

    // Build system prompt with real metrics from backend
    const systemPrompt = await buildSystemPromptWithMetrics();

    // Build conversation
    const fullMessages = [
      { role: "system" as const, content: systemPrompt },
      ...messages,
    ];

    // Initial call with tools
    let completion = await client.chat.completions.create({
      model: process.env.C1_MODEL || "c1-nightly",
      messages: fullMessages,
      tools,
      stream: false,
    });

    let assistantMessage = completion.choices[0].message;
    const allMessages = [...fullMessages];

    // Handle tool calls
    while (assistantMessage.tool_calls?.length) {
      allMessages.push(assistantMessage);
      const toolResults = await executeTools(assistantMessage.tool_calls);
      allMessages.push(...toolResults);

      completion = await client.chat.completions.create({
        model: process.env.C1_MODEL || "c1-nightly",
        messages: allMessages,
        tools,
        stream: false,
      });
      assistantMessage = completion.choices[0].message;
    }

    // Stream final response
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
    console.error("API error:", error);
    return NextResponse.json({ error: "Failed to process request" }, { status: 500 });
  }
}
