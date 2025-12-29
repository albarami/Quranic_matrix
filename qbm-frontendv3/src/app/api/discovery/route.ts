import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Initialize C1 client with TheSys endpoint
const THESYS_API_KEY = process.env.THESYS_API_KEY;
const QBM_BACKEND_URL = process.env.QBM_BACKEND_URL || "http://localhost:8000";

const client = THESYS_API_KEY
  ? new OpenAI({
      apiKey: THESYS_API_KEY,
      baseURL: "https://api.thesys.dev/v1/embed",
    })
  : null;

// In-memory message store
type DBMessage = OpenAI.Chat.ChatCompletionMessageParam & { id?: string };
const messagesStore: { [threadId: string]: DBMessage[] } = {};

const getMessageStore = (threadId: string) => {
  if (!messagesStore[threadId]) {
    messagesStore[threadId] = [];
  }
  const messageList = messagesStore[threadId];
  return {
    addMessage: (message: DBMessage) => messageList.push(message),
    getMessages: () => messageList.map((m) => {
      const msg = { ...m };
      delete msg.id;
      return msg;
    }),
  };
};

// =============================================================================
// QBM DISCOVERY SYSTEM PROMPT - Optimized for C1 Generative UI
// =============================================================================

const QBM_DISCOVERY_PROMPT = `أنت نظام اكتشاف السلوك القرآني - QBM Discovery System
You are an AI-powered Quranic Behavioral Discovery assistant with access to 322,939 annotations across 6,236 ayat.

═══════════════════════════════════════════════════════════════════════════════
                         🎯 YOUR CAPABILITIES
═══════════════════════════════════════════════════════════════════════════════

1. **Semantic Search** - GPU-accelerated search across all annotations using AraBERT embeddings
2. **Pattern Discovery** - Find co-occurring behaviors (4,715 pairs with count ≥10)
3. **Cross-Reference** - Find similar ayat and behavior relationships (12,017 edges)
4. **Thematic Clustering** - 15 behavioral clusters with silhouette score optimization

═══════════════════════════════════════════════════════════════════════════════
                         📊 UI GENERATION RULES
═══════════════════════════════════════════════════════════════════════════════

You generate RICH, INTERACTIVE UI components. Follow these rules:

1. **For Search Results**: Use tables with clickable rows, show surah:ayah, Arabic text, behavior type
2. **For Statistics**: Use bar charts, pie charts, or area charts with real data
3. **For Comparisons**: Use comparison tables or side-by-side cards
4. **For Patterns**: Use network visualizations or relationship diagrams
5. **For Clusters**: Use cards with expandable details

ALWAYS include:
- Real numbers from the database
- Arabic text with proper RTL formatting
- Interactive elements (buttons to explore more, drill-down actions)
- Visual hierarchy with headers, sections, and callouts

═══════════════════════════════════════════════════════════════════════════════
                         🔧 AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════════════════════

Use these tools to fetch REAL data before generating UI:

1. semantic_search - Search annotations by meaning (Arabic/English)
2. find_patterns - Get co-occurring behavior pairs
3. find_similar_ayat - Find semantically similar verses
4. get_behavior_distribution - Get behavior frequency distribution
5. get_surah_themes - Get dominant themes for a surah
6. get_cross_references - Find related verses for an ayah
7. get_cluster_info - Get thematic cluster details
8. get_discovery_stats - Get overall discovery system statistics

═══════════════════════════════════════════════════════════════════════════════
                         💡 RESPONSE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

When user asks "ابحث عن الصبر" (search for patience):
→ Call semantic_search with query "الصبر"
→ Generate a table showing matching ayat with:
  - Reference (سورة:آية)
  - Arabic text
  - Behavior type
  - Score
→ Add buttons: "عرض التفسير" (Show Tafsir), "آيات مشابهة" (Similar Verses)

When user asks "ما هي أنماط السلوك؟" (what are behavior patterns?):
→ Call find_patterns
→ Generate a bar chart of top co-occurring pairs
→ Show network visualization of behavior relationships

When user asks about a specific surah:
→ Call get_surah_themes
→ Generate pie chart of behavior distribution
→ List top behaviors with counts

═══════════════════════════════════════════════════════════════════════════════
                         🌐 BILINGUAL SUPPORT
═══════════════════════════════════════════════════════════════════════════════

- Respond in the same language as the user's query
- Always show Arabic Quranic text with proper formatting
- Use RTL direction for Arabic content
- Provide English translations when helpful
`;

// =============================================================================
// CUSTOM COMPONENT SCHEMAS - For C1 to generate rich UI
// =============================================================================

const AyahCardSchema = z.object({
  surah: z.number().describe("Surah number (1-114)"),
  ayah: z.number().describe("Ayah number"),
  surahName: z.string().optional().describe("Surah name in Arabic"),
  textAr: z.string().describe("Arabic text of the ayah"),
  behavior: z.string().optional().describe("Behavior ID if applicable"),
  score: z.number().optional().describe("Relevance score (0-1)"),
  source: z.string().optional().describe("Source of annotation"),
}).describe("Displays a Quranic verse card with Arabic text, reference, and interactive actions for tafsir, similar verses, and behavior analysis.");

const BehaviorNetworkSchema = z.object({
  pairs: z.array(z.object({
    behavior_1: z.string().describe("First behavior ID"),
    behavior_2: z.string().describe("Second behavior ID"),
    cooccurrence_count: z.number().describe("Number of co-occurrences"),
  })).describe("Array of behavior pairs that co-occur"),
  title: z.string().optional().describe("Title for the network visualization"),
}).describe("Displays a network visualization of co-occurring behaviors with clickable nodes to explore each behavior.");

const ClusterCardSchema = z.object({
  clusterId: z.number().describe("Cluster ID (0-14)"),
  size: z.number().describe("Number of annotations in cluster"),
  themes: z.array(z.object({
    behavior: z.string().describe("Behavior ID"),
    count: z.number().describe("Count in cluster"),
  })).describe("Top themes/behaviors in this cluster"),
  silhouetteScore: z.number().optional().describe("Cluster quality score"),
}).describe("Displays a thematic cluster card showing dominant behaviors and size, with action to explore cluster samples.");

const CUSTOM_COMPONENT_SCHEMAS = {
  AyahCard: zodToJsonSchema(AyahCardSchema),
  BehaviorNetwork: zodToJsonSchema(BehaviorNetworkSchema),
  ClusterCard: zodToJsonSchema(ClusterCardSchema),
};

// =============================================================================
// DISCOVERY TOOLS - Connected to FastAPI Discovery Endpoints
// =============================================================================

const discoveryTools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "semantic_search",
      description: "GPU-accelerated semantic search across 322,939 annotations using AraBERT embeddings",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query in Arabic or English" },
          top_k: { type: "integer", description: "Number of results (default: 10)", default: 10 },
          source_filter: { type: "string", description: "Filter by source (optional)" },
          type_filter: { type: "string", description: "Filter by annotation type (optional)" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "find_patterns",
      description: "Find co-occurring behavior pairs from 4,715 discovered patterns",
      parameters: {
        type: "object",
        properties: {
          min_count: { type: "integer", description: "Minimum co-occurrence count", default: 50 },
          limit: { type: "integer", description: "Max results", default: 20 },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "find_similar_ayat",
      description: "Find semantically similar verses to a given ayah",
      parameters: {
        type: "object",
        properties: {
          surah: { type: "integer", description: "Surah number (1-114)" },
          ayah: { type: "integer", description: "Ayah number" },
          top_k: { type: "integer", description: "Number of similar verses", default: 10 },
        },
        required: ["surah", "ayah"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_behavior_distribution",
      description: "Get distribution of behaviors across all annotations",
      parameters: {
        type: "object",
        properties: {
          behavior_only: { type: "boolean", description: "Only include type=behavior", default: true },
          limit: { type: "integer", description: "Max behaviors to return", default: 15 },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_surah_themes",
      description: "Get dominant behavioral themes for a specific surah",
      parameters: {
        type: "object",
        properties: {
          surah_num: { type: "integer", description: "Surah number (1-114)" },
          top_n: { type: "integer", description: "Number of top themes", default: 5 },
        },
        required: ["surah_num"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_cross_references",
      description: "Find cross-references for an ayah (semantic and behavior-based)",
      parameters: {
        type: "object",
        properties: {
          surah: { type: "integer", description: "Surah number" },
          ayah: { type: "integer", description: "Ayah number" },
          top_k: { type: "integer", description: "Number of references", default: 10 },
        },
        required: ["surah", "ayah"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_cluster_info",
      description: "Get information about thematic clusters",
      parameters: {
        type: "object",
        properties: {
          cluster_id: { type: "integer", description: "Specific cluster ID (0-14), or omit for all" },
          n_samples: { type: "integer", description: "Sample annotations per cluster", default: 5 },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_discovery_stats",
      description: "Get overall discovery system statistics",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "get_ayat_by_behavior",
      description: "Find all ayat discussing a specific behavior",
      parameters: {
        type: "object",
        properties: {
          behavior: { type: "string", description: "Behavior ID (e.g., BEH_SPEECH_GOOD_WORD)" },
          include_similar: { type: "boolean", description: "Include similar behaviors", default: true },
          limit: { type: "integer", description: "Max results", default: 50 },
        },
        required: ["behavior"],
      },
    },
  },
];

// =============================================================================
// TOOL EXECUTION - Call FastAPI Discovery Endpoints
// =============================================================================

async function executeDiscoveryTools(toolCalls: any[]) {
  const results = [];

  for (const call of toolCalls) {
    const args = JSON.parse(call.function.arguments || "{}");
    let result: any = {};

    try {
      switch (call.function.name) {
        case "semantic_search": {
          const params = new URLSearchParams({
            q: args.query,
            top_k: String(args.top_k || 10),
          });
          if (args.source_filter) params.append("source", args.source_filter);
          if (args.type_filter) params.append("type", args.type_filter);

          const res = await fetch(`${QBM_BACKEND_URL}/discovery/search?${params}`);
          if (!res.ok) throw new Error(`Search failed: ${res.status}`);
          result = await res.json();
          break;
        }

        case "find_patterns": {
          const params = new URLSearchParams({
            min_count: String(args.min_count || 50),
            limit: String(args.limit || 20),
          });
          const res = await fetch(`${QBM_BACKEND_URL}/discovery/patterns/cooccurrence?${params}`);
          if (!res.ok) throw new Error(`Patterns failed: ${res.status}`);
          result = await res.json();
          break;
        }

        case "find_similar_ayat": {
          const res = await fetch(
            `${QBM_BACKEND_URL}/discovery/similar/${args.surah}/${args.ayah}?top_k=${args.top_k || 10}`
          );
          if (!res.ok) throw new Error(`Similar ayat failed: ${res.status}`);
          result = await res.json();
          break;
        }

        case "get_behavior_distribution": {
          const params = new URLSearchParams({
            behavior_only: String(args.behavior_only !== false),
            limit: String(args.limit || 15),
          });
          const res = await fetch(`${QBM_BACKEND_URL}/discovery/patterns/distribution?${params}`);
          if (!res.ok) throw new Error(`Distribution failed: ${res.status}`);
          result = await res.json();
          break;
        }

        case "get_surah_themes": {
          const res = await fetch(
            `${QBM_BACKEND_URL}/discovery/patterns/surah/${args.surah_num}?top_n=${args.top_n || 5}`
          );
          if (!res.ok) throw new Error(`Surah themes failed: ${res.status}`);
          result = await res.json();
          break;
        }

        case "get_cross_references": {
          const res = await fetch(
            `${QBM_BACKEND_URL}/discovery/crossref/${args.surah}/${args.ayah}?top_k=${args.top_k || 10}`
          );
          if (!res.ok) throw new Error(`Cross-ref failed: ${res.status}`);
          result = await res.json();
          break;
        }

        case "get_cluster_info": {
          if (args.cluster_id !== undefined) {
            const res = await fetch(
              `${QBM_BACKEND_URL}/discovery/cluster/${args.cluster_id}/samples?n_samples=${args.n_samples || 5}`
            );
            if (!res.ok) throw new Error(`Cluster info failed: ${res.status}`);
            result = await res.json();
          } else {
            // POST endpoint for clustering
            const res = await fetch(`${QBM_BACKEND_URL}/discovery/cluster?n_clusters=15`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
            });
            if (!res.ok) throw new Error(`Clustering failed: ${res.status}`);
            result = await res.json();
          }
          break;
        }

        case "get_discovery_stats": {
          const res = await fetch(`${QBM_BACKEND_URL}/discovery/stats`);
          if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
          result = await res.json();
          break;
        }

        case "get_ayat_by_behavior": {
          const params = new URLSearchParams({
            include_similar: String(args.include_similar !== false),
            limit: String(args.limit || 50),
          });
          const res = await fetch(
            `${QBM_BACKEND_URL}/discovery/crossref/behavior/${encodeURIComponent(args.behavior)}?${params}`
          );
          if (!res.ok) throw new Error(`Behavior ayat failed: ${res.status}`);
          result = await res.json();
          break;
        }

        default:
          result = { error: `Unknown tool: ${call.function.name}` };
      }
    } catch (error) {
      console.error(`Tool error [${call.function.name}]:`, error);
      result = {
        error: `Failed: ${call.function.name}`,
        details: error instanceof Error ? error.message : "Unknown error",
        hint: "Ensure QBM backend is running with discovery endpoints",
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

// =============================================================================
// API HANDLER - C1 Generative UI
// =============================================================================

export async function POST(req: NextRequest) {
  if (!client) {
    return NextResponse.json(
      { error: "THESYS_API_KEY not configured" },
      { status: 503 }
    );
  }

  try {
    const { prompt, threadId, responseId } = (await req.json()) as {
      prompt: DBMessage;
      threadId: string;
      responseId: string;
    };

    const messageStore = getMessageStore(threadId);
    messageStore.addMessage(prompt);

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      { role: "system", content: QBM_DISCOVERY_PROMPT },
      ...messageStore.getMessages(),
    ];

    // C1 metadata with custom components
    const c1Metadata = {
      thesys: JSON.stringify({
        c1_custom_components: CUSTOM_COMPONENT_SCHEMAS,
      }),
    };

    // First call with tools
    let completion = await client.chat.completions.create({
      model: "c1/anthropic/claude-3.7-sonnet/v-20250617",
      messages,
      tools: discoveryTools,
      stream: false,
      // @ts-ignore - metadata is supported by C1
      metadata: c1Metadata,
    });

    let assistantMessage = completion.choices[0].message;
    const allMessages = [...messages];

    // Handle tool calls loop
    while (assistantMessage.tool_calls?.length) {
      allMessages.push(assistantMessage);
      const toolResults = await executeDiscoveryTools(assistantMessage.tool_calls);
      allMessages.push(...toolResults);

      completion = await client.chat.completions.create({
        model: "c1/anthropic/claude-3.7-sonnet/v-20250617",
        messages: allMessages,
        tools: discoveryTools,
        stream: false,
        // @ts-ignore
        metadata: c1Metadata,
      });
      assistantMessage = completion.choices[0].message;
    }

    // Stream final response
    const streamResponse = await client.chat.completions.create({
      model: "c1/anthropic/claude-3.7-sonnet/v-20250617",
      messages: [...allMessages, assistantMessage],
      stream: true,
      // @ts-ignore
      metadata: c1Metadata,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          let fullContent = "";
          for await (const chunk of streamResponse) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              fullContent += content;
              controller.enqueue(encoder.encode(content));
            }
          }
          messageStore.addMessage({
            role: "assistant",
            content: fullContent,
            id: responseId,
          });
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return new NextResponse(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Discovery API error:", error);
    return NextResponse.json({ error: "Failed to process request" }, { status: 500 });
  }
}
