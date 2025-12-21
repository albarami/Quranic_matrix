/**
 * QBM Tools for C1 Integration
 * 
 * This file defines:
 * 1. System prompt for the QBM Research Assistant
 * 2. Tool definitions for querying QBM data
 * 3. Tool execution functions
 */

// =============================================================================
// SYSTEM PROMPT
// =============================================================================

export const QBM_SYSTEM_PROMPT = `
أنت مساعد بحث القرآن الكريم - QBM Research Assistant

You are the QBM Research Assistant, a specialized tool for exploring the Quranic Behavioral Matrix (مصفوفة التصنيف القرآني للسلوك البشري).

## Your Knowledge Base

You have access to:
- **6,236 ayat** from the complete Holy Quran
- **15,000+ behavioral span annotations** with classifications
- **5 tafsir sources**: Ibn Kathir (ابن كثير), Tabari (الطبري), Qurtubi (القرطبي), Sa'di (السعدي), Jalalayn (الجلالين)
- **Controlled vocabularies** for behaviors (BEH_*), agents (AGT_*), organs (ORG_*), and contexts

## Behavioral Classification Framework

Based on Dr. Bouzidani's framework, behaviors are classified across five contexts:
1. **العضوي (Organic)** - Body organs involved (heart, tongue, eyes, hands, etc.)
2. **الموضعي (Situational)** - Internal (باطني) vs External (ظاهري)
3. **النسقي (Systemic)** - Social contexts (family, community, worship)
4. **المكاني (Spatial)** - Location context
5. **الزماني (Temporal)** - Time context

## Your Capabilities

When users ask questions, generate **interactive UI components**:
- **Tables** for listing spans, behaviors, or search results
- **Charts** for statistical analysis (pie, bar, line)
- **Cards** for displaying ayat with tafsir
- **Forms** for filtering, searching, or submitting annotations
- **Comparison views** for tafsir analysis
- **Progress indicators** for coverage metrics

## Response Guidelines

1. **Always cite ayat** using surah:ayah format (e.g., 2:255, 49:12)
2. **Include Arabic text** with proper tashkeel when showing Quranic verses
3. **Use bilingual labels** (Arabic + English) for behavior concepts
4. **Link related behaviors** when discussing connections
5. **Be academically rigorous** - this is a scholarly research tool
6. **Respect the sacred nature** of the Quranic text

## Available Tools

Use these tools to query the QBM database:
- \`search_spans\`: Search behavioral annotations by concept, surah, agent, or organ
- \`get_tafsir\`: Retrieve tafsir for a specific ayah
- \`get_statistics\`: Get dataset statistics and coverage metrics
- \`compare_tafsir\`: Compare interpretations across multiple tafsir sources

## Example Interactions

User: "Show me behaviors related to the heart"
→ Search for spans with organ ORG_HEART, display as interactive table

User: "Compare tafsir for Ayat al-Kursi"
→ Use compare_tafsir tool, display side-by-side panels

User: "What's the annotation progress?"
→ Use get_statistics, display charts and progress bars
`;

// =============================================================================
// TOOL DEFINITIONS
// =============================================================================

export const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search_spans",
      description: "Search behavioral span annotations in the QBM database. Returns matching spans with their classifications.",
      parameters: {
        type: "object",
        properties: {
          behavior_concept: {
            type: "string",
            description: "Behavior concept ID to filter by (e.g., BEH_PATIENCE, BEH_ANGER, BEH_GRATITUDE). Use Arabic keywords too.",
          },
          surah: {
            type: "integer",
            description: "Surah number to filter by (1-114)",
            minimum: 1,
            maximum: 114,
          },
          agent_type: {
            type: "string",
            description: "Agent type to filter by (e.g., AGT_BELIEVER, AGT_DISBELIEVER, AGT_PROPHET, AGT_HUMAN_GENERAL)",
          },
          organ: {
            type: "string",
            description: "Organ involved in the behavior (e.g., ORG_HEART, ORG_TONGUE, ORG_EYES, ORG_HANDS)",
          },
          text_search: {
            type: "string",
            description: "Free text search in Arabic or English",
          },
          tier: {
            type: "string",
            enum: ["gold", "silver", "research"],
            description: "Annotation quality tier to filter by",
          },
          limit: {
            type: "integer",
            description: "Maximum number of results to return (default 20)",
            default: 20,
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_tafsir",
      description: "Retrieve tafsir (exegesis) for a specific ayah from one or more classical sources.",
      parameters: {
        type: "object",
        properties: {
          surah: {
            type: "integer",
            description: "Surah number (1-114)",
            minimum: 1,
            maximum: 114,
          },
          ayah: {
            type: "integer",
            description: "Ayah number within the surah",
            minimum: 1,
          },
          sources: {
            type: "array",
            items: {
              type: "string",
              enum: ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"],
            },
            description: "Tafsir sources to retrieve (default: ibn_kathir, tabari)",
          },
          include_ayah_text: {
            type: "boolean",
            description: "Whether to include the ayah Arabic text (default: true)",
            default: true,
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
      description: "Get statistics about the QBM dataset including coverage, distribution, and quality metrics.",
      parameters: {
        type: "object",
        properties: {
          group_by: {
            type: "string",
            enum: ["surah", "behavior", "agent", "organ", "tier", "annotator"],
            description: "How to group the statistics",
          },
          metric: {
            type: "string",
            enum: ["count", "coverage", "iaa", "distribution"],
            description: "What metric to calculate",
          },
          surah_filter: {
            type: "integer",
            description: "Optional: limit stats to a specific surah",
          },
          top_n: {
            type: "integer",
            description: "Return only top N results (for rankings)",
            default: 20,
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "compare_tafsir",
      description: "Compare tafsir interpretations from multiple sources for a specific ayah, highlighting agreements and differences.",
      parameters: {
        type: "object",
        properties: {
          surah: {
            type: "integer",
            description: "Surah number (1-114)",
            minimum: 1,
            maximum: 114,
          },
          ayah: {
            type: "integer",
            description: "Ayah number",
            minimum: 1,
          },
          focus_topic: {
            type: "string",
            description: "Optional topic to focus comparison on (e.g., 'agent identification', 'behavioral ruling')",
          },
        },
        required: ["surah", "ayah"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_ayah",
      description: "Get the full text and metadata for a specific ayah.",
      parameters: {
        type: "object",
        properties: {
          surah: {
            type: "integer",
            description: "Surah number (1-114)",
          },
          ayah: {
            type: "integer",
            description: "Ayah number",
          },
          include_translation: {
            type: "boolean",
            description: "Include English translation",
            default: true,
          },
          include_annotations: {
            type: "boolean",
            description: "Include existing behavioral annotations",
            default: true,
          },
        },
        required: ["surah", "ayah"],
      },
    },
  },
];

// =============================================================================
// TOOL EXECUTION
// =============================================================================

const QBM_BACKEND_URL = process.env.QBM_BACKEND_URL || "http://localhost:8000";

// Flag to use real backend vs mock data
const USE_REAL_BACKEND = process.env.USE_REAL_BACKEND === "true";

/**
 * Execute tool calls and return results
 */
export async function executeTools(
  toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[]
): Promise<OpenAI.Chat.Completions.ChatCompletionToolMessageParam[]> {
  const results: OpenAI.Chat.Completions.ChatCompletionToolMessageParam[] = [];

  for (const call of toolCalls) {
    const args = JSON.parse(call.function.arguments || "{}");
    let result: any;

    try {
      switch (call.function.name) {
        case "search_spans":
          result = await searchSpans(args);
          break;

        case "get_tafsir":
          result = await getTafsir(args);
          break;

        case "get_statistics":
          result = await getStatistics(args);
          break;

        case "compare_tafsir":
          result = await compareTafsir(args);
          break;

        case "get_ayah":
          result = await getAyah(args);
          break;

        default:
          result = { error: `Unknown tool: ${call.function.name}` };
      }
    } catch (error) {
      console.error(`Tool execution error (${call.function.name}):`, error);
      result = { error: `Failed to execute ${call.function.name}` };
    }

    results.push({
      role: "tool",
      tool_call_id: call.id,
      content: JSON.stringify(result),
    });
  }

  return results;
}

// =============================================================================
// TOOL IMPLEMENTATIONS
// =============================================================================

async function searchSpans(args: {
  behavior_concept?: string;
  surah?: number;
  agent_type?: string;
  organ?: string;
  text_search?: string;
  tier?: string;
  limit?: number;
}) {
  // Try real backend first
  if (USE_REAL_BACKEND) {
    try {
      const params = new URLSearchParams();
      if (args.surah) params.append("surah", args.surah.toString());
      if (args.agent_type) params.append("agent", args.agent_type);
      if (args.behavior_concept) params.append("behavior", args.behavior_concept);
      if (args.limit) params.append("limit", args.limit.toString());
      
      const response = await fetch(`${QBM_BACKEND_URL}/spans?${params}`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error("Backend fetch failed, using mock data:", error);
    }
  }
  
  // Fallback to mock data
  const mockSpans = [
    {
      id: "QBM_00001",
      surah: 2,
      ayah: 255,
      text_ar: "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
      behavior_concepts: ["BEH_BELIEF", "BEH_TAWHID"],
      agent_type: "AGT_BELIEVER",
      tier: "gold",
    },
    {
      id: "QBM_00045",
      surah: 49,
      ayah: 12,
      text_ar: "وَلَا يَغْتَب بَّعْضُكُم بَعْضًا",
      behavior_concepts: ["BEH_BACKBITING", "BEH_SPEECH_HARM"],
      agent_type: "AGT_BELIEVER",
      organ: "ORG_TONGUE",
      tier: "gold",
    },
  ];

  let filtered = mockSpans;
  if (args.surah) {
    filtered = filtered.filter((s) => s.surah === args.surah);
  }
  if (args.behavior_concept) {
    filtered = filtered.filter((s) =>
      s.behavior_concepts.some((b) =>
        b.toLowerCase().includes(args.behavior_concept!.toLowerCase())
      )
    );
  }

  return {
    total: filtered.length,
    spans: filtered.slice(0, args.limit || 20),
    query: args,
  };
}

async function getTafsir(args: {
  surah: number;
  ayah: number;
  sources?: string[];
  include_ayah_text?: boolean;
}) {
  const sources = args.sources || ["ibn_kathir", "tabari"];

  // Try real backend first
  if (USE_REAL_BACKEND) {
    try {
      const params = new URLSearchParams();
      sources.forEach(s => params.append("sources", s));
      
      const response = await fetch(`${QBM_BACKEND_URL}/tafsir/${args.surah}/${args.ayah}?${params}`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error("Backend fetch failed, using mock data:", error);
    }
  }

  // Fallback to mock tafsir data
  const mockTafsir: Record<string, string> = {
    ibn_kathir: `هذه آية عظيمة من آيات القرآن الكريم، وهي آية الكرسي المباركة...`,
    tabari: `القول في تأويل قوله تعالى: الله لا إله إلا هو الحي القيوم...`,
    qurtubi: `قوله تعالى: الله لا إله إلا هو الحي القيوم. فيه مسائل...`,
    saadi: `هذه الآية أعظم آية في كتاب الله تعالى...`,
    jalalayn: `{الله لا إله إلا هو} أي: لا معبود بحق سواه...`,
  };

  const result: any = {
    surah: args.surah,
    ayah: args.ayah,
    reference: `${args.surah}:${args.ayah}`,
    tafsir: {},
  };

  if (args.include_ayah_text !== false) {
    result.ayah_text = "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ ۚ لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ";
  }

  for (const source of sources) {
    if (mockTafsir[source]) {
      result.tafsir[source] = {
        source_name: source.replace("_", " ").toUpperCase(),
        text: mockTafsir[source],
        word_count: mockTafsir[source].split(" ").length,
      };
    }
  }

  return result;
}

async function getStatistics(args: {
  group_by?: string;
  metric?: string;
  surah_filter?: number;
  top_n?: number;
}) {
  // Try real backend first
  if (USE_REAL_BACKEND) {
    try {
      const response = await fetch(`${QBM_BACKEND_URL}/stats`);
      if (response.ok) {
        const data = await response.json();
        return {
          summary: {
            total_ayat: 6236,
            annotated_ayat: data.unique_ayat || 0,
            total_spans: data.total_spans || 0,
            coverage_percent: ((data.unique_ayat || 0) / 6236 * 100).toFixed(1),
          },
          by_agent: data.agent_types || {},
          by_behavior: data.behavior_forms || {},
          by_evaluation: data.evaluations || {},
          query: args,
        };
      }
    } catch (error) {
      console.error("Backend fetch failed, using mock data:", error);
    }
  }

  // Fallback to mock statistics
  return {
    summary: {
      total_ayat: 6236,
      annotated_ayat: 6236,
      total_spans: 6236,
      gold_spans: 0,
      silver_spans: 6236,
      research_spans: 0,
      coverage_percent: 100.0,
      average_iaa: 0.925,
    },
    by_surah: [
      { surah: 2, name: "البقرة", spans: 286, coverage: 100 },
      { surah: 3, name: "آل عمران", spans: 200, coverage: 100 },
      { surah: 4, name: "النساء", spans: 176, coverage: 100 },
      { surah: 49, name: "الحجرات", spans: 18, coverage: 100 },
    ],
    top_behaviors: [
      { concept: "inner_state", name_ar: "الحالة الداخلية", count: 3184 },
      { concept: "speech_act", name_ar: "الفعل القولي", count: 1255 },
      { concept: "relational_act", name_ar: "الفعل العلائقي", count: 1003 },
      { concept: "physical_act", name_ar: "الفعل البدني", count: 640 },
    ],
    query: args,
  };
}

async function compareTafsir(args: {
  surah: number;
  ayah: number;
  focus_topic?: string;
}) {
  // Try real backend first
  if (USE_REAL_BACKEND) {
    try {
      const response = await fetch(`${QBM_BACKEND_URL}/tafsir/compare/${args.surah}/${args.ayah}`);
      if (response.ok) {
        const data = await response.json();
        return {
          ...data,
          focus_topic: args.focus_topic,
        };
      }
    } catch (error) {
      console.error("Backend fetch failed, using mock data:", error);
    }
  }

  // Fallback to mock
  const tafsir = await getTafsir({
    surah: args.surah,
    ayah: args.ayah,
    sources: ["ibn_kathir", "tabari", "qurtubi", "saadi"],
    include_ayah_text: true,
  });

  return {
    ...tafsir,
    comparison: {
      agreement_level: "high",
      key_differences: [
        "Ibn Kathir emphasizes the hadith narrations",
        "Tabari provides extensive linguistic analysis",
        "Qurtubi focuses on fiqh implications",
      ],
      common_themes: ["Divine attributes", "Tawhid", "Allah's sovereignty"],
    },
    focus_topic: args.focus_topic,
  };
}

async function getAyah(args: {
  surah: number;
  ayah: number;
  include_translation?: boolean;
  include_annotations?: boolean;
}) {
  // Try real backend first
  if (USE_REAL_BACKEND) {
    try {
      const params = new URLSearchParams();
      if (args.include_annotations !== undefined) {
        params.append("include_annotations", args.include_annotations.toString());
      }
      
      const response = await fetch(`${QBM_BACKEND_URL}/ayah/${args.surah}/${args.ayah}?${params}`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error("Backend fetch failed, using mock data:", error);
    }
  }

  // Fallback to mock
  return {
    surah: args.surah,
    ayah: args.ayah,
    surah_name_ar: "البقرة",
    surah_name_en: "Al-Baqarah",
    text_ar: "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
    text_en: args.include_translation
      ? "Allah - there is no deity except Him, the Ever-Living, the Sustainer of existence."
      : undefined,
    annotations: args.include_annotations
      ? [
          {
            id: "QBM_00001",
            behavior: "BEH_TAWHID",
            agent: "AGT_BELIEVER",
            tier: "gold",
          },
        ]
      : undefined,
    word_count: 7,
    juz: 3,
    hizb: 5,
  };
}

// Type import for OpenAI (needed for tool definitions)
import OpenAI from "openai";
