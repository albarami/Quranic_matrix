/**
 * QBM Tools for C1 Integration
 * 
 * These tools allow C1 to fetch real data from the QBM backend API
 * instead of hallucinating. C1 will call these tools and use the
 * results to generate rich, interactive UI components.
 */

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import type { JSONSchema } from "openai/lib/jsonschema.mjs";
import type { RunnableToolFunctionWithParse } from "openai/lib/RunnableFunction.mjs";

const QBM_BACKEND = process.env.QBM_BACKEND_URL || "http://localhost:8000";

// ============================================================================
// Tool Schemas
// ============================================================================

const searchSpansSchema = z.object({
  surah: z.number().min(1).max(114).optional().describe("Filter by surah number (1-114)"),
  agent: z.string().optional().describe("Filter by agent type: AGT_ALLAH, AGT_PROPHET, AGT_BELIEVER, AGT_DISBELIEVER, AGT_HUMAN_GENERAL, AGT_JINN, AGT_ANGEL, AGT_SHAYTAN"),
  behavior: z.string().optional().describe("Filter by behavior form: physical_act, speech_act, inner_state, cognitive_act, social_act"),
  evaluation: z.string().optional().describe("Filter by evaluation: praise, blame, neutral, warning, promise"),
  limit: z.number().min(1).max(50).default(20).describe("Maximum results to return (default 20)"),
});

const getSurahSchema = z.object({
  surah_number: z.number().min(1).max(114).describe("Surah number (1=Al-Fatiha, 2=Al-Baqarah, ... 114=An-Nas)"),
  limit: z.number().min(1).max(50).default(20).describe("Maximum annotations to return"),
});

const getStatisticsSchema = z.object({}).describe("No parameters needed - returns full project statistics");

const getTafsirSchema = z.object({
  surah: z.number().min(1).max(114).describe("Surah number"),
  ayah: z.number().min(1).describe("Ayah number within the surah"),
});

const compareTafsirSchema = z.object({
  surah: z.number().min(1).max(114).describe("Surah number"),
  ayah: z.number().min(1).describe("Ayah number within the surah"),
});

const getVocabulariesSchema = z.object({}).describe("No parameters needed - returns all controlled vocabularies");

const getAyahSchema = z.object({
  surah: z.number().min(1).max(114).describe("Surah number"),
  ayah: z.number().min(1).describe("Ayah number"),
  include_annotations: z.boolean().default(true).describe("Whether to include behavioral annotations"),
});

// ============================================================================
// Tool Implementations
// ============================================================================

async function searchSpans(params: z.infer<typeof searchSpansSchema>): Promise<string> {
  const queryParams = new URLSearchParams();
  if (params.surah) queryParams.set("surah", params.surah.toString());
  if (params.agent) queryParams.set("agent", params.agent);
  if (params.behavior) queryParams.set("behavior", params.behavior);
  if (params.evaluation) queryParams.set("evaluation", params.evaluation);
  queryParams.set("limit", (params.limit || 20).toString());

  try {
    const res = await fetch(`${QBM_BACKEND}/spans?${queryParams}`);
    const data = await res.json();
    return JSON.stringify(data);
  } catch (error) {
    return JSON.stringify({ error: "Failed to fetch spans from QBM backend" });
  }
}

async function getSurah(params: z.infer<typeof getSurahSchema>): Promise<string> {
  try {
    const res = await fetch(`${QBM_BACKEND}/surahs/${params.surah_number}?limit=${params.limit || 20}`);
    const data = await res.json();
    return JSON.stringify(data);
  } catch (error) {
    return JSON.stringify({ error: "Failed to fetch surah data" });
  }
}

async function getStatistics(): Promise<string> {
  try {
    const res = await fetch(`${QBM_BACKEND}/stats`);
    const data = await res.json();
    return JSON.stringify(data);
  } catch (error) {
    return JSON.stringify({ error: "Failed to fetch statistics" });
  }
}

async function getTafsir(params: z.infer<typeof getTafsirSchema>): Promise<string> {
  try {
    const res = await fetch(`${QBM_BACKEND}/tafsir/${params.surah}/${params.ayah}`);
    const data = await res.json();
    return JSON.stringify(data);
  } catch (error) {
    return JSON.stringify({ error: "Failed to fetch tafsir" });
  }
}

async function compareTafsir(params: z.infer<typeof compareTafsirSchema>): Promise<string> {
  try {
    const res = await fetch(`${QBM_BACKEND}/tafsir/compare/${params.surah}/${params.ayah}`);
    const data = await res.json();
    return JSON.stringify(data);
  } catch (error) {
    return JSON.stringify({ error: "Failed to compare tafsir" });
  }
}

async function getVocabularies(): Promise<string> {
  try {
    const res = await fetch(`${QBM_BACKEND}/vocabularies`);
    const data = await res.json();
    return JSON.stringify(data);
  } catch (error) {
    return JSON.stringify({ error: "Failed to fetch vocabularies" });
  }
}

async function getAyah(params: z.infer<typeof getAyahSchema>): Promise<string> {
  try {
    const res = await fetch(
      `${QBM_BACKEND}/ayah/${params.surah}/${params.ayah}?include_annotations=${params.include_annotations !== false}`
    );
    const data = await res.json();
    return JSON.stringify(data);
  } catch (error) {
    return JSON.stringify({ error: "Failed to fetch ayah" });
  }
}

// ============================================================================
// Export Tools Array for OpenAI runTools
// ============================================================================

export const qbmTools: RunnableToolFunctionWithParse<any>[] = [
  {
    type: "function",
    function: {
      name: "search_spans",
      description: `Search behavioral annotations in the Quranic Behavioral Matrix database.
      
Use this tool to find specific behaviors by:
- Surah number (1-114)
- Agent type (who performs the behavior): AGT_ALLAH, AGT_PROPHET, AGT_BELIEVER, AGT_DISBELIEVER, etc.
- Behavior form: physical_act, speech_act, inner_state, cognitive_act, social_act
- Evaluation: praise, blame, neutral, warning, promise

Returns annotated spans with:
- Arabic text (text_ar)
- Ayah reference (surah:ayah)
- Agent classification
- Behavior form
- Normative evaluation
- Deontic signal (amr/command, nahy/prohibition, etc.)`,
      parse: (input: string) => searchSpansSchema.parse(JSON.parse(input)),
      parameters: zodToJsonSchema(searchSpansSchema) as JSONSchema,
      function: searchSpans,
    },
  },
  {
    type: "function",
    function: {
      name: "get_surah_annotations",
      description: `Get all behavioral annotations for a specific surah.
      
Surah numbers: 1=Al-Fatiha, 2=Al-Baqarah, 3=Aal-Imran, ... 114=An-Nas

Returns all annotated spans in that surah with full details including Arabic text, agent types, behavior forms, and evaluations.`,
      parse: (input: string) => getSurahSchema.parse(JSON.parse(input)),
      parameters: zodToJsonSchema(getSurahSchema) as JSONSchema,
      function: getSurah,
    },
  },
  {
    type: "function",
    function: {
      name: "get_statistics",
      description: `Get comprehensive QBM project statistics.
      
Returns:
- Total spans (behavioral annotations)
- Unique surahs covered
- Unique ayat annotated
- Agent type distribution (counts per type)
- Behavior form distribution
- Evaluation distribution
- Deontic signal distribution

Use this for dashboards, progress reports, and overview visualizations.`,
      parse: (input: string) => getStatisticsSchema.parse(JSON.parse(input || "{}")),
      parameters: zodToJsonSchema(getStatisticsSchema) as JSONSchema,
      function: getStatistics,
    },
  },
  {
    type: "function",
    function: {
      name: "get_tafsir",
      description: `Get tafsir (Quranic exegesis/commentary) for a specific ayah.
      
Returns scholarly interpretations from classical tafsir sources like Ibn Kathir, Al-Tabari, Al-Qurtubi, etc.`,
      parse: (input: string) => getTafsirSchema.parse(JSON.parse(input)),
      parameters: zodToJsonSchema(getTafsirSchema) as JSONSchema,
      function: getTafsir,
    },
  },
  {
    type: "function",
    function: {
      name: "compare_tafsir",
      description: `Compare tafsir from multiple scholarly sources for a specific ayah.
      
Returns side-by-side comparison of interpretations from different classical scholars, highlighting similarities and differences in their understanding.`,
      parse: (input: string) => compareTafsirSchema.parse(JSON.parse(input)),
      parameters: zodToJsonSchema(compareTafsirSchema) as JSONSchema,
      function: compareTafsir,
    },
  },
  {
    type: "function",
    function: {
      name: "get_vocabularies",
      description: `Get all controlled vocabularies used in QBM annotations.
      
Returns lists of valid values for:
- Agent types (AGT_ALLAH, AGT_PROPHET, etc.)
- Behavior forms (physical_act, speech_act, etc.)
- Evaluations (praise, blame, neutral, etc.)
- Deontic signals (amr, nahy, ibaha, etc.)
- Speech modes
- Systemic categories

Use this to understand the classification system and provide accurate filter options.`,
      parse: (input: string) => getVocabulariesSchema.parse(JSON.parse(input || "{}")),
      parameters: zodToJsonSchema(getVocabulariesSchema) as JSONSchema,
      function: getVocabularies,
    },
  },
  {
    type: "function",
    function: {
      name: "get_ayah",
      description: `Get a specific ayah with its behavioral annotations.
      
Returns:
- Arabic text
- Reference (surah:ayah)
- All behavioral annotations for that ayah
- Agent, behavior form, evaluation for each annotation`,
      parse: (input: string) => getAyahSchema.parse(JSON.parse(input)),
      parameters: zodToJsonSchema(getAyahSchema) as JSONSchema,
      function: getAyah,
    },
  },
];

// Tool implementations map for manual execution
export const toolImplementations: Record<string, (...args: any[]) => Promise<string>> = {
  search_spans: searchSpans,
  get_surah_annotations: getSurah,
  get_statistics: getStatistics,
  get_tafsir: getTafsir,
  compare_tafsir: compareTafsir,
  get_vocabularies: getVocabularies,
  get_ayah: getAyah,
};
