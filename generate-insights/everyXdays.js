const pricingInfo = {
  // Prices are for Gemini 1.5/2.5 Pro, per 1 million tokens
  input: {
    standard_context_per_million_tokens: 1.25, // For prompts <= 200k tokens
    long_context_per_million_tokens: 2.5, // For prompts > 200k tokens
  },
  output: {
    standard_context_per_million_tokens: 10.0, // For prompts <= 200k tokens
    long_context_per_million_tokens: 15.0, // For prompts > 200k tokens
  },
  context_threshold: 200000, // The token threshold for long context pricing
  // We are ignoring context caching and grounding for this specific calculation
};

// ---- Payload minimizers for clip prompts ----
function sanitizeInsightForClips(insight) {
  try {
    const out = {
      title: String(insight?.title || ""),
      problem: String(insight?.problem || ""),
      solution: String(insight?.solution || ""),
    };
    /*   if (Array.isArray(insight?.evidence)) {
        out.evidence = insight.evidence
          .slice(0, 10)
          .map((e) => (typeof e === "string" ? e : String(e?.idVar ?? e?.analysisId ?? "")))
          .filter(Boolean);
      } */
    return out;
  } catch (_) {
    return { title: String(insight?.title || "") };
  }
}

function sanitizeSessionForClips(sess) {
  const maxChars = Number(process.env.CLIP_TRANSCRIPT_CHAR_LIMIT) || 20000;
  const rawTranscript = String(
    sess?.transcript || sess?.sessionTranscript || ""
  );
  let transcript = rawTranscript;
  if (rawTranscript.length > maxChars) {
    const half = Math.floor(maxChars / 2);
    transcript =
      rawTranscript.slice(0, half) + "\n...\n" + rawTranscript.slice(-half);
  }
  /* const lean = {
   
      transcript,
    }; */
  return transcript;
}

function calculateApiCallCost(usage = {}, prices = pricingInfo, options = {}) {
  const { suppressLog = false } = options || {};
  const resolvedUsage = usage || {};
  const inputPricing = prices?.input || {};
  const outputPricing = prices?.output || {};
  const longContextThreshold =
    typeof prices?.context_threshold === "number"
      ? prices.context_threshold
      : 200000;

  const result = {
    input: {
      tokens: 0,
      cost: 0,
      pricePerMillionTokens:
        typeof inputPricing.standard_context_per_million_tokens === "number"
          ? inputPricing.standard_context_per_million_tokens
          : 0,
      isLongContext: false,
      details: [],
    },
    output: {
      tokens: 0,
      cost: 0,
      pricePerMillionTokens:
        typeof outputPricing.standard_context_per_million_tokens === "number"
          ? outputPricing.standard_context_per_million_tokens
          : 0,
    },
    thoughts: {
      tokens: 0,
      billedWithinOutput: true,
      pricePerMillionTokens:
        typeof outputPricing.standard_context_per_million_tokens === "number"
          ? outputPricing.standard_context_per_million_tokens
          : 0,
      equivalentCost: 0,
    },
    totalCost: 0,
  };

  const promptTokensDetails = Array.isArray(resolvedUsage.promptTokensDetails)
    ? resolvedUsage.promptTokensDetails
    : null;

  if (promptTokensDetails) {
    for (const detail of promptTokensDetails) {
      const tokenCount =
        typeof detail?.tokenCount === "number" ? detail.tokenCount : 0;
      result.input.tokens += tokenCount;
      result.input.details.push({
        modality: detail?.modality || "unknown",
        tokens: tokenCount,
      });
    }
  } else if (typeof resolvedUsage.promptTokenCount === "number") {
    result.input.tokens = resolvedUsage.promptTokenCount;
  }

  // Account for explicit cache usage (discounted input cost for cached tokens)
  const cachedPromptTokensRaw =
    typeof resolvedUsage.cachedContentTokenCount === "number"
      ? resolvedUsage.cachedContentTokenCount
      : 0;
  const cachedPromptTokens = Math.max(
    0,
    Math.min(cachedPromptTokensRaw, result.input.tokens)
  );
  const nonCachedPromptTokens = Math.max(
    0,
    result.input.tokens - cachedPromptTokens
  );

  result.input.isLongContext = result.input.tokens > longContextThreshold;
  result.input.pricePerMillionTokens = result.input.isLongContext
    ? typeof inputPricing.long_context_per_million_tokens === "number"
      ? inputPricing.long_context_per_million_tokens
      : result.input.pricePerMillionTokens
    : result.input.pricePerMillionTokens;
  // 10% of input price for cached tokens
  const cachedDiscount = 0.1;
  const nonCachedCost =
    (nonCachedPromptTokens / 1_000_000) * result.input.pricePerMillionTokens;
  const cachedCost =
    (cachedPromptTokens / 1_000_000) *
    result.input.pricePerMillionTokens *
    cachedDiscount;
  result.input.cost = nonCachedCost + cachedCost;
  if (!Number.isFinite(result.input.cost)) {
    result.input.cost = 0;
  }
  result.totalCost += result.input.cost;
  if (cachedPromptTokens > 0) {
    result.input.details.push({
      modality: "cachedPrompt",
      tokens: cachedPromptTokens,
    });
  }

  result.output.tokens =
    typeof resolvedUsage.candidatesTokenCount === "number"
      ? resolvedUsage.candidatesTokenCount
      : 0;
  result.output.pricePerMillionTokens = result.input.isLongContext
    ? typeof outputPricing.long_context_per_million_tokens === "number"
      ? outputPricing.long_context_per_million_tokens
      : result.output.pricePerMillionTokens
    : result.output.pricePerMillionTokens;
  result.output.cost =
    (result.output.tokens / 1_000_000) * result.output.pricePerMillionTokens;
  if (!Number.isFinite(result.output.cost)) {
    result.output.cost = 0;
  }
  result.totalCost += result.output.cost;

  result.thoughts.tokens =
    typeof resolvedUsage.thoughtsTokenCount === "number"
      ? resolvedUsage.thoughtsTokenCount
      : 0;
  result.thoughts.pricePerMillionTokens = result.output.pricePerMillionTokens;
  result.thoughts.equivalentCost =
    (result.thoughts.tokens / 1_000_000) *
    result.thoughts.pricePerMillionTokens;
  if (!Number.isFinite(result.thoughts.equivalentCost)) {
    result.thoughts.equivalentCost = 0;
  }

  if (!suppressLog) {
    console.log(
      `Input is ${result.input.isLongContext ? "long" : "standard"} context.`
    );
    if (result.input.details.length > 0) {
      console.log("--- Input Token Breakdown ---");
      for (const detail of result.input.details) {
        console.log(`  - ${detail.modality}: ${detail.tokens} tokens`);
      }
      console.log("---------------------------");
    }
    if (result.output.tokens > 0) {
      console.log(
        `Output Cost: $${result.output.cost.toFixed(7)} for ${
          result.output.tokens
        } tokens`
      );
    }
    if (result.thoughts.tokens > 0) {
      console.log(
        `(Info) Output includes ${result.thoughts.tokens} 'thinking' tokens, which are billed at the standard output rate.`
      );
    }
    console.log(`Total API Call Cost: $${result.totalCost.toFixed(7)}`);
  }

  return result;
}

// ---------------- Session Cost Tracking ----------------

// Tracks costs per active sessionId
const sessionCostTracker = new Map();

function generateCostSessionId(userId) {
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  return `${String(userId ?? "unknown")}-${ts}`;
}

function initializeSessionCostTracking(sessionId, userId) {
  const id = String(sessionId || generateCostSessionId(userId));
  if (!sessionCostTracker.has(id)) {
    sessionCostTracker.set(id, {
      userId: userId ?? null,
      sessionId: id,
      startedAt: new Date().toISOString(),
      totals: {
        inputTokens: 0,
        outputTokens: 0,
        thoughtTokens: 0,
        inputCost: 0,
        outputCost: 0,
        totalCost: 0,
      },
      prompts: [],
    });
  }
  return id;
}

function recordSessionPromptCost({
  sessionId,
  promptLabel = "unnamed-prompt",
  usageMetadata = {},
  requestConfig = {},
  extra = {},
}) {
  const id = String(sessionId || "");
  if (!id || !sessionCostTracker.has(id)) return;

  const summary = calculateApiCallCost(usageMetadata, pricingInfo, {
    suppressLog: true,
  });

  const entry = sessionCostTracker.get(id);
  entry.totals.inputTokens += Number(summary?.input?.tokens || 0);
  entry.totals.outputTokens += Number(summary?.output?.tokens || 0);
  entry.totals.thoughtTokens += Number(summary?.thoughts?.tokens || 0);
  entry.totals.inputCost += Number(summary?.input?.cost || 0);
  entry.totals.outputCost += Number(summary?.output?.cost || 0);
  entry.totals.totalCost += Number(summary?.totalCost || 0);

  const model = String(requestConfig?.model || "");
  const config = requestConfig?.config || {};
  const record = {
    at: new Date().toISOString(),
    label: String(promptLabel || "unnamed-prompt"),
    model,
    config: {
      temperature: config?.temperature,
      top_p: config?.top_p,
      responseMimeType: config?.responseMimeType,
    },
    usage: {
      inputTokens: summary?.input?.tokens || 0,
      outputTokens: summary?.output?.tokens || 0,
      thoughtTokens: summary?.thoughts?.tokens || 0,
      inputIsLongContext: !!summary?.input?.isLongContext,
      inputBreakdown: Array.isArray(summary?.input?.details)
        ? summary.input.details
        : [],
    },
    cost: {
      input: summary?.input?.cost || 0,
      output: summary?.output?.cost || 0,
      total: summary?.totalCost || 0,
      pricesPerMillion: {
        input: summary?.input?.pricePerMillionTokens || 0,
        output: summary?.output?.pricePerMillionTokens || 0,
      },
      thoughtsEquivalent: summary?.thoughts?.equivalentCost || 0,
    },
    extra: extra || {},
  };

  entry.prompts.push(record);
}

async function writeSessionCostReport(sessionId) {
  const id = String(sessionId || "");
  const data = id ? sessionCostTracker.get(id) : undefined;
  if (!data) return "";

  const lines = [];
  lines.push(`Session Cost Report`);
  lines.push(`Session: ${data.sessionId}`);
  lines.push(`User ID: ${data.userId ?? "unknown"}`);
  lines.push(`Started: ${data.startedAt}`);
  lines.push(`Ended: ${new Date().toISOString()}`);
  lines.push("");
  lines.push(`Totals:`);
  lines.push(
    `  Input Tokens: ${data.totals.inputTokens} | Output Tokens: ${data.totals.outputTokens} | Thought Tokens: ${data.totals.thoughtTokens}`
  );
  lines.push(
    `  Input Cost: $${data.totals.inputCost.toFixed(
      7
    )} | Output Cost: $${data.totals.outputCost.toFixed(
      7
    )} | Total Cost: $${data.totals.totalCost.toFixed(7)}`
  );
  lines.push("");
  lines.push(`Prompts:`);
  for (const [i, p] of data.prompts.entries()) {
    lines.push(
      `#${i + 1} ${p.at} | ${p.label} | ${p.model || "model-unknown"}`
    );
    const inBD = Array.isArray(p?.usage?.inputBreakdown)
      ? p.usage.inputBreakdown
      : [];
    lines.push(
      `   tokens: in=${p.usage.inputTokens} (long=${p.usage.inputIsLongContext}) out=${p.usage.outputTokens} thoughts=${p.usage.thoughtTokens}`
    );
    if (inBD.length > 0) {
      for (const d of inBD) {
        lines.push(`     - input ${d.modality}: ${d.tokens}`);
      }
    }
    lines.push(
      `   rates: in=$${Number(p?.cost?.pricesPerMillion?.input || 0).toFixed(
        4
      )}/1M out=$${Number(p?.cost?.pricesPerMillion?.output || 0).toFixed(
        4
      )}/1M`
    );
    lines.push(
      `   cost: in=$${Number(p?.cost?.input || 0).toFixed(7)} out=$${Number(
        p?.cost?.output || 0
      ).toFixed(7)} thoughts~=$${Number(
        p?.cost?.thoughtsEquivalent || 0
      ).toFixed(7)} total=$${Number(p?.cost?.total || 0).toFixed(7)}`
    );
  }
  lines.push("");
  return lines.join("\n");
}

async function appendSessionCostReportToBucket(userId, sessionId, reportText) {
  try {
    const safeUser = String(userId ?? "unknown");
    const logName = `cost-logs-new/${safeUser}.log`;
    const file = bucket.file(logName);
    let existing = "";
    try {
      const [buf] = await file.download();
      existing = buf?.toString("utf8") || "";
    } catch (_) {
      existing = "";
    }
    const header = `\n\n===== Cost Session ${sessionId} @ ${new Date()
      .toISOString()
      .replace(/T/, " ")} =====\n`;
    const content = `${existing}${header}${reportText}\n`;
    await file.save(Buffer.from(content, "utf8"), {
      contentType: "text/plain; charset=utf-8",
      resumable: false,
    });
    try {
      await file.makePublic();
    } catch (_) {
      /* noop */
    }
    return true;
  } catch (e) {
    console.warn("Failed to append session cost report to bucket", e);
    return false;
  }
}

import dotenv from "dotenv";
import { GoogleGenAI, createUserContent, Type } from "@google/genai";
import * as fs from "fs/promises";
import fs2 from "fs";
import path from "path";
import * as os from "os";
import { Pool } from "pg";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { Storage } from "@google-cloud/storage";
import { spawn } from "child_process";
import crypto from "crypto";
import { FinishReason } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const Gemini = "models/gemini-3-pro-preview";

dotenv.config();
// Set the application timezone to UTC to ensure all date operations are consistent.
process.env.TZ = "UTC";

const storage = new Storage();
const bucketName = process.env.GCS_BUCKET_NAME;

const bucket = storage.bucket(bucketName);

// API Key Configuration - Add your 3 API keys here
const apiKeys = [process.env.GEMINI_API_KEY_0];

const ai = new GoogleGenAI({
  apiKey: apiKeys[0],
});

// ---- Explicit Prompt Caching (Gemini) ----
const promptCacheLocal = new Map(); // hash -> cachedContent name
function hashPrompt(prompt) {
  try {
    return crypto
      .createHash("sha256")
      .update(String(prompt || ""), "utf8")
      .digest("hex");
  } catch (_) {
    // fallback simple hash
    let h = 0;
    const s = String(prompt || "");
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    return String(h >>> 0);
  }
}
async function ensurePromptCache(aiClient, model, prompt, tracking = {}) {
  const hash = hashPrompt(prompt);
  if (promptCacheLocal.has(hash)) return promptCacheLocal.get(hash);
  const ttlEnv = process.env.GEMINI_CACHE_TTL || "1000s"; //
  const ttl = /s$/.test(ttlEnv) ? ttlEnv : `${ttlEnv}s`;
  const displayName = `prompt-${hash.slice(0, 16)}`;
  const cached = await aiClient.caches.create({
    model,
    config: {
      displayName,
      ttl,
      contents: [createUserContent(String(prompt || ""))],
    },
  });
  const name = cached?.name || "";
  if (name) promptCacheLocal.set(hash, name);
  // Record cache creation cost
  try {
    if (tracking?.sessionId) {
      const totalTokens = Number(cached?.usageMetadata?.totalTokenCount || 0);
      recordSessionPromptCost({
        sessionId: tracking.sessionId,
        promptLabel: tracking.promptLabel || "cache:create",
        usageMetadata: {
          promptTokenCount: totalTokens,
          candidatesTokenCount: 0,
          thoughtsTokenCount: 0,
        },
        requestConfig: { model },
        extra: { cacheName: name, displayName },
      });
    }
  } catch (_) {}
  return name;
}

// --- Database Configuration ---
const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_DATABASE,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,

});

// an example to work with the database
/* const insertQuery = `
  INSERT INTO sessionanalysis (posthogrecordingid, userid, status)
  VALUES ($1, $2, 'PROCESSING')
  RETURNING id;
  `;
  const insertValues = [session.id, user.id];
  const insertResult = await pool.query(insertQuery, insertValues); */

function resolveDaysWindowFromEnv() {
  const toNumber = (v) => {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
  };

  const truthy = (v) =>
    typeof v === "string" ? /^(1|true|yes|on)$/i.test(v) : !!v;

  // Preferred explicit numeric envs
  const daysBack = toNumber(process.env.DAYS_BACK);
  if (daysBack) return daysBack;
  const daysWindow = toNumber(process.env.DAYS_WINDOW);
  if (daysWindow) return daysWindow;

  // Pattern like "9_DAYS=true" → 9
  try {
    const candidates = Object.keys(process.env)
      .map((k) => {
        const m = k.match(/^(\d+)_DAYS$/i);
        if (m && truthy(process.env[k])) return Number(m[1]);
        return null;
      })
      .filter((n) => Number.isFinite(n) && n > 0);
    if (candidates.length > 0)
      return Math.max(...candidates.map((n) => Math.floor(n)));
  } catch (_) {}

  // Default window length used previously: 3 days (base day + 2 previous)
  return 3;
}

export async function getSessionAnalysesByEmail(email, limit = null) {
  try {
    const { rows: userRows } = await pool.query(
      `
          SELECT id
          FROM user2
          WHERE email = $1
          LIMIT 1;
        `,
      [email]
    );

    const user = userRows[0];

    if (!user) {
      return { sessions: [], formatted: "", totalCount: 0, userId: null };
    }

    // Compute the target day and the two previous days (UTC)
    const toYMD = (d) => {
      const pad = (n) => String(n).padStart(2, "0");
      return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(
        d.getUTCDate()
      )}`;
    };
    const particularDay = process.env.PARTICULAR_DAY;
    const runToday =
      String(process.env.RUN_TODAY || "").trim().toLowerCase() === "true";

    const baseDate = (() => {
      if (runToday) {
        // Normalize to current day in UTC to match how dates are stored/queried.
        const now = new Date();
        return new Date(
          Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
        );
      }

      if (!particularDay) {
        throw new Error(
          "PARTICULAR_DAY environment variable is required when RUN_TODAY is not true"
        );
      }

      const parsed = new Date(`${particularDay}`);
      if (Number.isNaN(parsed.getTime())) {
        throw new Error(`Invalid PARTICULAR_DAY format: ${particularDay}`);
      }
      return parsed;
    })();
    const windowDays = Math.max(1, Number(resolveDaysWindowFromEnv()) || 3);
    const startDate = new Date(baseDate);
    startDate.setUTCDate(startDate.getUTCDate() - (windowDays - 1));
    const startDay = toYMD(startDate);
    const endDay = toYMD(new Date(baseDate));

    let analysesQuery = `
        SELECT
          id,
          userid AS "userId",
          analysiscontent AS "analysisContent",
          createdat AS "createdAt"
        FROM sessionanalysis
        WHERE userid = $1
          AND processedat BETWEEN $2 AND $3
        ORDER BY createdat DESC
      `;

    const queryParams = [user.id, startDay, endDay];
    const limitValue = Number(limit);

    if (Number.isFinite(limitValue) && limitValue > 0) {
      analysesQuery += "\n      LIMIT $4";
      queryParams.push(Math.floor(limitValue));
    }

    const { rows: analyses } = await pool.query(analysesQuery, queryParams);

    const formattedAnalyses = [];

    console.log("analysefwefewfwefwes ", analyses.length);

    /*   return; */
    let firstIndex = 0;
    let secondIndex = 0;

    for (const analysis of analyses) {
      const content = analysis?.analysisContent;
      if (!content) continue;

      firstIndex += 1;

      const [rawMetadata, transcriptPart] = content.split("FIRST_PART");

      if (!rawMetadata || !transcriptPart) continue;

      secondIndex += 1;

      const transcript = transcriptPart.includes("SECOND_PART")
        ? transcriptPart.split("SECOND_PART")[0] ?? ""
        : transcriptPart;

      try {
        const parsedMetadata = JSON.parse(rawMetadata);

        const [
          startUrl,
          recordingDuration,
          clickCount,
          end_time,
          datetime,
          personId,
          browser,
          osVar,
          osVersion,
          deviceType,
          country,
          referrer,
          idVar,
          user_agent,
          numberOfPages,
          lastPage,
          allPagesRaw,
        ] = parsedMetadata;

        const parsedPages = Array.isArray(allPagesRaw)
          ? allPagesRaw
          : typeof allPagesRaw === "string" && allPagesRaw.length > 0
          ? JSON.parse(allPagesRaw)
          : [];

        formattedAnalyses.push({
          analysisId: analysis.id,
          startUrl,
          recording_duration: recordingDuration,
          click_count: clickCount,
          end_time,
          datetime,
          person_id: personId,
          browser,
          osVar,
          osVersion,
          device_type: deviceType,
          country,
          transcript,
          referrer,
          idVar,
          user_agent,
          numberOfPages: Number(numberOfPages) || 0,
          lastPage,
          allPages: parsedPages,
        });
      } catch (parseError) {
        console.warn("Failed to parse session analysis content for user", {
          email,
          analysisId: analysis.id,
          parseError,
        });
      }
    }
    const dedupedSessions = dedupeByIdVarKeepLatest(formattedAnalyses);
    console.log("dedupedSessionsdedupedSessions ", dedupedSessions.length);
    return {
      sessions: dedupedSessions,
      formatted: formatSessionsOutput(dedupedSessions),
      totalCount: dedupedSessions.length,
      userId: user.id,
    };
  } catch (error) {
    console.error("Failed to fetch session analyses by email", {
      email,
      error,
    });
  }
}

function parseDateToMs(value) {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function dedupeByIdVarKeepLatest(sessions) {
  const latestByIdVar = new Map();
  let sessionsreversed = sessions.reverse();
  for (const session of sessionsreversed) {
    const key = session.idVar ?? "";
    if (!key) continue;

    const existing = latestByIdVar.get(key);
    if (!existing) {
      latestByIdVar.set(key, session);
      continue;
    }

    const existingMs = parseDateToMs(existing.datetime);
    const candidateMs = parseDateToMs(session.datetime);

    if (candidateMs >= existingMs) {
      latestByIdVar.set(key, session);
    }
  }

  const deduped = Array.from(latestByIdVar.values());
  deduped.sort((a, b) => parseDateToMs(b.datetime) - parseDateToMs(a.datetime));
  return deduped;
}

export function formatSessionsOutput(sessions) {
  const sessionEntries = sessions.map((recording, index) => {
    return `<user_session id="${recording.idVar}">
  <info>
  Datetime (UTC): ${formatDateTime(recording.datetime) || "N/A"}
  Duration: ${formatDuration(recording.recording_duration) || "N/A"}
  Country: ${recording.country || "N/A"}
  Entry: ${recording.startUrl || "N/A"}
  Exit: ${recording.lastPage || "N/A"}
  Pages: ${recording.numberOfPages || 1}
  Clicks: ${recording.click_count || 0}
  User-Agent: ${recording?.user_agent || "N/A"}
  Referrer: ${recording.referrer?.replace("$", "") || "N/A"}
  UserID: ${recording.person_id || "N/A"}
  </info>
  
  <user_journey>
  ${recording.transcript || "No transcript available"}
  </user_journey>
  </user_session>`;
  });

  if (sessionEntries.length === 0) {
    return "<user_sessions></user_sessions>";
  }

  return `<user_sessions>
  ${sessionEntries.join("\n")}
  </user_sessions>`;
}

function formatDateTime(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  const pad = (num) => num.toString().padStart(2, "0");

  const year = date.getUTCFullYear();
  const month = pad(date.getUTCMonth() + 1);
  const day = pad(date.getUTCDate());
  const hours = pad(date.getUTCHours());
  const minutes = pad(date.getUTCMinutes());
  const seconds = pad(date.getUTCSeconds());

  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

function formatDuration(value) {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "0s";
  }

  const totalSeconds = Math.round(numeric);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  const parts = [];
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds}s`);

  return parts.join(" ");
}

// Helper function to retry content generation with intelligent backoff and API key rotation
async function generateContentWithRetry(
  ai,
  requestConfig,
  retriesPerKey = 1,
  apiKeys = null,
  trackingOptions = {}
) {
  const {
    sessionId = null,
    promptLabel = "unnamed-prompt",
    promptContext = {},
  } = trackingOptions || {};
  const useKeyRotation = Array.isArray(apiKeys) && apiKeys.length > 0;
  const keysToTry = useKeyRotation ? apiKeys : [null]; // Use a single null for the non-rotation case

  if (useKeyRotation) {
    console.log(
      `đź”‘ API key rotation enabled with ${keysToTry.length} keys available.`
    );
  }

  let lastError = null;
  let totalAttempts = 0;
  const maxTotalAttempts = 3; // Total attempts across all keys

  // Cycle through keys until we've tried 8 times total
  while (totalAttempts < maxTotalAttempts) {
    for (let keyIndex = 0; keyIndex < keysToTry.length; keyIndex++) {
      const currentKey = keysToTry[keyIndex];
      let currentAi = ai;

      if (useKeyRotation && currentKey) {
        console.log(
          `đź”‘ Using API key ${keyIndex + 1}/${keysToTry.length} (attempt ${
            totalAttempts + 1
          }/${maxTotalAttempts})`
        );
        currentAi = new GoogleGenAI({ apiKey: currentKey });
      }

      for (let attempt = 1; attempt <= retriesPerKey; attempt++) {
        totalAttempts++;

        if (totalAttempts > maxTotalAttempts) {
          console.error(
            `âťŚ Maximum total attempts (${maxTotalAttempts}) reached. All API keys and retries failed.`
          );
          throw lastError || new Error("All retry attempts failed.");
        }

        try {
          // Calculate request size for debugging
          const requestSize = JSON.stringify(requestConfig).length;
          console.log(
            `Attempting to generate content (key ${
              keyIndex + 1
            }, attempt ${attempt}/${retriesPerKey}, total attempt ${totalAttempts}/${maxTotalAttempts})...
    Request size: ${requestSize} characters`
          );
          const completion = await currentAi.models.generateContent(
            requestConfig
          );

          // Check if candidatesTokenCount is present in the response
          const hasValidTokenCount =
            completion?.usageMetadata?.candidatesTokenCount !== undefined;

          if (!hasValidTokenCount) {
            console.warn(
              `âš ď¸Ź Missing candidatesTokenCount in response (key ${
                keyIndex + 1
              }, attempt ${attempt}). Retrying...`
            );
            throw new Error(
              "Missing candidatesTokenCount in response - treating as retryable error"
            );
          }

          console.log(
            `âś… Content generation successful on key ${
              keyIndex + 1
            }, attempt ${attempt} (total attempt ${totalAttempts})`
          );

          if (sessionId) {
            recordSessionPromptCost({
              sessionId,
              promptLabel,
              usageMetadata: completion?.usageMetadata || null,
              requestConfig,
              extra: promptContext,
            });
          }
          return completion;
        } catch (error) {
          lastError = error;
          console.error(
            `Content generation attempt ${attempt} for key ${
              keyIndex + 1
            } failed (total attempt ${totalAttempts}): ${
              error.message
            }, error: ${error}`
          );

          const isRetryable =
            (error.status >= 500 && error.status <= 599) ||
            error.message.includes("ECONNRESET") ||
            error.message.includes("ETIMEDOUT") ||
            error.message.includes("ENOTFOUND") ||
            error.message.includes("ECONNREFUSED") ||
            error.message.includes("fetch failed") ||
            error.message.includes("network") ||
            error.message.includes("connection") ||
            error.message.includes("socket") ||
            error.message.includes("request timeout") ||
            error.message.includes("DNS lookup failed") ||
            error.message.includes("Missing candidatesTokenCount") ||
            error.message.includes("CONSUMER_SUSPENDED");

          const isQuotaExhaustion =
            error.status === 429 ||
            error.message.includes("RESOURCE_EXHAUSTED") ||
            error.message.includes("quota") ||
            error.message.includes("rate limit");

          if (isQuotaExhaustion) {
            const suggestedWaitMs = getRetryDelayMsFromError(error);
            if (suggestedWaitMs !== null && suggestedWaitMs < 15000) {
              console.log(
                `âŹł Quota/rate limit encountered. Waiting ${suggestedWaitMs} ms before retrying same key...`
              );
              await new Promise((resolve) =>
                setTimeout(resolve, suggestedWaitMs)
              );
              // retry same key (continue inner loop)
              continue;
            }
            console.log(
              `đź”‘ API key ${
                keyIndex + 1
              } hit quota limit (no short retry delay). Moving to next key.`
            );
            break; // Exit inner loop and move to the next key
          }

          if (!isRetryable) {
            console.error("âťŚ Non-retryable error encountered. Aborting.");
            throw error; // For non-retryable errors, fail fast
          }

          if (attempt >= retriesPerKey) {
            console.warn(
              `âš ď¸Ź  Max retries (${retriesPerKey}) reached for key ${
                keyIndex + 1
              }.`
            );
            break; // Exit inner loop to try next key
          }

          const backoff = Math.pow(2, attempt) * 1000;
          const jitter = Math.random() * 1000;
          const waitTime = backoff + jitter;

          console.log(`Retrying in ${(waitTime / 1000).toFixed(2)}s...`);
          await new Promise((resolve) => setTimeout(resolve, waitTime));
        }
      }
    }
  }

  console.error(
    `âťŚ All API keys and retries failed. Total attempts: ${totalAttempts}.`
  );
  throw lastError || new Error("All retry attempts failed.");
}

// ---- Evidence URL replacement with GCS clip URLs ----
const CLIPS_ROOT_PREFIX = "clipsssss114s0ewfw001entirevideosAAAAAAa99000";
const CLIPS_ROOT_PREFIX2 = "clipsssss114s0ewfw001entirevideosAAAAAAa";
const CLIPS_ROOT_PREFIX3 = "clipsssss114s0ewfw001entirevideosAAAAAAaaabb";
const CLIPS_ROOT_PREFIX4 = "clickclipsssss114s0ewfw001entirevideosAAAAAAa";

function parseEvidenceUrl(evidenceUrl) {
  try {
    const url = new URL(evidenceUrl);
    if (
      !url.hostname.includes("posthog.com") ||
      !url.pathname.includes("/replay/")
    ) {
      return null;
    }
    const pathParts = url.pathname.split("/").filter(Boolean);
    const replayIndex = pathParts.indexOf("replay");
    const sessionId = replayIndex >= 0 ? pathParts[replayIndex + 1] : null;
    const tParam = url.searchParams.get("t");
    const eventSeconds = tParam ? Number(tParam) : NaN;
    if (!sessionId || !Number.isFinite(eventSeconds)) return null;
    return { sessionId, eventSeconds };
  } catch (_) {
    return null;
  }

  // ---- Payload minimizers for clip prompts ----
  function sanitizeInsightForClips(insight) {
    try {
      const out = {
        title: String(insight?.title || ""),
        problem: String(insight?.problem || ""),
        solution: String(insight?.solution || ""),
      };
      if (insight?.priority) out.priority = String(insight.priority);
      if (Array.isArray(insight?.evidence)) {
        out.evidence = insight.evidence
          .slice(0, 10)
          .map((e) =>
            typeof e === "string" ? e : String(e?.idVar ?? e?.analysisId ?? "")
          )
          .filter(Boolean);
      }
      return out;
    } catch (_) {
      return { title: String(insight?.title || "") };
    }
  }

  function sanitizeSessionForClips(sess) {
    const maxChars = Number(process.env.CLIP_TRANSCRIPT_CHAR_LIMIT) || 20000;
    const rawTranscript = String(
      sess?.transcript || sess?.sessionTranscript || ""
    );
    let transcript = rawTranscript;
    if (rawTranscript.length > maxChars) {
      const half = Math.floor(maxChars / 2);
      transcript =
        rawTranscript.slice(0, half) + "\n...\n" + rawTranscript.slice(-half);
    }
    const lean = {
      idVar: sess?.idVar ?? sess?.id ?? null,
      analysisId: sess?.analysisId ?? null,
      datetime: sess?.datetime ?? null,
      startUrl: sess?.startUrl ?? sess?.entry ?? null,
      lastPage: sess?.lastPage ?? null,
      numberOfPages: sess?.numberOfPages ?? null,
      click_count: sess?.click_count ?? null,
      recording_duration: sess?.recording_duration ?? null,
      country: sess?.country ?? null,
      transcript,
    };
    return lean;
  }
}

function parseClipWindowFromName(name) {
  // Expect patterns like: .../click_0_from_3.667s_to_12.084s_processed.mp4 or .../noclick_1_from_0.000s_to_3.667s_raw.mp4
  const match = name.match(/from_(\d+(?:\.\d+)?)s_to_(\d+(?:\.\d+)?)s/i);
  if (!match) return null;
  const start = Number(match[1]);
  const end = Number(match[2]);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return { start, end };
}

function scoreClipName(name) {
  const lower = name.toLowerCase();
  const isClick = lower.includes("/click_") || lower.startsWith("click_");
  const isProcessed = lower.includes("processed");
  // Prefer click over noclick, processed over raw
  let score = 0;
  if (isClick) score += 2;
  else score += 0;
  if (isProcessed) score += 1;
  return score;
}

async function findClipPublicUrlForEvent(
  sessionId,
  eventSeconds,
  sessionFilesCache
) {
  // Cache files listing per session to avoid repeated list calls
  if (!sessionFilesCache.has(sessionId)) {
    const prefix = `${CLIPS_ROOT_PREFIX}/${sessionId}/${CLIPS_ROOT_PREFIX}/`;
    const prefix2 = `${CLIPS_ROOT_PREFIX2}/${sessionId}/${CLIPS_ROOT_PREFIX2}/`;
    const prefix3 = `${CLIPS_ROOT_PREFIX3}/${sessionId}/${CLIPS_ROOT_PREFIX3}/`;
    const prefix4 = `${CLIPS_ROOT_PREFIX4}/${sessionId}/${CLIPS_ROOT_PREFIX4}/`;

    console.log("prefix2prefix2 ", prefix2, prefix, prefix3);
    try {
      const [files] = await bucket.getFiles({ prefix });
      const [files2] = await bucket.getFiles({ prefix: prefix2 });
      const [files3] = await bucket.getFiles({ prefix: prefix3 });
      const [files4] = await bucket.getFiles({ prefix: prefix4 });
      console.log("fefjwenfw ", files, files2);
      sessionFilesCache.set(
        sessionId,
        [...files, ...files2, ...files3, ...files4] || []
      );
    } catch (e) {
      console.warn(`Failed listing files for session ${sessionId}`, e);
      sessionFilesCache.set(sessionId, []);
    }
  }

  const files = sessionFilesCache.get(sessionId) || [];
  const candidates = [];
  for (const file of files) {
    const name = file.name || "";
    if (!name.toLowerCase().endsWith(".mp4")) continue;
    const win = parseClipWindowFromName(name);
    if (!win) continue;
    if (eventSeconds >= win.start && eventSeconds <= win.end) {
      candidates.push({ file, name, win, score: scoreClipName(name) });
    }
  }

  if (candidates.length === 0) return null;

  // Choose best by score; tie-breaker: narrower window (end-start), then earliest start
  candidates.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    const aw = a.win.end - a.win.start;
    const bw = b.win.end - b.win.start;
    if (aw !== bw) return aw - bw;
    return a.win.start - b.win.start;
  });

  const chosen = candidates[0];
  try {
    // Ensure public access; ignore errors if already public or uniform bucket-level policy
    if (typeof chosen.file.makePublic === "function") {
      try {
        await chosen.file.makePublic();
      } catch (_) {
        /* noop */
      }
    }
  } catch (_) {
    /* noop */
  }

  if (typeof chosen.file.publicUrl === "function") {
    return chosen.file.publicUrl();
  }
  // Fallback manual URL
  const encodedName = chosen.name.split("/").map(encodeURIComponent).join("/");
  return `https://storage.googleapis.com/${bucketName}/${encodedName}`;
}

// Convert "mm:ss.sss" or seconds string to number of seconds
function timeToSecondsStrict(t) {
  if (typeof t !== "string") return NaN;
  const parts = t.split(":");
  if (parts.length === 2) {
    const m = Number(parts[0]);
    const s = Number(parts[1]);
    if (Number.isFinite(m) && Number.isFinite(s)) return m * 60 + s;
  }
  const n = Number(t);
  return Number.isFinite(n) ? n : NaN;
}

// Format seconds to m:ss.sss
function secondsToTimeStr(totalSeconds) {
  const s = Math.max(0, Number(totalSeconds) || 0);
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  const sec = rem.toFixed(3);
  const [intPart, fracPart] = sec.split(".");
  const secPadded = intPart.padStart(2, "0") + "." + (fracPart || "000");
  return `${m}:${secPadded}`;
}

// Find a set of clip URLs that cover [startSeconds, endSeconds]
async function findClipPublicUrlsForRange(
  sessionId,
  startSeconds,
  endSeconds,
  sessionFilesCache
) {
  const startReq = Number(startSeconds);
  const endReq = Number(endSeconds);
  if (
    !Number.isFinite(startReq) ||
    !Number.isFinite(endReq) ||
    endReq <= startReq
  ) {
    return [];
  }

  // Ensure files cached
  if (!sessionFilesCache.has(sessionId)) {
    const prefix = `${CLIPS_ROOT_PREFIX}/${sessionId}/${CLIPS_ROOT_PREFIX}/`;
    const prefix2 = `${CLIPS_ROOT_PREFIX2}/${sessionId}/${CLIPS_ROOT_PREFIX2}/`;
    const prefix3 = `${CLIPS_ROOT_PREFIX3}/${sessionId}/${CLIPS_ROOT_PREFIX3}/`;
    const prefix4 = `${CLIPS_ROOT_PREFIX4}/${sessionId}/${CLIPS_ROOT_PREFIX4}/`;
    console.log("prefix2prefix2 ", prefix2, prefix, prefix3);

    try {
      const [files] = await bucket.getFiles({ prefix });
      const [files2] = await bucket.getFiles({ prefix: prefix2 });
      const [files3] = await bucket.getFiles({ prefix: prefix3 });
      const [files4] = await bucket.getFiles({ prefix: prefix4 });
      console.log("fefjwenfw ", files, files2);
      sessionFilesCache.set(
        sessionId,
        [...files, ...files2, ...files3, ...files4] || []
      );
    } catch (e) {
      console.warn(`Failed listing files for session ${sessionId}`, e);
      sessionFilesCache.set(sessionId, []);
    }
  }

  const files = sessionFilesCache.get(sessionId) || [];
  // Build window candidates overlapping the requested range
  const candidates = [];
  for (const file of files) {
    const name = file.name || "";
    if (!name.toLowerCase().endsWith(".mp4")) continue;
    const win = parseClipWindowFromName(name);
    if (!win) continue;
    if (win.end < startReq || win.start > endReq) continue; // no overlap
    candidates.push({ file, name, win, score: scoreClipName(name) });
  }

  if (candidates.length === 0) return [];

  // Sort by start asc, end desc, score desc
  candidates.sort((a, b) => {
    if (a.win.start !== b.win.start) return a.win.start - b.win.start;
    if (a.win.end !== b.win.end) return b.win.end - a.win.end;
    return b.score - a.score;
  });

  const result = [];
  const tol = 1e-3;
  let current = startReq;
  let idx = 0;
  while (current < endReq - tol) {
    // among candidates with start <= current + tol, pick one with farthest end (tie: higher score)
    let best = null;
    for (let i = idx; i < candidates.length; i++) {
      const c = candidates[i];
      if (c.win.start <= current + tol) {
        if (
          !best ||
          c.win.end > best.win.end ||
          (Math.abs(c.win.end - best.win.end) <= tol && c.score > best.score)
        ) {
          best = c;
        }
      } else {
        break; // candidates sorted by start; no further ones will start before current
      }
    }
    if (!best || best.win.end <= current + tol) {
      break; // cannot progress further
    }
    // Advance current and add to result
    current = best.win.end;
    result.push(best);
    // Move idx forward past candidates that start before or at best.start
    while (
      idx < candidates.length &&
      candidates[idx].win.start <= best.win.start + tol
    )
      idx++;
  }

  // Map to public URLs and windows; ignore errors if already public
  const out = [];
  for (const chosen of result) {
    try {
      if (typeof chosen.file.makePublic === "function") {
        try {
          await chosen.file.makePublic();
        } catch (_) {
          /* noop */
        }
      }
    } catch (_) {
      /* noop */
    }
    let url;
    if (typeof chosen.file.publicUrl === "function") {
      url = chosen.file.publicUrl();
    } else {
      const encodedName = chosen.name
        .split("/")
        .map(encodeURIComponent)
        .join("/");
      url = `https://storage.googleapis.com/${bucketName}/${encodedName}`;
    }
    out.push({ url, win: chosen.win });
  }
  return out;
}

// Join multiple clip segment URLs into a single MP4 and upload to GCS.
// segments expects array of { url, win: { start, end } }.
async function joinClipsAndUpload(
  sessionId,
  startSeconds,
  endSeconds,
  segments,
  userId
) {
  try {
    const tmpBase = await fs.mkdtemp(path.join(os.tmpdir(), "join-"));
    const trimmedFiles = [];
    let idx = 0;
    for (const seg of segments) {
      const u = seg.url;
      const win = seg.win || { start: startSeconds, end: endSeconds };
      const overlapStart = Math.max(startSeconds, Number(win.start));
      const overlapEnd = Math.min(endSeconds, Number(win.end));
      const offsetWithinFile = Math.max(0, overlapStart - Number(win.start));
      const duration = Math.max(0, overlapEnd - overlapStart);
      if (!(duration > 0)) continue;

      const res = await fetch(u);
      if (!res.ok)
        throw new Error(`failed to download segment ${u}: ${res.status}`);
      const buf = Buffer.from(await res.arrayBuffer());
      const rawPath = path.join(
        tmpBase,
        `raw_${String(idx).padStart(3, "0")}.mp4`
      );
      await fs.writeFile(rawPath, buf);

      const trimPath = path.join(
        tmpBase,
        `trim_${String(idx++).padStart(3, "0")}.mp4`
      );
      // Re-encode to ensure exact cut boundaries
      await new Promise((resolve, reject) => {
        const args = [
          "-hide_banner",
          "-nostats",
          "-loglevel",
          "error",
          "-y",
          "-ss",
          String(offsetWithinFile),
          "-t",
          String(duration),
          "-i",
          rawPath,
          "-c:v",
          "libx264",
          "-c:a",
          "aac",
          "-movflags",
          "+faststart",
          "-pix_fmt",
          "yuv420p",
          trimPath,
        ];
        const cmd = spawn("ffmpeg", args, { stdio: "ignore" });
        cmd.on("error", reject);
        cmd.on("exit", (code) =>
          code === 0
            ? resolve(undefined)
            : reject(new Error(`ffmpeg trim exited ${code}`))
        );
      });

      trimmedFiles.push(trimPath);
    }
    if (trimmedFiles.length === 0)
      throw new Error("no trimmed segments produced");

    const listPath = path.join(tmpBase, "list.txt");
    const listContent = trimmedFiles
      .map((f) => `file '${f.replace(/'/g, "'\\''")}'`)
      .join("\n");
    await fs.writeFile(listPath, listContent, "utf8");
    const concatPath = path.join(tmpBase, "out_concat.mp4");

    const runFfmpeg = (args) =>
      new Promise((resolve, reject) => {
        const quietArgs = [
          "-hide_banner",
          "-nostats",
          "-loglevel",
          "error",
          ...args,
        ];
        const cmd = spawn("ffmpeg", quietArgs, { stdio: "ignore" });
        cmd.on("error", reject);
        cmd.on("exit", (code) =>
          code === 0
            ? resolve(undefined)
            : reject(new Error(`ffmpeg exited ${code}`))
        );
      });

    try {
      await runFfmpeg([
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        listPath,
        "-c",
        "copy",
        concatPath,
      ]);
    } catch (_) {
      // Fallback to re-encode if stream copy fails
      await runFfmpeg([
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        listPath,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        concatPath,
      ]);
    }

    // Final exact trim to the requested duration [0, end-start]
    const finalPath = path.join(tmpBase, "out_final.mp4");
    const targetDuration = Math.max(0, endSeconds - startSeconds);
    await runFfmpeg([
      "-y",
      "-ss",
      "0",
      "-t",
      String(targetDuration),
      "-i",
      concatPath,
      "-c:v",
      "libx264",
      "-c:a",
      "aac",
      "-movflags",
      "+faststart",
      "-pix_fmt",
      "yuv420p",
      finalPath,
    ]);

    const safeUser = String(userId ?? "unknown");
    const targetName = `report-clips/${safeUser}/${sessionId}/joined_from_${startSeconds.toFixed(
      3
    )}s_to_${endSeconds.toFixed(3)}s_exact_${targetDuration.toFixed(3)}s.mp4`;
    const file = bucket.file(targetName);
    await file.save(await fs.readFile(finalPath), {
      contentType: "video/mp4",
      resumable: false,
    });
    try {
      await file.makePublic();
    } catch (_) {
      /* noop */
    }
    const publicUrl =
      typeof file.publicUrl === "function"
        ? file.publicUrl()
        : `https://storage.googleapis.com/${bucketName}/${targetName
            .split("/")
            .map(encodeURIComponent)
            .join("/")}`;

    // Cleanup tmp dir best-effort
    try {
      await fs.rm(tmpBase, { recursive: true, force: true });
    } catch (_) {
      /* noop */
    }

    return publicUrl;
  } catch (e) {
    console.warn("Failed to join clips; falling back to multiple segments", e);
    return null;
  }
}

async function replaceEvidenceWithClipUrls(parsed, userId) {
  if (!Array.isArray(parsed)) return [];

  const sessionFilesCache = new Map();

  const appendUserId = (u, pid) => {
    try {
      const url = new URL(String(u));
      const person = (pid ?? "").toString();
      if (!person) return url.toString();
      // Do not duplicate if already set
      if (!url.searchParams.has("user_id")) {
        url.searchParams.set("user_id", person);
      } else if (!url.searchParams.get("user_id")) {
        url.searchParams.set("user_id", person);
      }
      return url.toString();
    } catch (_) {
      // Fallback string concat if URL constructor fails
      const s = String(u || "");
      if (!pid) return s;
      return `${s}${s.includes("?") ? "&" : "?"}user_id=${encodeURIComponent(
        pid
      )}`;
    }
  };

  const getSessionIdFromUrl = (u) => {
    try {
      const url = new URL(u);
      const parts = url.pathname.split("/").filter(Boolean);
      const idx = parts.indexOf("replay");
      return idx >= 0 ? parts[idx + 1] : null;
    } catch (_) {
      return null;
    }
  };

  for (const insight of parsed) {
    if (!insight || !Array.isArray(insight.evidence)) continue;
    const newEvidence = [];
    for (const item of insight.evidence) {
      if (item && typeof item === "object") {
        const analysisIdNum = Number(item?.analysisId);
       /*  if (Number.isFinite(analysisIdNum) && analysisIdNum <= 15176) {
          // Old analyses: keep PostHog URL, skip custom clip generation
          const originalUrl = String(item.url || "");
          let pid = String(item?.person_id || "");
          if (!pid) {
            try {
              const q = new URL(originalUrl).searchParams;
              pid = String(q.get("user_id") || "");
            } catch (_) {
            }
          }
          const url = appendUserId(originalUrl, pid);
          newEvidence.push({
            url,
            start_time: item.start_time || "",
            end_time: item.end_time || "",
          });
          continue;
        } */
        const originalUrl = String(item.url || "");
        let sessInfo = parseEvidenceUrl(originalUrl);
        const sessionId =
          sessInfo?.sessionId || getSessionIdFromUrl(originalUrl);
        const startSec = Number.isFinite(timeToSecondsStrict(item.start_time))
          ? timeToSecondsStrict(item.start_time)
          : undefined;
        const endSec = Number.isFinite(timeToSecondsStrict(item.end_time))
          ? timeToSecondsStrict(item.end_time)
          : undefined;
        let pid = String(item?.person_id || "");
        if (!pid) {
          try {
            const q = new URL(originalUrl).searchParams;
            pid = String(q.get("user_id") || "");
          } catch (_) {
            /* noop */
          }
        }
        if (
          sessionId &&
          Number.isFinite(startSec) &&
          Number.isFinite(endSec) &&
          endSec > startSec
        ) {
          try {
            const segments = await findClipPublicUrlsForRange(
              sessionId,
              startSec,
              endSec,
              sessionFilesCache
            );
            console.log("dataaa ", sessionId, startSec, endSec);
            console.log("segmentssegmentssegments ", segments);
            if (segments.length) {
              const joinedUrl = await joinClipsAndUpload(
                sessionId,
                startSec,
                endSec,
                segments,
                userId
              );
              if (joinedUrl) {
                console.log("joinedUrljoinedUrljoinedUrl ", joinedUrl);
                newEvidence.push({
                  url: appendUserId(joinedUrl, pid),
                  start_time: secondsToTimeStr(startSec),
                  end_time: secondsToTimeStr(endSec),
                });
                continue;
              }
              // Fallback if join failed: emit per segment
              for (const seg of segments) {
                const segStart = Math.max(startSec, seg.win.start);
                const segEnd = Math.min(endSec, seg.win.end);
                newEvidence.push({
                  url: appendUserId(seg.url, pid),
                  start_time: secondsToTimeStr(segStart),
                  end_time: secondsToTimeStr(segEnd),
                });
              }
              continue;
            }
          } catch (e) {
            console.warn("Failed to find covering clips for", originalUrl, e);
          }
        }
        // Fallback to single-point event lookup if times invalid
        const eventSeconds = Number.isFinite(sessInfo?.eventSeconds)
          ? sessInfo.eventSeconds
          : Number.isFinite(startSec)
          ? startSec
          : NaN;
        if (sessionId && Number.isFinite(eventSeconds)) {
          try {
            const clipUrl = await findClipPublicUrlForEvent(
              sessionId,
              eventSeconds,
              sessionFilesCache
            );
            newEvidence.push({
              url: appendUserId(clipUrl || originalUrl, pid),
              start_time: item.start_time || "",
              end_time: item.end_time || "",
            });
            continue;
          } catch (e) {
            console.warn(
              "Failed to find/replace evidence object for",
              originalUrl,
              e
            );
          }
        }
        // Fallback: keep original object
        newEvidence.push({
          url: appendUserId(originalUrl, pid),
          start_time: item.start_time || "",
          end_time: item.end_time || "",
        });
        continue;
      }

      // Back-compat: string entries -> wrap into object without times
      const str = String(item || "");
      let sessInfo = parseEvidenceUrl(str);
      let sessionId = sessInfo?.sessionId || getSessionIdFromUrl(str);
      let eventSeconds = Number.isFinite(sessInfo?.eventSeconds)
        ? sessInfo.eventSeconds
        : NaN;
      if (sessionId && Number.isFinite(eventSeconds)) {
        try {
          const clipUrl = await findClipPublicUrlForEvent(
            sessionId,
            eventSeconds,
            sessionFilesCache
          );
          newEvidence.push({
            url: clipUrl || str,
            start_time: "",
            end_time: "",
          });
          continue;
        } catch (e) {
          console.warn("Failed to find/replace evidence for", str, e);
        }
      }
      newEvidence.push({ url: str, start_time: "", end_time: "" });
    }
    insight.evidence = newEvidence;
  }

  const parsed2 = parsed.map((insight) => {
    const nowIso = new Date().toISOString();
    return {
      user_status: "new",
      first_seen: nowIso,
      last_seen: nowIso,
      priority: insight.priority,
      title: insight.title,
      diagnosis: insight.problem,
      recommendation: insight.solution,
      evidence: Array.isArray(insight.evidence)
        ? insight.evidence.map((ev) =>
            typeof ev === "string" ? ev : String(ev?.url || "")
          )
        : [],
    };
  });

  try {
    return JSON.stringify(parsed2);
  } catch (e) {
    console.warn("Failed to stringify modified report; returning original.");
    return JSON.stringify(parsed);
  }
}

async function main() {
  console.log("Starting script...");

  const flaggedUsersQuery = `
      SELECT email
      FROM user2
      WHERE generate_report = true
        AND email IS NOT NULL;
    `;

  const { rows: flaggedUsers } = await pool.query(flaggedUsersQuery);

  if (!flaggedUsers.length) {
    console.log("No users flagged for report generation.");
    return;
  }

  for (const { email } of flaggedUsers) {
    if (!email) continue;

    console.log(`Fetching session analyses for ${email}...`);

    let formattedSessions = "";
    let userId = null;
    let sessions = [];
    try {
      const analysisRes = await getSessionAnalysesByEmail(email);
      sessions = analysisRes?.sessions ?? [];
      formattedSessions = analysisRes?.formatted ?? "";
      userId = analysisRes?.userId ?? null;
    } catch (error) {
      console.error(`Failed to fetch analyses for ${email}`, error);
    }


    const prompt = formattedSessions;

    const costSessionId = initializeSessionCostTracking(null, userId);

    if (!userId) {
      console.warn(`No user ID found for ${email}. Skipping report save.`);
      continue;
    }

    const schema = {
      type: "array",
      items: {
        type: "object",
        properties: {
          title: {
            type: "string",
            description: "Title for the insight.",
          },
          problem: {
            type: "string",
            description:
              "A concise explanation of the problem. Describe the observed user behavior and its impact on the business. This section answers the question 'What is happening and why does it matter?'",
          },
          solution: {
            type: "string",
            description:
              "A solution to this insight.",
          },
          evidence: {
            type: "array",
            description:
              "Provide the ID of each session that demonstrates the problem.",
            items: {
              type: "string",
            },
          },
        },
        required: ["title", "problem", "solution", "evidence"],
      },
    };

    const schema2 = {
      type: "array",
      items: {
        type: "object",
        properties: {
          priority: {
            type: "string",
            description: "The overall business priority of the insight.",
            enum: ["Critical", "High", "Medium", "Low"],
          },
          title: {
            type: "string",
            description: "Title for the insight.",
          },
          problem: {
            type: "string",
            description:
              "A concise explanation of the problem. Describe the observed user behavior and its impact on the business. This section answers the question 'What is happening and why does it matter?'",
          },
          solution: {
            type: "string",
            description:
              "Propose what the SaaS founder should do to solve the diagnosed problem. This is the 'what to do next'.",
          },
          evidence: {
            type: "array",
            description:
              "Provide the ID of each session that demonstrates the problem.",
            items: {
              type: "string",
            },
          },
        },
        required: ["priority", "title", "problem", "solution", "evidence"],
      },
    };
    // No additional schemas needed; use `schema` for both prompts

    const schemaRankOnly = {
      type: "array",
      items: {
        type: "object",
        properties: {
          priority: {
            type: "string",
            description: "The overall business priority of the insight.",
            enum: ["Critical", "High", "Medium", "Low"],
          },
          title: {
            type: "string",
            description: "Title for the insight.",
          },
        },
        required: ["priority", "title"],
      },
    };

    // If USE_FILE=true, skip generation and load from user-scoped merged file
    const mergedFile = `merged-output-${userId}.json`;
    let merged;
    if (process.env.USE_FILE === "true") {
      try {
        const raw = fs2.readFileSync(mergedFile, "utf8");
        const data = JSON.parse(raw);
        merged = Array.isArray(data) ? data : [];
        console.log(
          `Loaded merged insights from ${mergedFile} (USE_FILE=true)`
        );
      } catch (e) {
        console.warn(`USE_FILE=true but failed to load ${mergedFile};`, e);
        return;
      }
    }

    console.log("fnkwefewjfnew ", prompt);
    /*     fs2.writeFileSync("./hi.txt", prompt);
    return; */
    if (merged === undefined) {
      // Run the content generation calls in parallel.
      // Add more by pushing to this tasks array.
      /*   const tasks = [
        {
          key: "moneyLoss",
          systemInstruction: `Show me where I'm losing money. Only consider insights that appear in 3 or more sessions.`,
          contents: `Show me where I'm losing money. Only consider insights that appear in 3 or more sessions.`,
        },
        {
          key: "actionPlan",
          systemInstruction: `Create my action plan. Only consider insights that appear in 3 or more sessions.`,
          contents: `Create my action plan. Only consider insights that appear in 3 or more sessions.`,
        },
        {
          key: "bugs",
          systemInstruction: `Was there any bug or error that occurred in 3 or more sessions? If not, return an empty array.`,
          contents:
            "Was there any bug or error that occurred in 3 or more sessions? If not, return an empty array.",
        },
        {
          key: "priorities",
          systemInstruction: `What should be my top priorities? Only consider insights that appear in 3 or more sessions.`,
          contents:
            "What should be my top priorities? Only consider insights that appear in 3 or more sessions.",
        },
      ]; */
      const tasks = [
        {
          key: "recurringBugs",
          systemInstruction: `Was there any bug or error that occurred in 3 or more sessions? If not, return an empty array.`,
          contents:
            "Was there any bug or error that occurred in 3 or more sessions? If not, return an empty array.",
        },
        {
          key: "revenueLeaks",
          systemInstruction: `Show me where I'm losing money. Only consider insights that appear in 3 or more sessions.`,
          contents: `Show me where I'm losing money. Only consider insights that appear in 3 or more sessions.`,
        },
 {
          key: "revenueGrowthIdeas",
          systemInstruction: `List ideas that would increase my revenue`,
          contents: `List ideas that would increase my revenue`,
        },
 
           {
          key: "retentionIdeas",
          systemInstruction: `List ideas that would increase retention.`,
          contents: `List ideas that would increase retention.`,
        },


           /*  {
          key: "topPriorities",
          systemInstruction: `What should be my top priorities? Only consider insights that appear in 3 or more sessions.`,
          contents:
            "What should be my top priorities? Only consider insights that appear in 3 or more sessions.",
        }, */
      /*          {
          key: "product10xIdeas",
          systemInstruction: `List ideas that would make my SaaS 10x better`,
          contents: `List ideas that would make my SaaS 10x better`,
        }, */
      /*   {
          key: "marketing",
          systemInstruction: `List ideas that would make my marketing better.`,
          contents: `List ideas that would make my marketing better.`,
        }, */
     /*    {
          key: "revenueGrowthIdeas",
          systemInstruction: `List ideas that would increase my revenue`,
          contents: `List ideas that would increase my revenue`,
        },
        {
          key: "churnReductionIdeas",
          systemInstruction: `List ideas that would decrease churn`,
          contents: `List ideas that would decrease churn`,
        },
        {
          key: "acquisitionGrowthIdeas",
          systemInstruction: `List ideas that would increase user acquisition`,
          contents: `List ideas that would increase user acquisition`,
        },
        {
          key: "acquisitionGrowthIdeas2",
          systemInstruction: `List ideas that would increase customer acquisition`,
          contents: `List ideas that would increase customer acquisition`,
        },
        {
          key: "retentionIdeas",
          systemInstruction: `List ideas that would increase retention.`,
          contents: `List ideas that would increase retention.`,
        },
        {
          key: "quickWinsPolish",
          systemInstruction: `Are there some subtle things that would improve my website and business?`,
          contents: `Are there some subtle things that would improve my website and business?`,
        },

 */









        

        /* {
          key: "quickWinsPolish",
          systemInstruction: `Please find all sessions (evidences) where this happened: "High Friction in Registration Process"`,
          contents: `Please find all sessions (evidences) where this happened: "High Friction in Registration Process"`,
        }, */
       /*  {
          key: "quickWinsPolish2",
          systemInstruction: `Please find all sessions (evidence) which could suggest this need: "Automated Weekly Performance Digest"

"diagnosis":"Users currently have to manually log in and navigate to the Dashboard to view their performance metrics, as seen in sessions where users toggle date ranges to assess value. This reliance on proactive user behavior increases the risk of churn if users become too busy to log in and lose sight of the platform's ROI.","recommendation":"Implement automated weekly email digests that summarize key performance indicators (e.g., new members, revenue generated). This pushes the value of the platform directly to the user's inbox, reinforcing the habit loop and demonstrating continuous value without requiring a login."          `,
          contents: `Please find all sessions (evidence) which could suggest this need: "Automated Weekly Performance Digest"

"diagnosis":"Users currently have to manually log in and navigate to the Dashboard to view their performance metrics, as seen in sessions where users toggle date ranges to assess value. This reliance on proactive user behavior increases the risk of churn if users become too busy to log in and lose sight of the platform's ROI.","recommendation":"Implement automated weekly email digests that summarize key performance indicators (e.g., new members, revenue generated). This pushes the value of the platform directly to the user's inbox, reinforcing the habit loop and demonstrating continuous value without requiring a login."`,
        }
        ,
        {
          key: "quickWinsPolish32",
          systemInstruction: `Please find all sessions (evidence) where this happened: "Insufficient Feedback During Async Actions Causes Rage Clicking" OR "Long waiting times for async actions"`,
          contents: `Please find all sessions (evidence) where this happened: "Insufficient Feedback During Async Actions Causes Rage Clicking" OR "Long waiting times for async actions"`,
        } */
      ];
      /*   const tasks =  [
            {
              key: "bugs",
              systemInstruction:
                `Was there any bug that occurred? If not, return an empty array.`,
              contents:
                "Was there any bug that occurred? If not, return an empty array."
            },
            {
              key: "errors",
              systemInstruction:
                `Was there any error that occurred? If not, return an empty array.`,
              contents:
                "Was there any error that occurred? If not, return an empty array."
            },
          ]
      */

      const commonRequest = {
        model: Gemini,
        config: {
          responseMimeType: "application/json",
          responseSchema: schema,
          temperature: 0.02,
          top_p: 0.95,
        },
      };

      // Create/reuse explicit cache for the large prompt once, then reuse across tasks
      const cachedContentName = await ensurePromptCache(ai, Gemini, prompt, {
        sessionId: costSessionId,
        promptLabel: "cache:create:main-prompt",
      });

      // Run tasks sequentially instead of in parallel
      const parsedArrays = [];
      for (let i = 0; i < tasks.length; i++) {
        const t = tasks[i];
        const label = t?.key || `task_${i}`;

        // Build ignore list from titles of all prior results
        const prior = parsedArrays.flat();
        const seen = new Set();
        const ignoreTitles = [];
        for (const it of prior) {
          const title = typeof it?.title === "string" ? it.title : "";
          if (title && !seen.has(title)) {
            seen.add(title);
            ignoreTitles.push(title);
          }
        }
        const ignoreBlock =
          ignoreTitles.length > 0
            ? /*  ? `` */
              `\n\nIgnore these insights: """\n- ${ignoreTitles.join(
                "\n- "
              )}\n"""`
            : "";

        console.log(
          "fewfnewjfewnfjwenf ",
          `${t.systemInstruction}\n\n${ignoreBlock}`
        );

        try {
          const res = await generateContentWithRetry(
            ai,
            {
              ...commonRequest,
              // Attach cached prompt to reduce repeated input tokens
              config: {
                ...commonRequest.config,
                cachedContent: cachedContentName,
              },
              systemInstruction: `${t.systemInstruction}\n\n${ignoreBlock}`,
              // Include ignore list in the per-task contents
              contents: `${t.systemInstruction}\n\n${ignoreBlock}`,
            },
            3,
            apiKeys,
            {
              sessionId: costSessionId,
              promptLabel: `task:${label}`,
              promptContext: { userId, email },
            }
          );
          const text = res?.text;
          try {
            const arr = JSON.parse(text);
            parsedArrays.push(Array.isArray(arr) ? arr : []);
          } catch (_) {
            console.warn(
              `Report output for task '${label}' was not valid JSON.`
            );
            parsedArrays.push([]);
          }
        } catch (err) {
          console.warn(`Task '${label}' failed`, err);
          parsedArrays.push([]);
        }
      }

      merged = parsedArrays.flat();

      try {
        fs2.writeFileSync(mergedFile, JSON.stringify(merged, null, 2));
        console.log(`Saved merged array to ${mergedFile}`);
      } catch (e) {
        console.warn("Failed to save merged array:", e);
      }
    }

    // Fetch existing reports for this user and split by status buckets
    let existingReports = [];
    try {
      const { rows } = await pool.query(
        `
            SELECT id, userid AS "userId", content, status, createdat AS "createdAt"
            FROM report
            WHERE userid = $1;
          `,
        [userId]
      );
      existingReports = Array.isArray(rows) ? rows : [];
    } catch (e) {
      console.warn("Failed to fetch existing reports for user", { userId, e });
      existingReports = [];
    }

    // Normalize user_status from the JSON in content and categorize
    const norm = (s) => (typeof s === "string" ? s.toLowerCase() : "");
    const getUserStatus = (row) => {
      try {
        const c = row?.content;
        const obj = typeof c === "string" ? JSON.parse(c) : c;
        const s = obj?.user_status ?? obj?.userStatus ?? obj?.status;
        return norm(s);
      } catch (_) {
        return "";
      }
    };

    const activeReports = existingReports.filter(
      (r) => !["accepted", "ignored", "resolved"].includes(getUserStatus(r))
    );
    const acceptedOrIgnoredReports = existingReports.filter((r) =>
      ["accepted", "ignored"].includes(getUserStatus(r))
    );
    const resolvedReports = existingReports.filter(
      (r) => getUserStatus(r) === "resolved"
    );

    console.log(
      `Report buckets for user ${userId}: active=${activeReports.length}, acceptedOrIgnored=${acceptedOrIgnoredReports.length}, resolved=${resolvedReports.length}`
    );

    // For each active and accepted/ignored report, compare to each merged candidate in parallel.
    // If active matches: merge candidate evidence into that active and remove candidate from merged.
    // If accepted/ignored matches: remove candidate from merged; do not merge evidence.
    // Exclude and merge flows (no per-pair boolean prompts)
    let mergedWorking;
    try {
      // Working copy of merged
      mergedWorking = Array.isArray(merged) ? merged.slice() : [];
      const activeParsed = activeReports.map((row) => {
        try {
          const c = row?.content;
          return typeof c === "string" ? JSON.parse(c) : c;
        } catch (_) {
          return null;
        }
      });
      const acceptedParsed = acceptedOrIgnoredReports.map((row) => {
        try {
          const c = row?.content;
          return typeof c === "string" ? JSON.parse(c) : c;
        } catch (_) {
          return null;
        }
      });
      // Resolved reports: only include ones created within the last 3 days
      const nowMs = Date.now();
      let threeDaysMs = 3 * 24 * 60 * 60 * 1000;
      if (process.env["7_DAYS"] === "true") {
        threeDaysMs = 7 * 24 * 60 * 60 * 1000;
      }
      const resolvedRecentReports = resolvedReports.filter((row) => {
        const ts = row?.createdAt ? new Date(row.createdAt).getTime() : NaN;
        return Number.isFinite(ts) && nowMs - ts <= threeDaysMs;
      });
      const resolvedParsed = resolvedRecentReports.map((row) => {
        try {
          const c = row?.content;
          return typeof c === "string" ? JSON.parse(c) : c;
        } catch (_) {
          return null;
        }
      });
      const resolvedCreatedAtMs = resolvedRecentReports.map((row) => {
        const ts = row?.createdAt ? new Date(row.createdAt).getTime() : NaN;
        return Number.isFinite(ts) ? ts : NaN;
      });

      // Optional: single prompt to filter candidates against Accepted/Ignored/Resolved<3d
      const useSingleFilterPrompt =
        (process.env.SINGLE_FILTER_PROMPT || "true").toLowerCase() !== "false";
      if (useSingleFilterPrompt) {
        try {
          // 1) Filtering prompt: exclude against accepted + recent resolved using structured objects
          const historical = []
            .concat(acceptedParsed || [])
            .concat(resolvedParsed || [])
            .filter(Boolean)
            .map((o) => ({
              title: o?.title ?? "",
              problem: o?.diagnosis ?? o?.problem ?? "",
              solution: o?.recommendation ?? o?.solution ?? "",
            }));
          const candidatesStructured = (mergedWorking || []).map((o) => ({
            title: o?.title ?? "",
            problem: o?.diagnosis ?? o?.problem ?? "",
            solution: o?.recommendation ?? o?.solution ?? "",
            evidence: Array.isArray(o?.evidence)
              ? o.evidence.map((ev) =>
                  typeof ev === "string" ? ev : String(ev?.url || "")
                )
              : [],
          }));

          /* console.log("nfwenfwejfjwn ", JSON.stringify(candidatesStructured), "\n\n\n\n\n\n", JSON.stringify(historical))
           */
          if (historical.length > 0 && candidatesStructured.length > 0) {
            const respFilter = await generateContentWithRetry(
              ai,
              {
                model: Gemini,
                config: {
                  responseMimeType: "application/json",
                  // Expect diagnosis/recommendation keys for filtering
                  responseSchema: schema,
                  temperature: 0.02,
                  top_p: 0.95,
                },
                systemInstruction:
                  "Return the main list, excluding any insights that are in the filtering list.",
                contents: `Return the main list, excluding any insights that are in the filtering list.\n\n<mainlist>\n${JSON.stringify(
                  candidatesStructured
                )}\n</mainlist>\n\n<filteringlist>\n${JSON.stringify(
                  historical
                )}\n</filteringlist>`,
              },
              3,
              apiKeys,
              {
                sessionId: costSessionId,
                promptLabel: "filter-candidates-against-history",
                promptContext: { userId, email },
              }
            )
              .then((r) => String(r?.text || "[]"))
              .catch(() => "[]");
            // console.log("filter response raw ", respFilter)
            try {
              const parsed = JSON.parse(respFilter);
              // console.log("parsedparsed ", parsed)
              if (Array.isArray(parsed)) {
                const before = mergedWorking.length;
                mergedWorking = parsed.map((o) => ({
                  title: o?.title ?? "",
                  // Accept either diagnosis/recommendation or problem/solution
                  problem: o?.problem ?? o?.diagnosis ?? "",
                  solution: o?.solution ?? o?.recommendation ?? "",
                  evidence: Array.isArray(o?.evidence) ? o.evidence : [],
                }));
                const after = mergedWorking.length;
                console.log(
                  `Filtered candidates into ${after} insights (was ${before}).`
                );
              }
            } catch (_) {}
          }

          // console.log("proceeding after filter step")

          // 2) Merge prompt: merge DB 'new' with the filtered mergedWorking
          const newReports = existingReports.filter(
            (r) => getUserStatus(r) === "new"
          );
          const newParsed = newReports
            .map((row) => {
              try {
                const c = row?.content;
                return typeof c === "string" ? JSON.parse(c) : c;
              } catch (_) {
                return null;
              }
            })
            .filter(Boolean);
          const newAsSchemaInput = (newParsed || []).map((o) => ({
            title: o?.title ?? "",
            problem: o?.problem ?? o?.diagnosis ?? "",
            solution: o?.solution ?? o?.recommendation ?? "",
            evidence: Array.isArray(o?.evidence)
              ? o.evidence.map((ev) =>
                  typeof ev === "string" ? ev : String(ev?.url || "")
                )
              : [],
          }));
          const candidatesObjects = (mergedWorking || []).map((o) => ({
            title: o?.title ?? "",
            problem: o?.problem ?? o?.diagnosis ?? "",
            solution: o?.solution ?? o?.recommendation ?? "",
            evidence: Array.isArray(o?.evidence)
              ? o.evidence.map((ev) =>
                  typeof ev === "string" ? ev : String(ev?.url || "")
                )
              : [],
          }));

          console.log(
            "nfwenfwejfjwn ",
            JSON.stringify(newAsSchemaInput),
            "\n\n\n\n\n\n",
            JSON.stringify(candidatesObjects)
          );
          if (newAsSchemaInput.length === 0 && candidatesObjects.length === 0) {
            // No new insights to merge; use the existing candidates as-is and skip the merge prompt
            const before = mergedWorking.length;
            mergedWorking = [];
            const after = mergedWorking.length;
            console.log(
              `No new insights; using ${after} candidate insight(s) (was ${before}).`
            );
          } else if (
            newAsSchemaInput.length > 0 &&
            candidatesObjects.length === 0
          ) {
            // No new insights to merge; use the existing candidates as-is and skip the merge prompt
            const before = mergedWorking.length;
            mergedWorking = newAsSchemaInput;
            const after = mergedWorking.length;
            console.log(
              `No new insights; using ${after} candidate insight(s) (was ${before}).`
            );
          } else if (
            newAsSchemaInput.length === 0 &&
            candidatesObjects.length > 0
          ) {
            // No new insights to merge; use the existing candidates as-is and skip the merge prompt
            const before = mergedWorking.length;
            mergedWorking = candidatesObjects;
            const after = mergedWorking.length;
            console.log(
              `No new insights; using ${after} candidate insight(s) (was ${before}).`
            );
          } else if (
            newAsSchemaInput.length > 0 &&
            candidatesObjects.length > 0
          ) {
            /*     return; */

            const respMerge = await generateContentWithRetry(
              ai,
              {
                model: Gemini,
                config: {
                  responseMimeType: "application/json",
                  responseSchema: schema,
                  temperature: 0.02,
                  top_p: 0.95,
                },
                systemInstruction:
                  "Please create one complete action plan from these two lists. Max: 50 insights",
                contents: `Please create one complete action plan from these two lists.  Max: 50 insights

<list>
${JSON.stringify(newAsSchemaInput)}
</list>

<list>
${JSON.stringify(candidatesObjects)}
</list>`,
              },
              3,
              apiKeys,
              {
                sessionId: costSessionId,
                promptLabel: "merge-after-filter-new-and-candidates",
                promptContext: { userId, email },
              }
            )
              .then((r) => String(r?.text || "[]"))
              .catch(() => "[]");

            try {
              const parsed = JSON.parse(respMerge);
              if (Array.isArray(parsed)) {
                const before = mergedWorking.length;
                mergedWorking = parsed.map((o) => ({
                  title: o?.title ?? "",
                  problem: o?.problem ?? "",
                  solution: o?.solution ?? "",
                  evidence: Array.isArray(o?.evidence) ? o.evidence : [],
                }));
                const after = mergedWorking.length;
                console.log(
                  `Merged newlist + filtered candidates into ${after} insights (was ${before}).`
                );
              }
            } catch (_) {}
          }
        } catch (singleErr) {
          console.warn(
            "Single filter+merge flow failed; proceeding without it",
            singleErr
          );
        }
      }

      // After filtering/merging flows, use the working set directly
      merged = mergedWorking;
    } catch (e) {
      console.warn("Failed equivalence matching and evidence merge step", e);
      // Fallback to original merged as working set to keep downstream prompts running
      mergedWorking = Array.isArray(merged) ? merged.slice() : [];
    }

    let dedupeAndMergePrompt;
    let dedupeAndMergeText;
    const dedupeFile = `dedupe-merge-output-${userId}.json`;
    if (process.env.USE_FILE2 === "true") {
      try {
        dedupeAndMergeText = fs2.readFileSync(dedupeFile, "utf8");
        console.log(
          `Loaded dedupe/merge insights from ${dedupeFile} (USE_FILE2=true)`
        );
      } catch (e) {
        console.warn(`USE_FILE2=true but failed to load ${dedupeFile};`, e);
        return;
      }
    } else {
      try {
        const isRankSeparately = process.env.RANK_SEPARATELY === "true";
        const currentSchema = isRankSeparately ? schemaRankOnly : schema2;

        dedupeAndMergePrompt = await generateContentWithRetry(
          ai,
          {
            model: Gemini,
            config: {
              responseMimeType: "application/json",
              responseSchema: currentSchema,
              /* thinkingConfig: {
                thinkingBudget: 32768,  // Max value for deepest reasoning
                // Alternative: thinkingBudget: -1 for dynamic auto-max
              },  */
              temperature: 0.02,
              top_p: 0.95,
            },
            systemInstruction: `Your job is to rank all of these insights.`,
            contents: `Your job is to rank all of these insights.

<insights>
${JSON.stringify(mergedWorking)}
</insights>`,
          },
          3,
          apiKeys,
          {
            sessionId: costSessionId,
            promptLabel: "dedupe-merge",
            promptContext: { userId, email },
          }
        );
        if (isRankSeparately) {
          const ranks = JSON.parse(dedupeAndMergePrompt?.text || "[]");
          const rankMap = new Map(ranks.map((r) => [r.title, r.priority]));
          const combined = mergedWorking.map((item) => ({
            ...item,
            priority: rankMap.get(item.title) || "Low",
          }));
          dedupeAndMergeText = JSON.stringify(combined, null, 2);
        } else {
          dedupeAndMergeText = String(dedupeAndMergePrompt?.text ?? "[]");
        }
        try {
          fs2.writeFileSync(dedupeFile, dedupeAndMergeText, "utf8");
          console.log(`Saved dedupe/merge insights to ${dedupeFile}`);
        } catch (writeErr) {
          console.warn(`Failed to save ${dedupeFile}`, writeErr);
        }
      } catch (error) {
        console.error(`Failed to merge arrays for ${email}`, error);
      }
    }

    try {
      let result = JSON.parse(dedupeAndMergeText ?? "[]");

      // Attempt to enrich from saved sessions file when using file-based stages
      try {
        const enrichedSessionsFile = `dedupe-merge-output-sessions-${userId}.json`;
        if (
          process.env.USE_FILE2 === "true" ||
          process.env.USE_FILE3 === "true" ||
          process.env.USE_FILE === "true"
        ) {
          if (fs2.existsSync(enrichedSessionsFile)) {
            const rawSessions = fs2.readFileSync(enrichedSessionsFile, "utf8");
            const enriched = JSON.parse(rawSessions);
            if (Array.isArray(enriched)) {
              result = enriched;
              console.log(
                `Loaded enriched evidenceSessions from ${enrichedSessionsFile}`
              );
            }
          }
        }
      } catch (enrichErr) {
        console.warn(
          "Failed to load enriched evidenceSessions; falling back to mapping",
          enrichErr
        );
      }

      // For each evidence id within each insight, find the matching session
      try {
        const sessionById = new Map(
          Array.isArray(sessions)
            ? sessions
                .filter(
                  (s) => s && typeof s.idVar === "string" && s.idVar.length > 0
                )
                .map((s) => [s.idVar, s])
            : []
        );

        for (const insight of Array.isArray(result) ? result : []) {
          if (!insight || !Array.isArray(insight.evidence)) continue;
          const matched = [];
          for (const evId of insight.evidence) {
            if (typeof evId !== "string") continue;
            const sess = sessionById.get(evId);
            if (sess) {
              matched.push({ ...sess });
            } else {
              // Optional visibility for missing references
              console.warn("No session found for evidence id", evId);
            }
          }
          // Attach for downstream use/debugging; safe to include as it is later filtered out
          if (!Array.isArray(insight.evidenceSessions)) {
            insight.evidenceSessions = matched;
          }
        }
      } catch (mapErr) {
        console.warn("Unable to map evidence ids to sessions", mapErr);
      }

      // Persist a sanitized version of result including evidenceSessions for reuse
      try {
        const enrichedSessionsFile = `dedupe-merge-output-sessions-${userId}.json`;
        const toSave = (Array.isArray(result) ? result : []).map((ins) => {
          const copy = { ...ins };
          if (Array.isArray(copy.evidenceSessions)) {
            copy.evidenceSessions = copy.evidenceSessions.map((s) =>
              sanitizeSessionForClips(s)
            );
          }
          return copy;
        });
        fs2.writeFileSync(
          enrichedSessionsFile,
          JSON.stringify(toSave, null, 2),
          "utf8"
        );
        console.log(
          `Saved enriched evidenceSessions to ${enrichedSessionsFile}`
        );
      } catch (saveEnrichedErr) {
        console.warn(
          "Failed to save enriched evidenceSessions file",
          saveEnrichedErr
        );
      }

      let evidences;
      const skipClips =
        String(process.env.SKIP_CLIPS || "").toLowerCase() === "true";
      // For each evidenceSession within each insight, invoke Gemini in parallel,
      // unless USE_FILE3=true, in which case load saved results from file.
      try {
        let loadedFromFile3 = false;
        const sessionTasksFile = `sescmhc8rtvq0000ie04azgjmndyt-${userId}.json`;
        if (skipClips) {
          // Skip creating clips entirely; ensure arrays exist but are empty
          for (const insight of Array.isArray(result) ? result : []) {
            if (!insight || !Array.isArray(insight.evidenceSessions)) continue;
            for (const sess of insight.evidenceSessions) {
              sess.insightClips = [];
            }
          }
          console.log(
            "SKIP_CLIPS=true; skipping clip generation and leaving empty arrays."
          );
          loadedFromFile3 = true; // prevent any further processing below
        } else if (process.env.USE_FILE3 === "true") {
          // Prefer enriched full file with clips if available
          try {
            const enrichedClipsFile = `dedupe-merge-output-with-clips-${userId}.json`;
            if (fs2.existsSync(enrichedClipsFile)) {
              const raw = fs2.readFileSync(enrichedClipsFile, "utf8");
              const saved = JSON.parse(raw);
              if (Array.isArray(saved)) {
                result = saved;
                console.log(
                  `Loaded enriched clips from ${enrichedClipsFile} (USE_FILE3=true)`
                );
                loadedFromFile3 = true;
              }
            }
          } catch (_) {
            // ignore and fallback to sessionTasksFile
          }
          try {
            const raw = fs2.readFileSync(sessionTasksFile, "utf8");
            const saved = JSON.parse(raw);
            let mapById;
            if (Array.isArray(saved)) {
              mapById = new Map(
                saved.map((it) => [
                  it?.idVar || it?.id || it?.sessionId || it?.idvar,
                  Array.isArray(it?.insightClips) ? it.insightClips : [],
                ])
              );
            } else if (saved && typeof saved === "object") {
              mapById = new Map(Object.entries(saved));
            } else {
              mapById = new Map();
            }

            for (const insight of Array.isArray(result) ? result : []) {
              if (!insight || !Array.isArray(insight.evidenceSessions))
                continue;
              for (const sess of insight.evidenceSessions) {
                const clips = mapById.get(sess.idVar);
                if (Array.isArray(clips)) {
                  sess.insightClips = clips;
                } else {
                  sess.insightClips = Array.isArray(sess.insightClips)
                    ? sess.insightClips
                    : [];
                }
              }
            }
            console.log(
              `Loaded session clip results from ${sessionTasksFile} (USE_FILE3=true)`
            );
            loadedFromFile3 = true;
          } catch (e) {
            console.warn(
              `USE_FILE3=true but failed to load ${sessionTasksFile};`,
              e
            );
            return;
          }
        }

        if (!loadedFromFile3) {
          const sessionTasks = [];
          for (const insight of Array.isArray(result) ? result : []) {
            if (!insight || !Array.isArray(insight.evidenceSessions)) continue;
            for (const sess of insight.evidenceSessions) {
              const analysisIdNum = Number(sess?.analysisId);
         /*      const shouldRunTask = Number.isFinite(analysisIdNum)
                ? analysisIdNum > 15176
                : true; */
           /*    if (!shouldRunTask) {
                // Skip running the sessionTask for older analyses; we'll keep the PostHog URL later.
                continue;
              }
 */
              const leanInsight = sanitizeInsightForClips(insight);
              const leanSess = sanitizeSessionForClips(sess);

              console.log(
                "wnjefnewjnw5wee ",
                leanInsight,
                "\n\n\n\n\n\n",
                leanSess
              );
              /*    return; */
              sessionTasks.push(
                generateContentWithRetry(
                  ai,
                  {
                    model: Gemini,
                    config: {
                      responseMimeType: "application/json",
                      responseSchema: {
                        type: "array",
                        items: {
                          type: "object",
                          properties: {
                            start_time: {
                              type: "string",
                              description: "mm:ss.sss",
                            },
                            end_time: {
                              type: "string",
                              description: "mm:ss.sss",
                            },
                          },
                          required: ["start_time", "end_time"],
                        },
                      },
                      temperature: 0.02,
                      top_p: 0.95,
                    },
                    systemInstruction: `Your task is to analyze the <session> in order to select one or more self-contained clips that serve as convincing evidence of the <insight>.
  
Requirements:
- The insight must be fully understandable from each clip alone, without needing context before or after the timestamps you select.
- If you find exactly one strong self-contained clip, return an array with a single object.
- If you find several distinct strong examples of the insight, return an array of objects, one per clip.
- Each clip must be between 5 and 60 seconds long.`,
                    contents: `Your task is to analyze the <session> in order to select one or more self-contained clips that serve as convincing evidence of the <insight>.
  
Requirements:
- The insight must be fully understandable from each clip alone, without needing context before or after the timestamps you select.
- If you find exactly one strong self-contained clip, return an array with a single object.
- If you find several distinct strong examples of the insight, return an array of objects, one per clip.
- Each clip must be between 5 and 60 seconds long.

<insight>
${JSON.stringify(leanInsight)}
</insight>

<session>
${JSON.stringify(leanSess)}
</session>`,
                  },
                  3,
                  apiKeys,
                  {
                    sessionId: costSessionId,
                    promptLabel: `insight-clips:${String(
                      sess?.idVar ?? "unknown"
                    )}`,
                    promptContext: {
                      userId,
                      email,
                      analysisId: sess?.analysisId,
                      insightTitle: insight?.title,
                    },
                  }
                )
                  .then((resp) => {
                    try {
                      const text = String(resp?.text ?? "[]");
                      const parsed = JSON.parse(text);
                      if (Array.isArray(parsed)) {
                        sess.insightClips = parsed.filter(
                          (c) =>
                            c &&
                            typeof c.start_time === "string" &&
                            typeof c.end_time === "string"
                        );
                      } else {
                        sess.insightClips = [];
                      }
                    } catch (_) {
                      sess.insightClips = [];
                    }
                  })
                  .catch((e) => {
                    console.warn(
                      "Gemini insight-clips call failed for evidenceSession",
                      e
                    );
                  })
              );
            }
          }
          if (sessionTasks.length > 0) {
            evidences = await Promise.all(sessionTasks);
          }

          // Save results to file for reuse when USE_FILE3=true
          try {
            const clipsBySessionId = {};
            for (const insight of Array.isArray(result) ? result : []) {
              if (!insight || !Array.isArray(insight.evidenceSessions)) continue;
              for (const sess of insight.evidenceSessions) {
                const key = String(sess?.idVar || "");
                if (!key) continue;
                const clipsArr = Array.isArray(sess?.insightClips)
                  ? sess.insightClips
                  : [];
                // Always include the session id, even if no clips were found yet
                clipsBySessionId[key] = clipsArr;
              }
            }
            fs2.writeFileSync(
              sessionTasksFile,
              JSON.stringify(clipsBySessionId, null, 2),
              "utf8"
            );
            console.log(`Saved session clip results to ${sessionTasksFile}`);
          } catch (saveErr) {
            console.warn(`Failed to save ${sessionTasksFile}`, saveErr);
          }
          // Also persist the full dedupe result enriched with evidenceSessions + insightClips
          try {
            const enrichedClipsFile = `dedupe-merge-output-with-clips-${userId}.json`;
            const toSave = (Array.isArray(result) ? result : []).map((ins) => {
              const copy = { ...ins };
              if (Array.isArray(copy.evidenceSessions)) {
                copy.evidenceSessions = copy.evidenceSessions.map((s) => {
                  const slim = sanitizeSessionForClips(s);
                  if (Array.isArray(s.insightClips)) {
                    slim.insightClips = s.insightClips;
                  }
                  return slim;
                });
              }
              return copy;
            });
            fs2.writeFileSync(
              enrichedClipsFile,
              JSON.stringify(toSave, null, 2),
              "utf8"
            );
            console.log(`Saved enriched clips to ${enrichedClipsFile}`);
          } catch (saveFullErr) {
            console.warn("Failed to save enriched clips file", saveFullErr);
          }
        }
      } catch (hiErr) {
        console.warn("Failed running Gemini 'hi' calls in parallel", hiErr);
      }

      // Filter out evidenceSessions for which we ran a sessionTask and received an empty array
  /*     try {
        for (const insight of Array.isArray(result) ? result : []) {
          if (!insight || !Array.isArray(insight.evidenceSessions)) continue;
          insight.evidenceSessions = insight.evidenceSessions.filter((sess) => {
            const analysisIdNum = Number(sess?.analysisId);
            console.log(
              "analysisIdNumanalysisIdNumanalysisIdNum ",
              analysisIdNum
            );
            if (String(process.env.SKIP_CLIPS || "").toLowerCase() === "true") {
              // When skipping clips, keep all evidence sessions
              return true;
            }
            const ranTask =
              !Number.isFinite(analysisIdNum) || analysisIdNum > 15176;
            if (ranTask) {
              return (
                Array.isArray(sess?.insightClips) &&
                sess.insightClips.length > 0
              );
            }
            // Did not run a task for older analyses; keep for fallback handling
            return true;
          });
        }
      } catch (filterErr) {
        console.warn("Failed to filter empty evidence sessions", filterErr);
      }
 */
      // Build evidence entries as objects with url, start_time, end_time
      const resultArgument = result.map((value) => {
        const evidenceObjects = [];
        if (Array.isArray(value.evidenceSessions)) {
          for (const sess of value.evidenceSessions) {
            const baseUrl = `https://app.posthog.com/replay/${
              sess?.idVar ?? ""
            }`;
            const personIdForUrl = sess?.person_id;
            const urlWithUser = personIdForUrl
              ? `${baseUrl}${
                  baseUrl.includes("?") ? "&" : "?"
                }user_id=${encodeURIComponent(personIdForUrl)}`
              : baseUrl;
            const clips = Array.isArray(sess?.insightClips)
              ? sess.insightClips
              : [];
            for (const clip of clips) {
              if (
                clip &&
                typeof clip.start_time === "string" &&
                typeof clip.end_time === "string"
              ) {
                evidenceObjects.push({
                  url: urlWithUser,
                  start_time: clip.start_time,
                  end_time: clip.end_time,
                  analysisId: sess?.analysisId,
                  person_id: sess?.person_id,
                });
              }
            }
            // Fallback: when no clips were found for this session, keep the PostHog URL entry
            if (clips.length === 0 && (sess?.idVar || sess?.person_id)) {
              evidenceObjects.push({
                url: urlWithUser,
                start_time: "",
                end_time: "",
                analysisId: sess?.analysisId,
                person_id: sess?.person_id,
              });
            }
          }
        }
        return {
          ...value,
          evidence: evidenceObjects,
        };
      });
      console.log(
        "resultresultresult ",
        JSON.stringify(resultArgument, null, 2)
      );

      // Replace PostHog evidence links with public GCS clip URLs when possible
      const processedInsightsRaw = await replaceEvidenceWithClipUrls(
        resultArgument,
        userId
      );

      let structuredInsights = null;
      try {
        const parsedOutput = JSON.parse(processedInsightsRaw);
        if (Array.isArray(parsedOutput)) {
          structuredInsights = parsedOutput;
        } else {
          console.warn(
            "Processed insights output is not an array; falling back to single report.",
            typeof parsedOutput
          );
        }
      } catch (parseErr) {
        console.warn(
          "Failed to parse processed insights; falling back to single report.",
          parseErr
        );
      }

      if (Array.isArray(structuredInsights) && structuredInsights.length > 0) {
        // Delete any existing 'new' status reports for this user before inserting new ones
        try {
          const { rowCount: deleted } = await pool.query(
            `
                DELETE FROM report
                WHERE userid = $1
                  AND content ~ '"user_status"\s*:\s*"new"';
              `,
            [userId]
          );
          console.log(
            `Deleted ${
              deleted || 0
            } existing 'new' report(s) for ${email} before insert.`
          );
        } catch (delErr) {
          console.warn(
            "Failed to delete existing 'new' reports before insert",
            { userId, delErr }
          );
        }

        for (const insight of structuredInsights) {
          await pool.query(
            `
                INSERT INTO report (userid, content, status)
                VALUES ($1, $2, $3);
              `,
            [userId, JSON.stringify(insight), "COMPLETED"]
          );
        }
        console.log(
          `Saved ${structuredInsights.length} report${
            structuredInsights.length === 1 ? "" : "s"
          } for ${email}.`
        );
      }
    } catch (error) {
      console.error(`Failed to save report for ${email}`, error);
    }

    // Generate and persist the cost report for this session
    try {
      const costReport = await writeSessionCostReport(costSessionId);
      if (costReport && costReport.length > 0) {
        await appendSessionCostReportToBucket(
          userId,
          costSessionId,
          costReport
        );
        console.log(
          `Appended cost report for ${email} (session ${costSessionId}).`
        );
      }
    } catch (e) {
      console.warn(
        `Failed generating/appending cost report for ${email} (session ${costSessionId})`,
        e
      );
    }
  }
}

// For immediate execution when the script starts
main().catch((error) => {
  console.log("process.env.DB_HOST ", process.env.DB_HOST);
  console.error(error);
  process.exit(1);
});
