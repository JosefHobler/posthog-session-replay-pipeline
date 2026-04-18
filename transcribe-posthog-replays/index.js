import dotenv from "dotenv";
import { GoogleGenAI, createUserContent, Type } from "@google/genai";
import { Storage } from "@google-cloud/storage";
import { PuppeteerScreenRecorder } from "puppeteer-screen-recorder";
import puppeteer from "puppeteer";
import * as fs from "fs/promises";
import fs2 from "fs";
import path from "path";
import axios from "axios";
import * as os from "os";
import pako from "pako";
import { Pool } from "pg";
import crypto from "crypto";
import { execSync, spawn } from "child_process";
import { spawnSync } from "child_process";
import ffmpegStatic from "ffmpeg-static";
import ffprobeStatic from "ffprobe-static";
import sharp from "sharp";

import {
  GEMINI_BATCH_FOR_CLIP_PROMPTS,
  GEMINI_BATCH_FOR_CLIP_PROMPTS_2,
  processPendingClipBatchJobsForUser,
} from "./batchProcessing.js";
import { saveBatchContext, deleteBatchContext } from "./batchContext.js";

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const pricingInfo = {
  input: {
    standard_context_per_million_tokens: 1.25, // For prompts <= 200k tokens
    long_context_per_million_tokens: 2.5, // For prompts > 200k tokens
  },
  output: {
    standard_context_per_million_tokens: 10.0, // For prompts <= 200k tokens
    long_context_per_million_tokens: 15.0, // For prompts > 200k tokens
  },
  context_threshold: 2000000, 
};

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

  result.input.isLongContext = result.input.tokens > longContextThreshold;
  result.input.pricePerMillionTokens = result.input.isLongContext
    ? typeof inputPricing.long_context_per_million_tokens === "number"
      ? inputPricing.long_context_per_million_tokens
      : result.input.pricePerMillionTokens
    : result.input.pricePerMillionTokens;
  result.input.cost =
    (result.input.tokens / 1_000_000) * result.input.pricePerMillionTokens;
  if (!Number.isFinite(result.input.cost)) {
    result.input.cost = 0;
  }
  result.totalCost += result.input.cost;

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

const sessionCostTracker = new Map();

async function appendSessionCostReportToBucket(userId, content) {
  if (!bucketName) {
    console.warn(
      "[COST-TRACKING] GCS_BUCKET_NAME is not configured; skipping cost report upload."
    );
    return;
  }

  if (!userId) {
    console.warn(
      "[COST-TRACKING] Missing userId; cannot upload session cost report."
    );
    return;
  }

  const fileName = `${userId}-session-cost-report.log`;
  const file = bucket.file(fileName);
  let existingContent = "";

  try {
    const [exists] = await file.exists();
    if (exists) {
      const [contents] = await file.download();
      existingContent = contents.toString("utf-8");
    }
  } catch (error) {
    if (error?.code !== 404) {
      console.error(
        `[COST-TRACKING] Failed to read existing cost report for user ${userId}:`,
        error
      );
    }
  }

  const mergedContent = existingContent
    ? `${
        existingContent.endsWith("\n")
          ? existingContent
          : `${existingContent}\n`
      }${content}`
    : content;

  try {
    await file.save(mergedContent, {
      resumable: false,
      contentType: "text/plain",
    });
    console.log(
      `[COST-TRACKING] Updated cost report at gs://${bucketName}/${fileName}`
    );
  } catch (error) {
    console.error(
      `[COST-TRACKING] Failed to upload cost report for user ${userId}:`,
      error
    );
    throw error;
  }
}

function initializeSessionCostTracking(sessionId) {
  if (!sessionId) {
    return;
  }
  if (!sessionCostTracker.has(sessionId)) {
    sessionCostTracker.set(sessionId, {
      sessionId,
      createdAt: new Date().toISOString(),
      prompts: [],
      totals: {
        promptCount: 0,
        inputTokens: 0,
        inputCost: 0,
        outputTokens: 0,
        outputCost: 0,
        thoughtsTokens: 0,
        totalCost: 0,
        missingUsageCount: 0,
      },
    });
  }
}

function recordSessionPromptCost({
  sessionId,
  promptLabel,
  usageMetadata,
  requestConfig,
  extra = {},
}) {
  if (!sessionId) {
    return;
  }

  initializeSessionCostTracking(sessionId);
  const state = sessionCostTracker.get(sessionId);
  if (!state) {
    return;
  }

  const timestamp = new Date().toISOString();
  const label = promptLabel || "unnamed-prompt";

  let breakdown = null;
  if (usageMetadata) {
    try {
      breakdown = calculateApiCallCost(usageMetadata, pricingInfo, {
        suppressLog: true,
      });
    } catch (error) {
      console.warn(
        `[COST-TRACKING] Failed to compute cost for prompt "${label}" (session ${sessionId}):`,
        error
      );
    }
  } else {
    state.totals.missingUsageCount += 1;
  }

  const promptSummary = {
    label,
    timestamp,
    model: requestConfig?.model || "unknown",
    missingUsage: !usageMetadata,
    context: extra && Object.keys(extra).length > 0 ? extra : null,
    cost: breakdown
      ? {
          inputTokens: breakdown.input.tokens,
          inputCost: breakdown.input.cost,
          outputTokens: breakdown.output.tokens,
          outputCost: breakdown.output.cost,
          thoughtsTokens: breakdown.thoughts.tokens,
          totalCost: breakdown.totalCost,
        }
      : null,
    usage: usageMetadata
      ? {
          promptTokenCount:
            typeof usageMetadata.promptTokenCount === "number"
              ? usageMetadata.promptTokenCount
              : null,
          candidatesTokenCount:
            typeof usageMetadata.candidatesTokenCount === "number"
              ? usageMetadata.candidatesTokenCount
              : null,
          thoughtsTokenCount:
            typeof usageMetadata.thoughtsTokenCount === "number"
              ? usageMetadata.thoughtsTokenCount
              : null,
          promptTokensDetails: Array.isArray(usageMetadata.promptTokensDetails)
            ? usageMetadata.promptTokensDetails.map((detail) => ({
                modality: detail?.modality || "unknown",
                tokens:
                  typeof detail?.tokenCount === "number"
                    ? detail.tokenCount
                    : 0,
              }))
            : null,
        }
      : null,
  };

  state.prompts.push(promptSummary);
  state.totals.promptCount += 1;
  state.lastUpdated = timestamp;

  if (breakdown) {
    state.totals.inputTokens += breakdown.input.tokens;
    state.totals.inputCost += breakdown.input.cost;
    state.totals.outputTokens += breakdown.output.tokens;
    state.totals.outputCost += breakdown.output.cost;
    state.totals.thoughtsTokens += breakdown.thoughts.tokens;
    state.totals.totalCost += breakdown.totalCost;
  } else if (!usageMetadata) {
    console.warn(
      `[COST-TRACKING] Missing usage metadata for prompt "${label}" (session ${sessionId}).`
    );
  }
}

function resetSessionCostTracking(sessionId) {
  if (sessionId) {
    sessionCostTracker.delete(sessionId);
  }
}

function formatCurrency(value, fractionDigits = 6) {
  return Number.isFinite(value) ? value.toFixed(fractionDigits) : "0.000000";
}

async function writeSessionCostReport(sessionId, userId, metadata = {}) {
  if (!sessionId) {
    return;
  }

  const state = sessionCostTracker.get(sessionId);
  const nowIso = new Date().toISOString();
  const lines = [];

  lines.push(
    "================================================================"
  );
  lines.push(
    `[${nowIso}] Session ${sessionId} Cost Report (${
      metadata.status || "UNKNOWN"
    })`
  );
  lines.push(`Result: ${metadata.status || "UNKNOWN"}`);
  if (metadata.errorMessage) {
    lines.push(`Error: ${metadata.errorMessage}`);
  }

  if (!state || state.prompts.length === 0) {
    lines.push("Prompts Recorded: 0");
    lines.push("No prompts were recorded for this session.");
    lines.push("");
  } else {
    const totals = state.totals;
    lines.push(`Prompts Recorded: ${totals.promptCount}`);
    lines.push(
      `Totals -> Input: ${totals.inputTokens} tokens ($${formatCurrency(
        totals.inputCost
      )}), Output: ${totals.outputTokens} tokens ($${formatCurrency(
        totals.outputCost
      )}), Thoughts: ${totals.thoughtsTokens} tokens, Total: $${formatCurrency(
        totals.totalCost
      )})`
    );
    if (totals.missingUsageCount > 0) {
      lines.push(`Prompts missing usage metadata: ${totals.missingUsageCount}`);
    }
    lines.push("Prompt Breakdown:");
    state.prompts.forEach((prompt, index) => {
      lines.push(
        `  ${index + 1}. ${prompt.label} | model=${prompt.model} | timestamp=${
          prompt.timestamp
        }`
      );
      if (prompt.context) {
        lines.push(`     context=${JSON.stringify(prompt.context)}`);
      }
      if (prompt.missingUsage) {
        lines.push("     usage=missing");
      } else if (prompt.cost) {
        lines.push(
          `     input=${prompt.cost.inputTokens} tokens ($${formatCurrency(
            prompt.cost.inputCost
          )}), output=${prompt.cost.outputTokens} tokens ($${formatCurrency(
            prompt.cost.outputCost
          )}), thoughts=${
            prompt.cost.thoughtsTokens
          } tokens, total=$${formatCurrency(prompt.cost.totalCost)})`
        );
        if (prompt.usage?.promptTokensDetails) {
          const detailSummary = prompt.usage.promptTokensDetails
            .map((detail) => `${detail.modality}:${detail.tokens}`)
            .join(", ");
          if (detailSummary) {
            lines.push(`     promptTokensDetails=${detailSummary}`);
          }
        }
      }
    });
    lines.push("");
  }

  const logEntry = lines.join("\n");
  try {
    await appendSessionCostReportToBucket(userId, logEntry);
  } catch (error) {
    console.error(
      `[COST-TRACKING] Failed to persist cost report for session ${sessionId}:`,
      error
    );
  }

  resetSessionCostTracking(sessionId);
}

const rrwebPlayerBundle = readFileSync(
  join(__dirname, "node_modules", "rrweb-player", "dist", "index.js"),
  "utf-8"
);
// Build a file:// URL to load rrweb-player from disk instead of inlining
const rrwebPlayerPath = join(
  __dirname,
  "node_modules",
  "rrweb-player",
  "dist",
  "index.js"
);
const rrwebPlayerFileUrl = "file://" + rrwebPlayerPath.replace(/\\/g, "/");

// Validate the rrweb-player bundle
console.log("🔍 BUNDLE VALIDATION: rrweb-player bundle loaded");
console.log(
  `🔍 BUNDLE VALIDATION: Bundle size: ${rrwebPlayerBundle.length} characters`
);
if (rrwebPlayerBundle.length < 1000) {
  console.error(
    "❌ BUNDLE VALIDATION: Bundle is suspiciously small - may be corrupted"
  );
} else if (
  !rrwebPlayerBundle.includes("rrwebPlayer") &&
  !rrwebPlayerBundle.includes("RRWebPlayer")
) {
  console.error(
    "❌ BUNDLE VALIDATION: Bundle doesn't contain expected rrwebPlayer constructor"
  );
} else {
  console.log("✅ BUNDLE VALIDATION: Bundle appears valid");
}

// Extra diagnostics about the rrweb-player bundle to catch inline injection issues
try {
  const countOccurrences = (str, sub) => {
    try {
      const escaped = sub.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const re = new RegExp(escaped, "g");
      return (str.match(re) || []).length;
    } catch (_) {
      return 0;
    }
  };
  const hasBOM = rrwebPlayerBundle.charCodeAt(0) === 0xfeff;
  let nonAsciiCount = 0;
  for (let i = 0; i < rrwebPlayerBundle.length; i++) {
    if (rrwebPlayerBundle.charCodeAt(i) > 127) nonAsciiCount++;
  }
  const nonAsciiRatio = (
    nonAsciiCount / Math.max(1, rrwebPlayerBundle.length)
  ).toFixed(4);
  const inlineScriptTagCount = countOccurrences(rrwebPlayerBundle, "</script");
  console.log("🔍 BUNDLE VALIDATION: Extended summary:", {
    length: rrwebPlayerBundle.length,
    containsBOM: hasBOM,
    nonAsciiRatio,
    inlineScriptTagCount,
  });
  if (inlineScriptTagCount > 0) {
    console.warn(
      "⚠️ BUNDLE VALIDATION: Bundle contains </script which can break inline <script> injection. Consider using external src."
    );
  }
} catch (e) {
  console.warn(
    "⚠️ BUNDLE VALIDATION: Extended analysis failed:",
    e?.message || e
  );
}

// If true, skip all PostHog fetching and use local outputFile.json
const USE_OUTPUTFILE =
  String(process.env.USE_OUTPUTFILE || "false")
    .toLowerCase()
    .trim() === "true";

const NONCLICK_MAX_CLIP_DURATION_MS = 10000; // Max length for a single non-click clip
// Beacon detection and trimming tuning
const BEACON_DETECT_FPS_MAX = Math.max(
  8,
  Number(process.env.BEACON_DETECT_FPS_MAX || 30)
); // analyze up to this fps (higher catches shorter flashes)
const BEACON_EXCLUSION_MARGIN_FRAMES = Math.max(
  0,
  Number(process.env.BEACON_EXCLUSION_MARGIN_FRAMES || 2)
); // frames to trim before/after detected beacons
// Toggle verbose diagnostics for non-click clip event selection
const DEBUG_NOCLICK = true;
// Toggle verbose diagnostics for click clip timing/coverage
const DEBUG_CLICK = true;

async function processInParallel(items, processor, concurrency = 1) {
  const queue = [...items];
  const results = [];
  const promises = [];

  const runTask = async () => {
    while (queue.length > 0) {
      const item = queue.shift();
      if (item) {
        try {
          const result = await processor(item);
          results.push(result);
        } catch (error) {
          console.error(`Error processing item: ${item}`, error);
          results.push({ error: `Failed to process ${item}` });
        }
      }
    }
  };

  for (let i = 0; i < concurrency; i++) {
    promises.push(runTask());
  }

  await Promise.all(promises);
  return results;
}

dotenv.config();

const INCLUDE_CLICK_POSITION_PROMPT =
  process.env.INCLUDE_CLICK_POSITION_PROMPT === "true";
const INCLUDE_CLIP_LOG_PROMPT = process.env.INCLUDE_CLIP_LOG_PROMPT === "true";
// Controls whether to run AI analysis for noclick clips.
// If set to "false", we skip AI for noclick clips and emit the raw clip logs inside </segment>.
const INCLUDE_NOCLICK_CLIPS = process.env.INCLUDE_NOCLICK_CLIPS !== "false";
// If true, generate noclick coverage from gaps between click clips (log-only, no video files).
const NOCLICK_FROM_GAPS = process.env.NOCLICK_FROM_GAPS === "true";
// Controls whether beacon cutting should emit noclick segments.
// Set to "false" to skip generating noclick clips while keeping click clips.
const INCLUDE_NOCLICK_SEGMENTS =
  process.env.INCLUDE_NOCLICK_SEGMENTS !== "false";
const INCLUDE_ENTIRE_LOG_PROMPT =
  process.env.INCLUDE_ENTIRE_LOG_PROMPT === "true";
// If true, assume per-event clips_folder already exist on disk and skip
// main session recording and beacon-based splitting. The pipeline
// starts from analyzing existing clips_folder and generating any missing
// processed outputs.
const START_FROM_RECORDED_CLIPS =
  process.env.START_FROM_RECORDED_CLIPS === "true";
const USE_RELATIVE_TIME = process.env.USE_RELATIVE_TIME === "true";

// Set the application timezone to UTC to ensure all date operations are consistent.
process.env.TZ = "UTC";

// --- Database Configuration ---
const pool = new Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_DATABASE,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,

});

// --- Decryption Configuration ---
// IMPORTANT: Ensure these values match your encryption setup

const storage = new Storage();
const bucketName = process.env.GCS_BUCKET_NAME;

const bucket = storage.bucket(bucketName);
/**
 * Build a public GCS URL for a clip file stored under our standard layout.
 * Layout: clips_folder/<sessionId>/clips_folder/<fileName>
 */
function getGcsClipUrl(sessionId, fileName) {
  try {
    if (!bucketName || !sessionId || !fileName) return null;
    const key = `clips_folder/${sessionId}/clips_folder/${fileName}`;
    return `https://storage.googleapis.com/${bucketName}/${key}`;
  } catch (_) {
    return null;
  }
}
const BATCH_CONTEXT_VERSION = 1;

function mkEventId(entry) {
  if (!entry) {
    return "unknown";
  }
  return `${entry.originalAbsoluteMs || entry.adjustedMs}_${entry.text}_${entry.source}`;
}

function isEventFilterable(entry) {
  if (!entry || !entry.text || entry.isPreRecording || entry.isReplayMarker) {
    return false;
  }
  const isRrwebEvent =
    entry.source === "rrweb-input" ||
    entry.source === "rrweb-navigation" ||
    entry.source === "rrweb-scroll" ||
    (entry.eventIndex && entry.eventIndex.toString().startsWith("rrweb-"));
  const isMatchedNavigationEvent = entry.source === "matched-navigation";
  const isSystemEvent =
    entry.text.includes("Session ended") ||
    entry.text.includes("Page view:") ||
    entry.text.includes("Page leave:");
  const isInactivityOverlay = entry.source === "inactivity-overlay";
  const isAIAnalyzedClick = entry.text.includes("/segment>");
  return (
    isRrwebEvent ||
    isMatchedNavigationEvent ||
    isSystemEvent ||
    isInactivityOverlay ||
    isAIAnalyzedClick
  );
}

function inferClipsDir(finalClips = [], clipLogsFilePath = null) {
  const clipWithPath =
    Array.isArray(finalClips) && finalClips.length > 0
      ? finalClips.find((clip) => clip && clip.clipPath)
      : null;
  if (clipWithPath?.clipPath) {
    return path.dirname(clipWithPath.clipPath);
  }
  if (clipLogsFilePath) {
    return path.dirname(clipLogsFilePath);
  }
  return path.join(
    process.cwd(),
    "clips_folder"
  );
}

function materializeClickAnalysisResult({ metadata, responseText }) {
  if (!metadata) {
    throw new Error("[CLIP-RESULT] Missing metadata for click analysis result.");
  }

  const {
    eventIndex,
    clipStartMs,
    clipEndMs,
    clipEventLog,
    clipEventIds,
    currentPageResolved,
    shouldAddCurrentPageInstruction,
    timestampShiftMs,
    rightClickCount,
    clipPath,
  } = metadata;

  let contextDescription = (responseText || "").trim();
  if (contextDescription && Number.isFinite(timestampShiftMs) && timestampShiftMs !== 0) {
    try {
      contextDescription = shiftTimestampsInText(contextDescription, timestampShiftMs);
    } catch (error) {
      console.warn(
        `[CLIP-RESULT] Failed to shift timestamps for event ${eventIndex}:`,
        error?.message || error
      );
    }
  }

  const pageInstruction = currentPageResolved
    ? shouldAddCurrentPageInstruction
      ? `<current_page>${currentPageResolved}</current_page>\n\n`
      : `<initial_page>${currentPageResolved}</initial_page>\n\n`
    : "";

  const reviewedText = `\n${pageInstruction}${contextDescription}\n</segment>`;

  return {
    eventIndex,
    reviewedText,
    contextDescription,
    clipStartMs,
    clipEndMs,
    clipEventLog,
    clipEventIds,
    currentPageResolved,
    shouldAddCurrentPageInstruction,
    rightClickCount,
    clipPath,
  };
}

function sanitizeForGcsComponent(value, fallback = "value") {
  const safeValue = String(value ?? fallback).replace(/[^a-zA-Z0-9/_-]/g, "_");
  return safeValue.length > 0 ? safeValue : fallback;
}

function buildGeminiBatchOutputUri(
  sessionId,
  analysisId,
  suffix = "session"
) {
  const targetBucketName = bucket?.name || bucketName;
  if (!targetBucketName) {
    throw new Error(
      "GCS_BUCKET_NAME is not configured; cannot determine batch output path."
    );
  }
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const safeSession = sanitizeForGcsComponent(sessionId, "session");
  const safeAnalysis = sanitizeForGcsComponent(analysisId, "analysis");
  const safeSuffix = sanitizeForGcsComponent(suffix, "job");
  return `gs://${targetBucketName}/gemini-batch/${safeSuffix}/${safeSession}/${safeAnalysis}-${timestamp}`;
}

function buildGeminiBatchDisplayName(
  sessionId,
  analysisId,
  suffix = "session"
) {
  const baseName = `${sanitizeForGcsComponent(
    suffix,
    "job"
  )}-${sanitizeForGcsComponent(sessionId, "session")}-${sanitizeForGcsComponent(
    analysisId,
    "analysis"
  )}`;
  return baseName.length <= 63 ? baseName : baseName.slice(-63);
}

// -------------------------
// Subtitle (mov_text) helpers
// -------------------------

function toSrtTimestamp(seconds) {
  const totalMs = Math.max(0, Math.round(seconds * 1000));
  const h = Math.floor(totalMs / 3600000)
    .toString()
    .padStart(2, "0");
  const m = Math.floor((totalMs % 3600000) / 60000)
    .toString()
    .padStart(2, "0");
  const s = Math.floor((totalMs % 60000) / 1000)
    .toString()
    .padStart(2, "0");
  const ms = (totalMs % 1000).toString().padStart(3, "0");
  return `${h}:${m}:${s},${ms}`;
}

function fromSrtTimestamp(ts) {
  // HH:MM:SS,mmm
  const m = /^(\d{2}):(\d{2}):(\d{2}),(\d{3})$/.exec(String(ts).trim());
  if (!m) return null;
  const h = parseInt(m[1], 10) || 0;
  const min = parseInt(m[2], 10) || 0;
  const s = parseInt(m[3], 10) || 0;
  const ms = parseInt(m[4], 10) || 0;
  return h * 3600 + min * 60 + s + ms / 1000;
}

function generateBeaconId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  // Fallback: 32-hex random
  return crypto.randomBytes(16).toString("hex");
}

function deterministicBeaconId(sessionId, seq, kind = "y") {
  try {
    const h = crypto.createHash("sha1");
    h.update(String(sessionId || "no-session"));
    h.update(":");
    h.update(String(kind));
    h.update(":");
    h.update(String(seq));
    return h.digest("hex").slice(0, 20); // 20 hex chars (~80 bits) is plenty
  } catch (_) {
    // Fallback to random if hashing not available
    return generateBeaconId();
  }
}

async function writeBeaconSrt(beacons, srtPath, sessionId) {
  // Only include yellow beacons for logical pause markers
  const yellow = (beacons || []).filter((b) => b && b.color === "yellow");
  let index = 1;
  const lines = [];
  const idMap = [];
  let seq = 0;
  for (const b of yellow) {
    const start = Number(b.startTime);
    const end = Number(b.endTime);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      continue;
    }
    // Deterministic per-session, per-order id to match rrweb marker IDs
    const id = deterministicBeaconId(sessionId, seq++);
    const payload = {
      beacon_id: id,
      color: "yellow",
      session_id: sessionId || null,
      start: start,
      end: end,
    };
    lines.push(String(index++));
    lines.push(`${toSrtTimestamp(start)} --> ${toSrtTimestamp(end)}`);
    lines.push(JSON.stringify(payload));
    lines.push("");
    idMap.push({ id, start, end, color: "yellow" });
  }
  await fs.writeFile(srtPath, lines.join(os.EOL), "utf-8");
  return idMap;
}

async function muxSrtIntoMp4(inputPath, srtPath, outputPath) {
  // Add SRT as mov_text subtitle, clear default disposition, tag track
  // Maps: keep all streams from input, add srt as new subtitle
  const ffmpegCommand = ffmpegStatic || "ffmpeg";
  const args = [
    "-y",
    "-i",
    inputPath,
    "-i",
    srtPath,
    "-map",
    "0",
    "-map",
    "1:0",
    "-c:v",
    "copy",
    "-c:a",
    "copy",
    "-c:s",
    "mov_text",
    "-metadata:s:s:0",
    "title=beacons",
    "-metadata:s:s:0",
    "language=zxx",
    "-disposition:s:0",
    "0",
    outputPath,
  ];
  const { status, stderr } = spawnSync(ffmpegCommand, args, {
    encoding: "utf-8",
  });
  if (status !== 0) {
    throw new Error(`Failed to mux subtitles: ${stderr || "ffmpeg error"}`);
  }
}

// Variant that does not rescan the video: uses provided beacons
async function embedBeaconSubtitlesForKnownBeacons(
  videoPath,
  beacons,
  sessionId
) {
  try {
    const resolved = path.resolve(videoPath);
    const { dir, name, ext } = path.parse(resolved);
    const tempMuxInput = path.join(dir, `${name}.pre_sub${ext || ".mp4"}`);
    const tempSrtPath = path.join(dir, `${name}_beacons.srt`);

    const idMap = await writeBeaconSrt(beacons || [], tempSrtPath, sessionId);
    if (!idMap || idMap.length === 0) {
      try {
        await fs.unlink(tempSrtPath);
      } catch (_) {}
      return { added: false, reason: "no_yellow" };
    }

    await fs.rename(resolved, tempMuxInput);
    try {
      await muxSrtIntoMp4(tempMuxInput, tempSrtPath, resolved);
    } catch (e) {
      try {
        await fs.rename(tempMuxInput, resolved);
      } catch (_) {}
      try {
        await fs.unlink(tempSrtPath);
      } catch (_) {}
      throw e;
    }
    try {
      await fs.unlink(tempMuxInput);
    } catch (_) {}
    try {
      await fs.unlink(tempSrtPath);
    } catch (_) {}
    return { added: true, count: idMap.length, idMap };
  } catch (e) {
    console.warn(
      `[SUB-MUX] Failed to embed beacon subtitles (known): ${e?.message || e}`
    );
    return { added: false, error: e?.message || String(e) };
  }
}
async function extractBeaconSubtitleCues(videoPath) {
  // Use ffprobe to discover subtitle stream(s); prefer one titled "beacons".
  const ffprobeCommand = ffprobeStatic?.path || "ffprobe";
  try {
    const probe = spawnSync(
      ffprobeCommand,
      [
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_streams",
        "-print_format",
        "json",
        videoPath,
      ],
      { encoding: "utf-8" }
    );
    if (probe.status !== 0) return [];
    const data = JSON.parse(probe.stdout || "{}");
    const streams = Array.isArray(data.streams) ? data.streams : [];
    if (streams.length === 0) return [];
    let selectedRelIndex = 0; // relative subtitle index
    for (let i = 0; i < streams.length; i++) {
      const st = streams[i];
      const title = (st.tags && (st.tags.title || st.tags.handler_name)) || "";
      if (String(title).toLowerCase().includes("beacons")) {
        selectedRelIndex = i; // relative within subtitle selection
        break;
      }
    }
    // Extract the selected subtitle stream as SRT to a temp file
    const { dir, name } = path.parse(videoPath);
    const srtOut = path.join(dir, `${name}_extracted.srt`);
    const ffmpegCommand = ffmpegStatic || "ffmpeg";
    const dump = spawnSync(
      ffmpegCommand,
      ["-y", "-i", videoPath, "-map", `0:s:${selectedRelIndex}`, srtOut],
      { encoding: "utf-8" }
    );
    if (dump.status !== 0) {
      try {
        await fs.unlink(srtOut);
      } catch (_) {}
      return [];
    }
    const raw = await fs.readFile(srtOut, "utf-8");
    try {
      await fs.unlink(srtOut);
    } catch (_) {}
    // Parse SRT cues
    const blocks = raw.replace(/\r\n/g, "\n").split(/\n\s*\n/);
    const cues = [];
    for (const block of blocks) {
      const lines = block.trim().split("\n");
      if (lines.length < 2) continue;
      // Handle optional index line
      let tLine = lines[0];
      if (/^\d+$/.test(lines[0]) && lines[1]?.includes("-->")) {
        tLine = lines[1];
        lines.splice(0, 2); // remove index + timing
      } else if (lines[0].includes("-->")) {
        lines.splice(0, 1); // remove timing
      } else {
        continue;
      }
      const tm = /([^\s]+)\s+-->\s+([^\s]+)/.exec(tLine);
      if (!tm) continue;
      const start = fromSrtTimestamp(tm[1]);
      const end = fromSrtTimestamp(tm[2]);
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
      const text = lines.join("\n").trim();
      let id = null;
      try {
        const obj = JSON.parse(text);
        if (obj && obj.beacon_id) id = String(obj.beacon_id);
      } catch (_) {
        const m = /beacon_id\s*=\s*([A-Za-z0-9_-]+)/.exec(text);
        if (m) id = m[1];
      }
      cues.push({ start, end, text, id });
    }
    return cues;
  } catch (_) {
    return [];
  }
}

/**
 * Injects pauses into the rrweb event stream after yellow/blue clip markers.
 * @param {Array<Object>} events - The rrweb event stream (with markers).
 * @param {number} pauseDurationMs - The duration of the pause to inject in milliseconds.
 * @returns {Array<Object>} A new event stream with timestamps shifted to create pauses.
 */
function addPausesForMarkers(events, pauseDurationMs = 2000) {
  console.log(
    `[PAUSE_INJECT] Adding pauses of ${pauseDurationMs}ms for yellow/blue markers.`
  );
  const newEvents = [];
  let timeShift = 0;
  let markerCount = 0;

  for (const event of events) {
    const newEvent = JSON.parse(JSON.stringify(event)); // Deep copy to avoid side effects
    newEvent.timestamp += timeShift;

    const isPauseMarker =
      newEvent.type === 5 &&
      newEvent.data?.tag === "CLIP_MARKER" &&
      newEvent.data?.payload?.color === "yellow";

    if (isPauseMarker) {
      // Apply a pause BEFORE the marker: shift the marker itself
      newEvent.timestamp += pauseDurationMs;
      // Apply an additional pause AT the marker: shift subsequent events again
      timeShift += pauseDurationMs * 2;
      markerCount++;
     
    }
    newEvents.push(newEvent);
  }

  return newEvents;
}

/**
 * Defines all clip segments (click and no-click) for an entire session based on a purely logical
 * analysis of rrweb event timestamps. This version faithfully recreates the original, correct logic
 * for calculating fixed-duration click clips_folder and truncating them only when they would contain a subsequent click.
 *
 * @param {Array<Object>} rrwebMatches - The array of click matches from `createClipsFromRrwebClicks`.
 * @param {number} totalLogicalDurationSeconds - The total duration of the rrweb session in seconds.
 * @returns {{allClips: Array<{start: number, end: number, type: 'click' | 'noclick', originalEvent?: object}>}}
 */
function defineLogicalClipSegments(rrwebMatches, totalLogicalDurationSeconds) {
  console.log(
    "Defining clip segments with synchronized logical/visual boundaries (v5)..."
  );
  console.log(`[DEBUG] Received ${rrwebMatches?.length || 0} rrweb matches.`);

  const FREEZE_DURATION_MS = 3900;

  if (!rrwebMatches || rrwebMatches.length === 0) {
    const result = {
      allClips: [
        {
          start: 0,
          end: totalLogicalDurationSeconds,
          logicalStart: 0,
          logicalEnd: totalLogicalDurationSeconds,
          type: "noclick",
        },
      ],
      inflatedTotalDurationSeconds: totalLogicalDurationSeconds,
    };
    console.log(
      "No click events found. Treating entire clip as a single no-click clip."
    );
    return result;
  }

  // 1. Map logical click times to their inflated visual times.
  const sortedMatches = [...rrwebMatches].sort(
    (a, b) => a.videoTimeSeconds - b.videoTimeSeconds
  );
  let cumulativeInflationMs = 0;
  const timelineMap = sortedMatches.map((match) => {
    const logicalClickTimeMs = match.videoTimeSeconds * 1000;
    const visualClickTimeMs = logicalClickTimeMs + cumulativeInflationMs;
    const result = {
      ...match,
      logicalClickTimeSeconds: match.videoTimeSeconds,
      visualClickTimeSeconds: visualClickTimeMs / 1000,
    };
    cumulativeInflationMs += FREEZE_DURATION_MS;
    return result;
  });

  const inflatedTotalDurationSeconds =
    totalLogicalDurationSeconds + cumulativeInflationMs / 1000;
  console.log(
    `Original logical duration: ${totalLogicalDurationSeconds.toFixed(
      2
    )}s. Inflated visual duration: ${inflatedTotalDurationSeconds.toFixed(2)}s.`
  );

  // 2. Calculate clip windows on the inflated visual timeline.
  const clickWindows = [];
  const freezeSeconds = FREEZE_DURATION_MS / 1000;
  for (let i = 0; i < timelineMap.length; i++) {
    const mappedMatch = timelineMap[i];
    const visualClickTime = mappedMatch.visualClickTimeSeconds;
    const logicalClickTime = mappedMatch.logicalClickTimeSeconds;

    console.log(
      `[DEBUG] Processing click #${i}: logicalClickTime=${logicalClickTime.toFixed(
        3
      )}s, visualClickTime=${visualClickTime.toFixed(3)}s`
    );

    const intermediateClipDurationSeconds =
      (CLIP_DURATION_MS + FREEZE_DURATION_MS) / 1000;

    // Calculate start times on both timelines
    const gapToPreviousMs =
      i === 0
        ? Number.POSITIVE_INFINITY
        : Math.max(
            0,
            (visualClickTime -
              timelineMap[i - 1].visualClickTimeSeconds -
              freezeSeconds) *
              1000
          );
    const preBufferMs = computeInterClickBufferMs(gapToPreviousMs);
    const visualStart = Math.max(0, visualClickTime - preBufferMs / 1000);
    const logicalStart = Math.max(0, logicalClickTime - preBufferMs / 1000);

    let visualEnd = visualStart + intermediateClipDurationSeconds;
    let logicalEnd = logicalStart + CLIP_DURATION_MS / 1000;

    console.log(
      `[DEBUG] Initial clip window #${i}: logical=[${logicalStart.toFixed(
        3
      )}s, ${logicalEnd.toFixed(3)}s], visual=[${visualStart.toFixed(
        3
      )}s, ${visualEnd.toFixed(3)}s]`
    );
    /* 
    // Truncate if the window overlaps with the START of the NEXT visual window.
    if (i < timelineMap.length - 1) {
      const nextMappedMatch = timelineMap[i + 1];
      const nextVisualClickTime = nextMappedMatch.visualClickTimeSeconds;
      const nextLogicalClickTime = nextMappedMatch.logicalClickTimeSeconds;

      const nextVisualWindowStart = Math.max(
        0,
        nextVisualClickTime + CLIP_POST_CLICK_DELAY_MS / 1000
      );
      const nextLogicalWindowStart = Math.max(
        0,
        nextLogicalClickTime + CLIP_POST_CLICK_DELAY_MS / 1000
      );

      if (visualEnd > nextVisualWindowStart) {
        const EXCLUSION_GAP_SECONDS = 0.0;

        const newVisualEnd = nextVisualWindowStart - EXCLUSION_GAP_SECONDS;
        const newLogicalEnd = nextLogicalWindowStart - EXCLUSION_GAP_SECONDS;

        console.log(
          `  -> Truncating click clip #${
            mappedMatch.postHogEvent.eventIndex
          } to avoid overlap. End time ${visualEnd.toFixed(
            3
          )}s -> ${newVisualEnd.toFixed(3)}s`
        );
        console.log(
          `[DEBUG] Overlap detected with next click. Truncating clip #${i}. New logical end: ${newLogicalEnd.toFixed(
            3
          )}s, new visual end: ${newVisualEnd.toFixed(3)}s`
        );

        visualEnd = newVisualEnd;
        logicalEnd = newLogicalEnd;
      }
    } */

    if (i < timelineMap.length - 1) {
      const nextMappedMatch = timelineMap[i + 1];
      const nextVisualClickTime = nextMappedMatch.visualClickTimeSeconds;

      // The next clip's freeze sequence will start at its visual click time.
      // We must end the current clip *before* that time.
      const nextFreezeStartTime = nextVisualClickTime;

      if (visualEnd > nextFreezeStartTime) {
        const gapToNextMs = Math.max(
          0,
          (nextVisualClickTime - visualClickTime - freezeSeconds) * 1000
        );
        const postBufferMs = computeInterClickBufferMs(gapToNextMs);
        const targetVisualEnd = nextFreezeStartTime - postBufferMs / 1000;

        console.log(
          `[DEBUG] Overlap detected. Current clip's planned end (${visualEnd.toFixed(
            3
          )}s) is after the next click's freeze begins (${nextFreezeStartTime.toFixed(
            3
          )}s).`
        );
        console.log(
          `[DEBUG] Gap between clicks: ${(gapToNextMs / 1000).toFixed(
            3
          )}s. Buffer before next click: ${postBufferMs}ms.`
        );

        visualEnd = targetVisualEnd;

        // Also adjust the logical end proportionally.
        const visualDuration = visualEnd - visualStart;
        const logicalDuration = visualDuration - FREEZE_DURATION_MS / 1000;
        logicalEnd = logicalStart + Math.max(0, logicalDuration);
        console.log(
          `[DEBUG] Truncating with dynamic buffer (${postBufferMs}ms). New visual end: ${visualEnd.toFixed(
            3
          )}s, new logical end: ${logicalEnd.toFixed(3)}s`
        );
      }
    }

    // **NEW SAFETY NET:** Enforce a minimum visual duration for every clip.
    // This is the key to preventing zero-duration clips_folder.
    const MINIMUM_VISUAL_DURATION_SECONDS = 0.001;
    if (visualEnd - visualStart < MINIMUM_VISUAL_DURATION_SECONDS) {
      console.warn(
        `[DEBUG] Clip #${i} is too short after truncation (${(
          visualEnd - visualStart
        ).toFixed(
          3
        )}s). Enforcing minimum duration of ${MINIMUM_VISUAL_DURATION_SECONDS}s.`
      );
      visualEnd = visualStart + MINIMUM_VISUAL_DURATION_SECONDS;
    }
    // ==================================================================
    // <<< 🎯 END: REPLACEMENT LOGIC 🎯 >>>
    // ==================================================================

    visualEnd = Math.min(visualEnd, inflatedTotalDurationSeconds);
    logicalEnd = Math.min(logicalEnd, totalLogicalDurationSeconds);

    if (visualStart >= visualEnd) continue;

    clickWindows.push({
      start: visualStart,
      end: visualEnd,
      logicalStart: logicalStart,
      logicalEnd: logicalEnd,
      type: "click",
      originalEvent: mappedMatch.postHogEvent,
      rrwebEvent: mappedMatch.rrwebEvent,
    });
  }

  // 3. Define "no-click" segments using the final click window boundaries.
  const allSegments = [];
  let lastVisualEndTime = 0;
  let lastLogicalEndTime = 0;

  for (const clickWindow of clickWindows) {
    if (clickWindow.start > lastVisualEndTime) {
      allSegments.push({
        start: lastVisualEndTime,
        end: clickWindow.start,
        logicalStart: lastLogicalEndTime,
        logicalEnd: clickWindow.logicalStart,
        type: "noclick",
      });
    }
    allSegments.push(clickWindow);
    lastVisualEndTime = clickWindow.end;
    lastLogicalEndTime = clickWindow.logicalEnd;
  }

  if (lastVisualEndTime < inflatedTotalDurationSeconds) {
    allSegments.push({
      start: lastVisualEndTime,
      end: inflatedTotalDurationSeconds,
      logicalStart: lastLogicalEndTime,
      logicalEnd: totalLogicalDurationSeconds,
      type: "noclick",
    });
  }

  /*   const finalClips = allSegments.filter((clip) => clip.end - clip.start >= 0.1); */
  const finalClips = allSegments;
  console.log(
    `Logically defined ${finalClips.length} total segments with synchronized boundaries.`
  );

  return { allClips: finalClips };
}

/**
 * Gets the text content of a node at a specific point in time by looking at rrweb events
 * @param {number} nodeId - The node ID to get text for
 * @param {number} timestamp - The timestamp to get the text at
 * @param {Array} events - All rrweb events
 * @returns {string|null} The text content at that time, or null if not found
 */
function getTextContentAtTime(nodeId, timestamp, events) {
  // First, find the initial text from FullSnapshot
  let textContent = null;

  // Find the most recent FullSnapshot before the timestamp
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (event.timestamp > timestamp) continue;

    if (event.type === EventType.FullSnapshot) {
      // Search for the node in the snapshot
      const findNodeText = (node) => {
        if (node.id === nodeId) {
          // For text nodes, return textContent directly
          if (node.type === 3) return node.textContent || "";
          // For element nodes, concatenate all child text nodes
          let text = "";
          if (node.childNodes) {
            for (const child of node.childNodes) {
              if (child.type === 3 && child.textContent) {
                text += child.textContent;
              }
            }
          }
          return text;
        }
        if (node.childNodes) {
          for (const child of node.childNodes) {
            const found = findNodeText(child);
            if (found !== null) return found;
          }
        }
        return null;
      };

      textContent = findNodeText(event.data.node);
      if (textContent !== null) break;
    }
  }

  // Then apply any text mutations that happened before the timestamp
  for (const event of events) {
    if (event.timestamp > timestamp) break;

    if (
      event.type === EventType.IncrementalSnapshot &&
      event.data.source === IncrementalSource.Mutation
    ) {
      // Check text mutations
      if (event.data.texts) {
        for (const text of event.data.texts) {
          if (text.id === nodeId) {
            textContent = text.value || "";
          }
        }
      }

      // Check if node was added with text
      if (event.data.adds) {
        for (const add of event.data.adds) {
          if (add.node && add.node.id === nodeId) {
            if (add.node.type === 3) {
              textContent = add.node.textContent || "";
            } else if (add.node.childNodes) {
              // For element nodes, get text from child text nodes
              let text = "";
              for (const child of add.node.childNodes) {
                if (child.type === 3 && child.textContent) {
                  text += child.textContent;
                }
              }
              textContent = text;
            }
          }
        }
      }

      // Check if node was removed (text becomes null)
      if (event.data.removes) {
        for (const remove of event.data.removes) {
          if (remove.id === nodeId) {
            textContent = null;
          }
        }
      }
    }
  }

  return textContent;
}

// Helper: return clip duration in seconds using ffprobe
function getVideoDurationSecondsSync(videoPath) {
  try {
    const ffprobeCommand = ffprobeStatic?.path || "ffprobe";
    const probeResult = spawnSync(
      ffprobeCommand,
      [
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        videoPath,
      ],
      { encoding: "utf-8" }
    );
    if (probeResult.status !== 0 || !probeResult.stdout) return null;
    const secs = parseFloat(String(probeResult.stdout).trim());
    return Number.isFinite(secs) ? secs : null;
  } catch (_) {
    return null;
  }
}
async function uploadToGCS(localFilePath, destinationFileName) {
  try {
    console.log(
      `Starting upload of "${localFilePath}" to GCS bucket "${bucketName}" as "${destinationFileName}"...`
    );

    await bucket.upload(localFilePath, {
      destination: destinationFileName,
    });
  } catch (error) {
    console.error(`Failed to upload file to GCS:`, error);
    throw error; // Re-throw the error to be caught by the calling function
  }
}

async function findBeaconFrames_TrueStream(videoPath) {
  console.log(
    `[TRUE-STREAM] Analyzing for beacon frames in: ${path.basename(videoPath)}`
  );
  const ffmpegCommand = ffmpegStatic || "ffmpeg";
  const ffprobeCommand = ffprobeStatic?.path || "ffprobe";
  const resolvedVideoPath = path.resolve(videoPath);

  return new Promise(async (resolve, reject) => {
    try {
      // 1. Get source clip properties (frame rate and dimensions)
      const probeResult = spawnSync(
        ffprobeCommand,
        [
          "-v",
          "error",
          "-select_streams",
          "v:0",
          "-show_entries",
          "stream=r_frame_rate,width,height",
          "-of",
          "default=noprint_wrappers=1:nokey=1",
          resolvedVideoPath,
        ],
        { encoding: "utf-8" }
      );

      if (probeResult.status !== 0 || !probeResult.stdout) {
        return reject(
          new Error(
            `ffprobe failed to get clip properties. Stderr: ${probeResult.stderr}`
          )
        );
      }

      const [widthStr, heightStr, frameRateStr] = probeResult.stdout
        .trim()
        .split("\n");
      const srcWidth = parseInt(widthStr, 10);
      const srcHeight = parseInt(heightStr, 10);
      const srcFrameRate = Math.round(eval(frameRateStr));

      if (!srcFrameRate || !srcWidth || !srcHeight) {
        return reject(
          new Error(`Invalid clip properties detected: ${probeResult.stdout}`)
        );
      }

      // Downscale and decimate for fast analysis.
      // We only need a coarse average color to detect full-screen overlays.
      const targetFps = Math.min(BEACON_DETECT_FPS_MAX, srcFrameRate); // analyze at up to configured fps
      const targetWidth = 16; // tiny frame to reduce CPU
      const targetHeight = Math.max(
        9,
        Math.round((srcHeight / srcWidth) * targetWidth)
      );

      const secondsPerFrame = 1 / targetFps;
      const frameSize = targetWidth * targetHeight * 3; // 3 bytes (R, G, B) per pixel

      // 2. Use ASYNCHRONOUS spawn to pipe tiny raw clip frames
      //    Apply fps decimation and scaling inside ffmpeg for huge speedups.
      const args = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        resolvedVideoPath,
        "-vf",
        `fps=${targetFps},scale=${targetWidth}:${targetHeight}`,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
      ];

      const ffmpegProcess = spawn(ffmpegCommand, args);

      let buffer = Buffer.alloc(0);
      const beacons = [];
      let currentBeacon = null;
      let frameIndex = 0;
      let errorOutput = "";

      ffmpegProcess.stderr.on("data", (data) => {
        errorOutput += data.toString();
      });

      ffmpegProcess.stdout.on("data", (chunk) => {
        buffer = Buffer.concat([buffer, chunk]);

        // Process as many full frames as we have in the buffer
        while (buffer.length >= frameSize) {
          const frameBuffer = buffer.slice(0, frameSize);
          buffer = buffer.slice(frameSize); // Keep the rest of the buffer

          // --- Analyze the single frame ---
          let totalR = 0,
            totalG = 0,
            totalB = 0;
          for (let p = 0; p < frameBuffer.length; p += 3) {
            totalR += frameBuffer[p];
            totalG += frameBuffer[p + 1];
            totalB += frameBuffer[p + 2];
          }
          const numPixels = targetWidth * targetHeight;
          const r = totalR / numPixels;
          const g = totalG / numPixels;
          const b = totalB / numPixels;
          const frameTime = frameIndex * secondsPerFrame;

          let detectedColor = null;
          if (g > 150 && r < 100 && b < 100) detectedColor = "green";
          else if (r > 150 && g < 100 && b < 100) detectedColor = "red";
          else if (b > 150 && r < 100 && g < 100) detectedColor = "blue";
          else if (r > 150 && g > 150 && b < 100) detectedColor = "yellow";

          if (detectedColor) {
            if (!currentBeacon || currentBeacon.color !== detectedColor) {
              if (currentBeacon) {
                currentBeacon.endTime = frameTime; // Use current frame's start as previous's end
                currentBeacon.endFrameIndex = frameIndex - 1;
                beacons.push(currentBeacon);
              }
              currentBeacon = {
                color: detectedColor,
                startTime: frameTime,
                startFrameIndex: frameIndex,
              };
            }
          } else {
            if (currentBeacon) {
              currentBeacon.endTime = frameTime; // Use current frame's start as previous's end
              currentBeacon.endFrameIndex = frameIndex - 1;
              beacons.push(currentBeacon);
              currentBeacon = null;
            }
          }
          // --- End of frame analysis ---
          frameIndex++;
        }
      });

      ffmpegProcess.on("close", (code) => {
        if (code !== 0) {
          return reject(
            new Error(
              `ffmpeg process exited with code ${code}. Stderr: ${errorOutput}`
            )
          );
        }

        if (currentBeacon) {
          // The stream ended, so the beacon lasts until the end of the last processed frame.
          currentBeacon.endTime = frameIndex * secondsPerFrame;
          currentBeacon.endFrameIndex = frameIndex - 1;
          beacons.push(currentBeacon);
        }

        if (beacons.length > 0) {
          console.log(
            `[TRUE-STREAM] Detected ${beacons.length} beacon segments.`
          );
        } else {
          console.warn(
            `[TRUE-STREAM] No beacons were detected in ${path.basename(
              videoPath
            )}.`
          );
        }

        resolve(beacons);
      });

      ffmpegProcess.on("error", (err) => {
        reject(err);
      });
    } catch (error) {
      console.error(
        `[TRUE-STREAM] A critical error occurred for ${videoPath}:`,
        error
      );
      reject(error);
    }
  });
}

const ALGORITHM = "aes-256-gcm";
const IV_LENGTH = 16;
const TAG_LENGTH = 16;
const KEY = crypto.scryptSync(process.env.ENCRYPTION_SECRET, process.env.SALT, 32); 
const KEY2 = crypto.scryptSync(process.env.ENCRYPTION_SECRET_2, process.env.SALT, 32); 
const KEY3 = crypto.scryptSync(process.env.ENCRYPTION_SECRET_3, process.env.SALT, 32); 
function decrypt(encryptedText) {
  try {
    console.log("fewkfmwk ", encryptedText);
    const data = Buffer.from(String(encryptedText), "hex");
    const iv = data.slice(0, IV_LENGTH);
    const tag = data.slice(IV_LENGTH, IV_LENGTH + TAG_LENGTH);
    const text = data.slice(IV_LENGTH + TAG_LENGTH);
    console.log("KEY ", KEY);
    const decipher = crypto.createDecipheriv(ALGORITHM, KEY, iv);
    decipher.setAuthTag(tag);
    return decipher.update(text, "binary", "utf8") + decipher.final("utf8");
  } catch (error) {
    try {
      const data = Buffer.from(String(encryptedText), "hex");
      const iv = data.slice(0, IV_LENGTH);
      const tag = data.slice(IV_LENGTH, IV_LENGTH + TAG_LENGTH);
      const text = data.slice(IV_LENGTH + TAG_LENGTH);
      const decipher = crypto.createDecipheriv(ALGORITHM, KEY2, iv);
      decipher.setAuthTag(tag);
      return decipher.update(text, "binary", "utf8") + decipher.final("utf8");
    } catch (error) {
      try {
        const data = Buffer.from(String(encryptedText), "hex");
        const iv = data.slice(0, IV_LENGTH);
        const tag = data.slice(IV_LENGTH, IV_LENGTH + TAG_LENGTH);
        const text = data.slice(IV_LENGTH + TAG_LENGTH);
        const decipher = crypto.createDecipheriv(ALGORITHM, KEY3, iv);
        decipher.setAuthTag(tag);
        return decipher.update(text, "binary", "utf8") + decipher.final("utf8");
      } catch (error) {
        console.error("Decryption failed:", error);
        throw new Error("Failed to decrypt PostHog API key.");
      }
    }
  }
}

/* const GeminiFlash = "models/gemini-2.5-flash-preview-05-20"; */
/* const Gemini = "gemini-2.5-pro"; */
/* const Gemini = "gemini-2.5-flash"; */
let Gemini= "gemini-2.5-flash-preview-09-2025"
let Gemini2= "models/gemini-3-pro-preview"
console.log("ewkmfwekmfw ", process.env.NODE_ENV)
if (process.env.NODE_ENV === "development") {
 /*  Gemini = "gemini-2.5-flash-lite"
  
  
  Gemini2 = "gemini-2.5-flash-lite" */
}

/* function preprocessEventsForSkipWithMessage(
  originalEvents,
  maxInactiveThresholdMs = 5000,
  messageDisplayDurationMs = 1000,
  initialTimeShiftMs = 0
) {
  console.log(
    "ℹ️ Inactivity skipping is disabled. Returning original events without modification."
  );
  // Return the original events and empty arrays for overlays and skips.
  // This effectively disables this inactivity skipping feature.
  return {
    events: originalEvents,
    overlayInstructions: [],
    skips: [],
  };
}
 */

function preprocessEventsForSkipWithMessage(
  originalEvents,
  maxInactiveThresholdMs = 5000,
  messageDisplayDurationMs = 1000,
  initialTimeShiftMs = 0 
) {
  if (!originalEvents || originalEvents.length < 2) {
    return { events: originalEvents, overlayInstructions: [], skips: [] };
  }

  // For clarity and to make this function self-contained, we define
  // the necessary rrweb constants directly within it.
  const EventType = {
    FullSnapshot: 2,
    IncrementalSnapshot: 3,
    Meta: 4,
  };
  const IncrementalSource = {
    MouseInteraction: 2,
    Scroll: 3,
    Input: 5,
    Drag: 10,
  };
  // Keep these for the existing lookahead logic which uses them directly
  const RRWEB_EVENT_TYPE_FULL_SNAPSHOT = EventType.FullSnapshot;
  const RRWEB_EVENT_TYPE_META = EventType.Meta;

  function formatMsToSkippedTime(ms) {
    if (ms < 1000) return `${ms}ms`;
    let seconds = Math.floor(ms / 1000);
    let minutes = Math.floor(seconds / 60);
    let hours = Math.floor(minutes / 60);
    minutes = minutes % 60;
    seconds = seconds % 60;
    let parts = [];
    if (hours > 0) parts.push(`${hours} hour${hours > 1 ? "s" : ""}`);
    if (minutes > 0) parts.push(`${minutes} minute${minutes > 1 ? "s" : ""}`);
    if (seconds > 0 || (hours === 0 && minutes === 0)) {
      parts.push(`${seconds} second${seconds !== 1 ? "s" : ""}`);
    }
    return parts.length > 0 ? parts.join(" ") : "A moment";
  }

  let processedEvents = [];
  let overlayInstructions = [];
  let accumulatedTimeShift = initialTimeShiftMs;
  let skips = [];
  // Safety margin to avoid overlay hide colliding exactly with anchor/meta/snapshot events
  const OVERLAY_EDGE_SAFETY_MS = 80; // ~5 frames at 60fps
  processedEvents.push({
    ...originalEvents[0],
    originalTimestamp: originalEvents[0].timestamp,
  }); // Add the first event as is

  for (let i = 1; i < originalEvents.length; i++) {
    const prevOriginalEvent = originalEvents[i - 1];
    const currentOriginalEvent = originalEvents[i];
    const prevShiftedEventTimestamp =
      processedEvents[processedEvents.length - 1].timestamp;
    const originalGapMs =
      currentOriginalEvent.timestamp - prevOriginalEvent.timestamp;

    // A "significant interaction" is a user-initiated event that we don't want to
    // preface with a "skipped inactivity" message. For example, a user might
    // read for 10 seconds and then click. We shouldn't say we skipped 10 seconds of inactivity.
    // IMPORTANT: Look ahead in the next 2 seconds to find significant interactions,
    // since clicks often come after mouse movements.
    let isSignificantInteraction = false;
    let significantInteractionEvent = null;
    const LOOKAHEAD_WINDOW_MS = 2000; // Look ahead 2 seconds for interactions

    for (let j = i; j < originalEvents.length; j++) {
      const lookaheadEvent = originalEvents[j];
      const timeAfterGap =
        lookaheadEvent.timestamp - currentOriginalEvent.timestamp;

      if (timeAfterGap > LOOKAHEAD_WINDOW_MS) break; // Stop looking beyond 2s window

      if (
        lookaheadEvent.type === EventType.IncrementalSnapshot &&
        lookaheadEvent.data &&
        (lookaheadEvent.data.source === IncrementalSource.MouseInteraction ||
          lookaheadEvent.data.source === IncrementalSource.Scroll ||
          lookaheadEvent.data.source === IncrementalSource.Input ||
          lookaheadEvent.data.source === IncrementalSource.Drag)
      ) {
        isSignificantInteraction = true;
        significantInteractionEvent = lookaheadEvent;
        console.log(
          `[INTERACTION-DEBUG] Found significant interaction ${timeAfterGap}ms after gap: type=${lookaheadEvent.type}, source=${lookaheadEvent.data.source}`
        );
        break;
      }
    }

    if (originalGapMs > maxInactiveThresholdMs) {
      console.log(
        `[INTERACTION-DEBUG] Gap detected (${originalGapMs}ms). Event after gap: type=${currentOriginalEvent.type}, source=${currentOriginalEvent.data?.source}, isSignificant=${isSignificantInteraction}`
      );
      console.log(
        `[INTERACTION-DEBUG] Expected values: IncrementalSnapshot=3, MouseInteraction=2, Scroll=3, Input=5, Drag=10`
      );
      // Determine how much time to keep at the end of the gap.
      // For significant interactions, we keep 2 seconds to show context before the action.
      // For non-significant, we keep `messageDisplayDurationMs` for the overlay.
      /*  const keepDurationMs = isSignificantInteraction
        ? 2000
        : messageDisplayDurationMs; */

      const keepDurationMs = messageDisplayDurationMs;

      if (originalGapMs > keepDurationMs) {
        const timeToEffectivelyRemove = originalGapMs - keepDurationMs;

        // The rest of the logic for skipping remains mostly the same.
        // We just won't show an overlay for significant interactions.
        accumulatedTimeShift += timeToEffectivelyRemove;

        skips.push({
          startTime: prevOriginalEvent.timestamp,
          endTime: currentOriginalEvent.timestamp,
          timeRemoved: timeToEffectivelyRemove,
        });

        const newTimestampForCurrentEvent =
          currentOriginalEvent.timestamp - accumulatedTimeShift;

        // Always show the "inactivity skipped" message for any skipped gap
        // Look ahead for a significant event that indicates the UI is about to change drastically.
        let significantEvent = null;
        for (let j = i; j < originalEvents.length; j++) {
          const eventType = originalEvents[j].type;
          if (
            eventType === RRWEB_EVENT_TYPE_FULL_SNAPSHOT ||
            eventType === RRWEB_EVENT_TYPE_META
          ) {
            const timeAfterGap =
              originalEvents[j].timestamp - currentOriginalEvent.timestamp;
            if (timeAfterGap < 1000) {
              significantEvent = originalEvents[j];
              break;
            }
          }
        }
        const anchorEvent = significantEvent || currentOriginalEvent;
        const newTimestampForAnchorEvent =
          anchorEvent.timestamp - accumulatedTimeShift;

        // Always show an overlay; duration should follow configured display duration
        const overlayDurationMs = messageDisplayDurationMs;
        const desiredOverlayShow =
          newTimestampForAnchorEvent - overlayDurationMs;
        // Avoid starting before the previous shifted event; clamp forward if needed
        const overlayShowTimestamp = Math.max(
          prevShiftedEventTimestamp + 16, // ensure not the exact same ms as prior event
          desiredOverlayShow
        );
        // Hide slightly before the anchor to avoid heavy UI change at the same millisecond
        let overlayHideTimestamp = Math.min(
          newTimestampForAnchorEvent - OVERLAY_EDGE_SAFETY_MS,
          overlayShowTimestamp + overlayDurationMs
        );

        // If this overlay precedes a significant interaction (click/scroll/input/drag),
        // ensure it ends at least 450ms before the interaction to prevent overlays
        // from appearing mid-clip (clips start ~350ms before interactions)
        if (isSignificantInteraction && significantInteractionEvent) {
          const minGapBeforeInteractionMs = 450;
          // Calculate the adjusted timestamp for the significant interaction event
          const adjustedInteractionTimestamp =
            significantInteractionEvent.timestamp - accumulatedTimeShift;
          const maxAllowedHideTime =
            adjustedInteractionTimestamp - minGapBeforeInteractionMs;
          const originalHideTime = overlayHideTimestamp;
          const gapBeforeInteraction =
            adjustedInteractionTimestamp - overlayHideTimestamp;

          console.log(
            `[OVERLAY-SHORTEN] Detected significant interaction (source=${significantInteractionEvent.data?.source}) at adjusted time ${adjustedInteractionTimestamp}ms`
          );
          console.log(
            `[OVERLAY-SHORTEN] Original overlay hide time: ${originalHideTime}ms (gap: ${gapBeforeInteraction}ms before interaction)`
          );

          if (overlayHideTimestamp > maxAllowedHideTime) {
            overlayHideTimestamp = maxAllowedHideTime;
            console.log(
              `[OVERLAY-SHORTEN] ⚠️ Overlay too close to interaction! Adjusting hide time from ${originalHideTime}ms to ${overlayHideTimestamp}ms`
            );
            console.log(
              `[OVERLAY-SHORTEN] New gap before interaction: ${minGapBeforeInteractionMs}ms (prevents mid-clip overlay)`
            );
          } else {
            console.log(
              `[OVERLAY-SHORTEN] ✓ Overlay already ends with sufficient gap (${gapBeforeInteraction}ms >= ${minGapBeforeInteractionMs}ms). No adjustment needed.`
            );
          }
        }

        const skipMessageText = `${formatMsToSkippedTime(
          originalGapMs
        )} of inactivity collapsed (Mention the collapsed time (${formatMsToSkippedTime(
          originalGapMs
        )}) in the output)`;
        overlayInstructions.push({
          showAt: overlayShowTimestamp,
          hideAt: overlayHideTimestamp,
          message: skipMessageText,
        });

        processedEvents.push({
          ...currentOriginalEvent,
          originalTimestamp: currentOriginalEvent.timestamp,
          timestamp: newTimestampForCurrentEvent,
        });
      } else {
        // Gap is not long enough to be skipped, treat as a normal event.
        processedEvents.push({
          ...currentOriginalEvent,
          originalTimestamp: currentOriginalEvent.timestamp,
          timestamp: currentOriginalEvent.timestamp - accumulatedTimeShift,
        });
      }
    } else {
      // No gap, just apply accumulated shift.
      processedEvents.push({
        ...currentOriginalEvent,
        originalTimestamp: currentOriginalEvent.timestamp, // Preserve original timestamp
        timestamp: currentOriginalEvent.timestamp - accumulatedTimeShift,
      });
    }
  }
  return { events: processedEvents, overlayInstructions, skips };
}

const DEFAULT_OUTPUT_VIDEO_PATH = "puppeteer_output3.mp4"; // Default output path

// const RRWEB_PLAYER_CDN = './rrweb-player.min.js'; // Changed to local path
// const RRWEB_PLAYER_CSS_CDN = './rrweb-player.min.css';
const RRWEB_PLAYER_CDN =
  "https://cdn.jsdelivr.net/npm/rrweb-player@latest/dist/index.js";
const RRWEB_PLAYER_CSS_CDN =
  "https://cdn.jsdelivr.net/npm/rrweb-player@latest/dist/style.css";
const INACTIVITY_THRESHOLD_MS = 5000; // e.g., Skip gaps longer than 5 seconds
const SKIP_MESSAGE_DURATION_MS = 1000; // Keep overlay at 1s; stability handled via safety margins
const CLIP_POST_CLICK_DELAY_MS = -350; // Start clip 5s BEFORE click to show pre-click state
const CLIP_DURATION_MS = 9050; // Total clip duration in milliseconds (5s before + 5s after click)

function computeInterClickBufferMs(gapMs) {
  // Map inter-click gaps to a shared buffer so adjacent clips have matching padding.
  if (!Number.isFinite(gapMs)) return 2000;
  if (gapMs >= 6000) return 2000;
  if (gapMs >= 4000) return 1500;
  if (gapMs >= 3000) return 1000;
  if (gapMs >= 2000) return 750;
  if (gapMs >= 1500) return 500;
  if (gapMs >= 350) return 350;
  return 0;
}

/**
 * This file contains the logic to dynamically determine clip recording settings
 * based on clip duration to keep the final.fil;f.f; file size under a specific target.
 */

function getRecordingSettings(durationInSeconds) {
  // Map duration to API FPS while recording stays at a fixed 20 FPS.
  let apiFps;
  if (durationInSeconds <= 1) apiFps = 12;
  else if (durationInSeconds <= 2) apiFps = 10;
  else if (durationInSeconds <= 3) apiFps = 8;
  else if (durationInSeconds <= 4) apiFps = 6;
  else if (durationInSeconds <= 5) apiFps = 6;
  else if (durationInSeconds <= 6) apiFps = 6;
  else if (durationInSeconds <= 8) apiFps = 6;
  else apiFps = 6;

  const config = {
    recordingFps: 20,
    apiFps,
    videoCrf: 24,
  };
  return config;
}
/**
 * Adds a silent audio track to a clip file using ffmpeg
 * @param {string} inputPath - Path to the input clip file
 * @param {string} outputPath - Path to save the output clip file with audio
 * @returns {Promise<boolean>} - Returns true if successful, false otherwise
 */
async function addSilentAudioTrack(inputPath, outputPath) {
  try {
    console.log(`Adding silent audio track to video: ${inputPath}`);

    // Determine ffmpeg command based on environment
    // Prefer the binary provided by ffmpeg-static when available
    let ffmpegCommand = ffmpegStatic || "ffmpeg";

    // In production, ffmpeg might be at a specific path
    if (process.env.NODE_ENV === "production") {
      // Check if ffmpeg is available in PATH first
      try {
        execSync("ffmpeg -version", { stdio: "ignore" });
      } catch (error) {
        // If not in PATH, try common positions
        const possiblePaths = [
          "/usr/bin/ffmpeg",
          "/usr/local/bin/ffmpeg",
          "/opt/render/project/.render/ffmpeg/bin/ffmpeg",
        ];

        for (const path of possiblePaths) {
          try {
            execSync(`${path} -version`, { stdio: "ignore" });
            ffmpegCommand = path;
            console.log(`Using ffmpeg at: ${path}`);
            break;
          } catch (e) {
            // Continue to next path
          }
        }
      }
    }

    // FFmpeg command to add a silent audio track
    // Using spawnSync avoids Windows quoting issues.
    const resolvedInputPath = path.resolve(inputPath);
    const resolvedOutputPath = path.resolve(outputPath);

    const ffArgs = [
      "-i",
      resolvedInputPath,
      "-f",
      "lavfi",
      "-i",
      "anullsrc=r=44100:cl=stereo",
      "-c:v",
      "copy",
      "-c:a",
      "aac",
      "-shortest",
      "-y",
      resolvedOutputPath,
    ];

    const { status, error, stderr } = spawnSync(ffmpegCommand, ffArgs, {
      stdio: "inherit",
    });

    if (status !== 0) {
      throw new Error(
        `ffmpeg exited with code ${status}. ${(error && error.message) || ""}`
      );
    }

    console.log(
      `Successfully added silent audio track. Output saved to: ${outputPath}`
    );
    return true;
  } catch (error) {
    console.error("Error adding silent audio track:", error);
    return false;
  }
}
async function trimVideo(
  inputPath,
  outputPath,
  startTimeSeconds,
  durationSeconds
) {
  try {
    const durationText = durationSeconds
      ? `for ${durationSeconds.toFixed(3)}s`
      : "to the end of the clip";
    console.log(
      `Trimming video: ${inputPath} starting from ${startTimeSeconds.toFixed(
        3
      )}s ${durationText}`
    );

    let ffmpegCommand = ffmpegStatic || "ffmpeg";

    // In production, ffmpeg might be at a specific path
    if (process.env.NODE_ENV === "production") {
      // Check if ffmpeg is available in PATH first
      try {
        execSync("ffmpeg -version", { stdio: "ignore" });
      } catch (error) {
        // If not in PATH, try common positions
        const possiblePaths = [
          "/usr/bin/ffmpeg",
          "/usr/local/bin/ffmpeg",
          "/opt/render/project/.render/ffmpeg/bin/ffmpeg",
        ];

        for (const path of possiblePaths) {
          try {
            execSync(`${path} -version`, { stdio: "ignore" });
            ffmpegCommand = path;
            console.log(`Using ffmpeg at: ${path}`);
            break;
          } catch (e) {
            // Continue to next path
          }
        }
      }
    }

    const resolvedInputPath = path.resolve(inputPath);
    const resolvedOutputPath = path.resolve(outputPath);

    // Using -ss after -i for accurate seeking, combined with re-encoding.
    const ffArgs = ["-i", resolvedInputPath, "-ss", String(startTimeSeconds)];

    if (durationSeconds) {
      ffArgs.push("-t", String(durationSeconds));
    }

    ffArgs.push(
      "-c:v",
      "libx264",
      "-preset",
      "ultrafast",
      "-crf",
      "18",
      "-an", // The source clip from puppeteer-screen-recorder has no audio
      "-y",
      resolvedOutputPath
    );

    console.log(`Executing ffmpeg with args: ${ffArgs.join(" ")}`);

    const { status, error, stderr, stdout } = spawnSync(ffmpegCommand, ffArgs, {
      encoding: "utf-8", // To see stdout/stderr as strings
    });

    if (status !== 0) {
      console.error("ffmpeg stdout:", stdout);
      console.error("ffmpeg stderr:", stderr);
      throw new Error(
        `ffmpeg exited with code ${status}. ${
          (error && error.message) || stderr
        }`
      );
    }

    console.log(`Successfully trimmed video. Output saved to: ${outputPath}`);
    return true;
  } catch (error) {
    console.error("Error trimming video:", error);
    return false;
  }
}

/**
 * Splits a clip into click and no-click clips_folder based on beacon timings.
 * This function implements the logic described by the user to partition a video.
 *
 * @param {Array<{color: string, startTime: number, endTime: number}>} beacons - Detected beacons from the video.
 * @param {number} videoDurationSeconds - The total duration of the clip in seconds.
 * @returns {{allClips: Array<{start: number, end: number, type: 'click' | 'no-click'}>}}
 */
function splitVideoIntoSegments(beacons, videoDurationSeconds) {
  console.log(
    "Splitting clip into segments based on beacons (v2 with single boundary marker)..."
  );

  const sortByStart = (arr) => arr.sort((a, b) => a.startTime - b.startTime);

  const greenBeacons = sortByStart(
    (beacons || []).filter((b) => b.color === "green")
  );
  const redBeacons = sortByStart(
    (beacons || []).filter((b) => b.color === "red")
  );
  const yellowBeacons = sortByStart(
    (beacons || []).filter((b) => b.color === "yellow")
  );
  const blueBeacons = sortByStart(
    (beacons || []).filter((b) => b.color === "blue")
  );

  if (DEBUG_NOCLICK) {
    console.log(
      `[SEGMENT-DEBUG] Beacons: green=${greenBeacons.length}, red=${redBeacons.length}, yellow=${yellowBeacons.length}, blue=${blueBeacons.length}`
    );
  }

  // Pair green->red for click intervals
  const clickEvents = [];
  if (greenBeacons.length > 0) {
    const usedRed = new Set();
    for (const g of greenBeacons) {
      const r = redBeacons.find(
        (rb) => rb.startTime > g.endTime && !usedRed.has(rb)
      );
      if (r) {
        clickEvents.push({ green: g, red: r });
        usedRed.add(r);
      }
    }
  }

  console.log(
    `Found ${clickEvents.length} click events (green/red beacon pairs).`
  );

  // Build concrete click clip intervals strictly bounded by real yellow beacons
  const clickIntervals = [];
  const syntheticYellowBeacons = [];
  const YELLOW_EPS = 0.001; // 1ms tiny duration for boundary markers

  // Precompute yellow boundary times
  const yellowStartTimes = (yellowBeacons || [])
    .map((b) => b?.startTime)
    .filter((t) => Number.isFinite(t))
    .sort((a, b) => a - b);
  const yellowEndTimes = (yellowBeacons || [])
    .map((b) => b?.endTime)
    .filter((t) => Number.isFinite(t))
    .sort((a, b) => a - b);

  const prevYellowEndBefore = (t) => {
    // Last yellow end strictly before or equal to t
    let v = 0;
    let found = false;
    for (let i = 0; i < yellowEndTimes.length; i++) {
      const yt = yellowEndTimes[i];
      if (yt <= t) {
        v = yt;
        found = true;
      } else break;
    }
    return found ? v : 0;
  };
  const nextYellowStartAfter = (t) => {
    // First yellow start greater than or equal to t
    for (let i = 0; i < yellowStartTimes.length; i++) {
      const ys = yellowStartTimes[i];
      if (ys >= t) return ys;
    }
    return videoDurationSeconds;
  };

  for (let i = 0; i < clickEvents.length; i++) {
    const currentClick = clickEvents[i];
    const nextClick = clickEvents[i + 1];

    // 1) Start at previous yellow boundary (end of the last yellow)
    let startYellow = prevYellowEndBefore(currentClick.green.startTime);
   
    // 2) End at next yellow boundary (start of the next yellow)
    let endYellow = nextYellowStartAfter(currentClick.red.endTime);

    // Guard rails: if something odd, fall back to prior heuristic
    if (!Number.isFinite(startYellow) || startYellow < 0) startYellow = 0;
    if (!Number.isFinite(endYellow) || endYellow > videoDurationSeconds)
      endYellow = videoDurationSeconds;
    // Ensure non-negative, and if degenerate, widen minimally around green/red
    if (endYellow - startYellow < 0.05) {
      // fallback: small window around anchors
      startYellow = Math.max(0, currentClick.green.startTime - 0.35);
      endYellow = Math.min(
        videoDurationSeconds,
        currentClick.red.endTime + 0.35
      );
    }

    // Optional extra safety: if the end spills into the next click's start boundary, clamp
    if (nextClick) {
      const nextClickStartBoundary = prevYellowEndBefore(
        nextClick.green.startTime
      );
      if (endYellow > nextClickStartBoundary) {
        endYellow = nextClickStartBoundary;
      }
    }

    if (DEBUG_CLICK || DEBUG_NOCLICK) {
      console.log(
        `[CLICK-SEG] start=${startYellow.toFixed(3)}s end=${endYellow.toFixed(
          3
        )}s boundedBy=yellow`
      );
    }

    // Record intervals using the yellow boundaries
    clickIntervals.push({
      start: startYellow,
      end: endYellow,
      type: "click",
      clickTimestamp: currentClick.green.startTime,
      greenBeacon: currentClick.green,
      redBeacon: currentClick.red,
    });

    // Create tiny synthetic yellow markers at the boundaries for internal boundary handling
    // Start boundary marker: [start, start+eps]
    syntheticYellowBeacons.push({
      color: "yellow",
      startTime: startYellow,
      endTime: Math.min(videoDurationSeconds, startYellow + YELLOW_EPS),
      markerType: "CLICK_START_BOUNDARY",
      clickIndex: i,
    });
    // End boundary marker: [end-eps, end]
    const endStart = Math.max(0, endYellow - YELLOW_EPS);
    syntheticYellowBeacons.push({
      color: "yellow",
      startTime: endStart,
      endTime: endYellow,
      markerType: "CLICK_END_BOUNDARY",
      clickIndex: i,
    });
  }

  // Use all boundaries (clicks, yellow beacons) to define final segments
  const allBoundaries = new Set([0, videoDurationSeconds]);
  clickIntervals.forEach((ci) => {
    allBoundaries.add(ci.start);
    allBoundaries.add(ci.end);
  });
  // Add pause beacon boundaries (start and end) so we can exclude them cleanly
  const augmentedYellowBeacons = sortByStart([
    ...yellowBeacons,
    ...syntheticYellowBeacons,
  ]);
  augmentedYellowBeacons.forEach((yb) => {
    if (Number.isFinite(yb.startTime)) allBoundaries.add(yb.startTime);
  });
  blueBeacons.forEach((bb) => {
    if (Number.isFinite(bb.startTime)) allBoundaries.add(bb.startTime);
  });

  // Build pause ranges for filtering
  const pauseRanges = [
    ...augmentedYellowBeacons.map((b) => ({
      start: b.startTime,
      end: b.endTime,
    })),
    ...blueBeacons.map((b) => ({ start: b.startTime, end: b.endTime })),
  ].filter(
    (r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.end > r.start
  );
  const isInsidePause = (t) =>
    pauseRanges.some((r) => t >= r.start && t < r.end);

  const sortedBoundaries = Array.from(allBoundaries).sort((a, b) => a - b);
  if (DEBUG_NOCLICK) {
    // Quick stats on boundary spacing to spot 1-frame slivers
    let minGap = Infinity;
    let smallGapCount = 0;
    for (let i = 0; i < sortedBoundaries.length - 1; i++) {
      const gap = sortedBoundaries[i + 1] - sortedBoundaries[i];
      if (gap < minGap) minGap = gap;
      if (gap < 0.11) smallGapCount++; // ~<1/9s; indicative of frame rounding
    }
    console.log(
      `[SEGMENT-DEBUG] Boundaries: count=${sortedBoundaries.length}, minGap=${
        isFinite(minGap) ? minGap.toFixed(3) : "n/a"
      }s, smallGaps<0.11s=${smallGapCount}`
    );
  }
  const finalClips = [];
  const addedClickIntervals = new Set();

  for (let i = 0; i < sortedBoundaries.length - 1; i++) {
    const start = sortedBoundaries[i];
    const end = sortedBoundaries[i + 1];
    const duration = end - start;
    if (duration < 0.1) continue;

    const midPoint = start + duration / 2;
    const matchingClick = clickIntervals.find(
      (ci) => midPoint >= ci.start && midPoint < ci.end
    );

    if (matchingClick) {
      if (!addedClickIntervals.has(matchingClick)) {
        finalClips.push(matchingClick);
        addedClickIntervals.add(matchingClick);
      }
    } else {
      // Skip creating standalone noclick segments that fall fully within a system-injected pause
      /*  if (isInsidePause(midPoint)) {
        console.log(
          `[BEACON-CUT] Skipping pause-only segment from ${start.toFixed(
            3
          )}s to ${end.toFixed(3)}s (yellow/blue overlay).`
        );
        continue;
      } */
      if (DEBUG_NOCLICK) {
        console.log(
          `[NOC-SEG][CANDIDATE] start=${start.toFixed(3)}s end=${end.toFixed(
            3
          )}s dur=${(end - start).toFixed(3)}s mid=${midPoint.toFixed(
            3
          )}s (i=${i})`
        );
      }
      // ==================================================================
      // <<< 🎯 THIS IS THE NEW LOGIC TO ADD 🎯 >>>
      // ==================================================================
      const nextBoundaryIndex = i + 2;
      if (nextBoundaryIndex < sortedBoundaries.length) {
        const nextSegmentStart = sortedBoundaries[i + 1];
        const nextSegmentEnd = sortedBoundaries[i + 2];
        const nextMidPoint =
          nextSegmentStart + (nextSegmentEnd - nextSegmentStart) / 2;
        const nextSegmentIsClick = clickIntervals.some(
          (ci) => nextMidPoint >= ci.start && nextMidPoint < ci.end
        );

        // If this noclick segment is short AND the very next segment is a click, discard it.
        if (nextSegmentIsClick && duration < 3.0) {
          // 3.0 second threshold
          console.log(
            `[BEACON-CUT] Discarding short pre-click noclick segment from ${start.toFixed(
              2
            )}s to ${end.toFixed(2)}s.`
          );
          continue; // Skip adding this clip
        }
      }
      // ==================================================================
      // <<< 🎯 END OF NEW LOGIC 🎯 >>>
      // ==================================================================
      if (DEBUG_NOCLICK) {
        console.log(
          `[NOC-SEG][CREATE] start=${start.toFixed(3)}s end=${end.toFixed(
            3
          )}s dur=${(end - start).toFixed(3)}s`
        );
      }
      finalClips.push({ start, end, type: "noclick" });
    }
  }

  // Sort final clips_folder by start time to ensure order
  finalClips.sort((a, b) => a.start - b.start);

  if (DEBUG_NOCLICK) {
    const clickCount = finalClips.filter((c) => c.type === "click").length;
    const noclickCount = finalClips.filter((c) => c.type === "noclick").length;
    console.log(
      `[SEGMENT-DEBUG] Final clips_folder: total=${finalClips.length} click=${clickCount} noclick=${noclickCount}`
    );
  }
  console.log(
    "Final segmented clips_folder:",
    finalClips
  );
  return { allClips: finalClips };
}
/**
 * Cuts a master clip into segments based on detected beacon flashes.
 * - Detects beacons via raw frame analysis
 * - Builds click/noclick segments from green/red pairs
 * - Trims those segments from the master video
 * - Optionally removes beacon frames from each segment and adds silent audio
 *
 * @param {string} videoPath Absolute or relative path to master video
 * @param {string} outputDir Directory to write segments into
 * @param {{
 *   includeClick?: boolean,
 *   includeNoClick?: boolean,
 *   removeBeaconFrames?: boolean,
 *   addAudio?: boolean
 * }} [options]
 * @returns {Promise<{segments: Array<{type: 'click'|'noclick', start: number, end: number, path: string}>}>}
 */
export async function cutVideoBasedOnBeacons(
  videoPath,
  outputDir,
  options = {}
) {
  const {
    includeClick = true,
    includeNoClick = true,
    removeBeaconFrames = true,
    addAudio = true,
    sessionId: optionsSessionId = null,
    preserveGapDuration = false,
  } = options || {};

  try {
    const resolvedVideoPath = path.resolve(videoPath);
    await fs.mkdir(outputDir, { recursive: true });

    // Try to load cached beacon analysis if present for this output directory
    const beaconDir = path.join(outputDir, "beacon_segments");
    await fs.mkdir(beaconDir, { recursive: true });
    const beaconJsonPath = path.join(beaconDir, "beacon_segments.json");

    let beacons = null;
    try {
      if (fs2.existsSync(beaconJsonPath)) {
        const cached = JSON.parse(await fs.readFile(beaconJsonPath, "utf-8"));
        if (
          cached &&
          cached.videoPath &&
          path.resolve(String(cached.videoPath)) === resolvedVideoPath &&
          Array.isArray(cached.beacons)
        ) {
          beacons = cached.beacons;
          console.log(
            `[BEACON-CUT] Using cached beacon analysis: ${beaconJsonPath}`
          );
        }
      }
    
    } catch (_) {}

    if (!beacons) {
      console.log(
        `[BEACON-CUT] Analyzing master clip for beacons: ${resolvedVideoPath}`
      );
      beacons = await findBeaconFrames_TrueStream(resolvedVideoPath);
    }
    const durationSeconds = getVideoDurationSecondsSync(resolvedVideoPath);
    if (!Number.isFinite(durationSeconds)) {
      throw new Error("Could not determine clip duration");
    }

    // Build a pause map (yellow/blue overlays) to translate visual time back to logical time
    const pauseBeacons = (beacons || [])
      .filter(
        (b) =>
          b &&
          (b.color === "yellow" || b.color === "blue") &&
          Number.isFinite(b.startTime) &&
          Number.isFinite(b.endTime) &&
          b.endTime > b.startTime
      )
      .sort((a, b) => a.endTime - b.endTime);

    const getClickFreezeTimeSecondsUpTo = (t) => {
      let sum = 0;
      const greenBeacons = (beacons || []).filter((b) => b.color === "green");
      const redBeacons = (beacons || []).filter((b) => b.color === "red");

      for (const g of greenBeacons) {
        if (g.endTime <= t) {
          const r = redBeacons.find((rb) => rb.startTime > g.endTime);
          if (r) {
            sum += r.endTime - g.startTime;
          }
        }
      }
      return sum;
    };

    const cumulativePauseSecondsUpTo = (t) => {
      let sum = 0;
      for (const pb of pauseBeacons) {
        if (pb.endTime <= t) sum += pb.endTime - pb.startTime;
        else if (pb.startTime < t && pb.endTime > t) {
          sum += t - pb.startTime;
          break; // remaining pauses start after t
        } else if (pb.startTime >= t) {
          break;
        }
      }
      return sum;
    };

    // Persist raw beacon data for inspection (augmented with IDs from subtitle track when available)
    try {
      const cues = await extractBeaconSubtitleCues(resolvedVideoPath);
      const matchIdFor = (b) => {
        if (!cues || cues.length === 0) return null;
        const bs = Number(b.startTime),
          be = Number(b.endTime);
        if (!Number.isFinite(bs) || !Number.isFinite(be) || be <= bs)
          return null;
        let best = null;
        let bestOverlap = 0;
        for (const c of cues) {
          const s = Math.max(bs, c.start);
          const e = Math.min(be, c.end);
          const overlap = Math.max(0, e - s);
          if (overlap > bestOverlap) {
            best = c;
            bestOverlap = overlap;
          }
        }
        // Require at least 0.1s overlap to consider it a match
        if (best && bestOverlap >= 0.1) return best.id || null;
        return null;
      };
      const beaconsAugmented = (beacons || []).map((b) => {
        if (b && b.color === "yellow") {
          return { ...b, id: matchIdFor(b) };
        }
        return b;
      });

      const beaconDir = path.join(outputDir, "beacon_segments");
      await fs.mkdir(beaconDir, { recursive: true });
      const beaconJsonPath = path.join(beaconDir, "beacon_segments.json");
      await fs.writeFile(
        beaconJsonPath,
        JSON.stringify(
          { videoPath: resolvedVideoPath, beacons: beaconsAugmented },
          null,
          2
        )
      );
      console.log(`[BEACON-CUT] Saved beacon analysis: ${beaconJsonPath}`);
    } catch (_) {}

    // Build segments from beacons
    const { allClips } = splitVideoIntoSegments(beacons || [], durationSeconds);
    const segmentsOut = [];
    const MAX_NOCLICK_SECONDS =
      (typeof NONCLICK_MAX_CLIP_DURATION_MS === "number"
        ? NONCLICK_MAX_CLIP_DURATION_MS
        : 10000) / 1000;
    // Initialize counters to continue after any existing files in the outputDir
    let noclickCounter = 1;
    let clickCounter = 0;
    try {
      const existing = await fs.readdir(outputDir);
      const reNo = /^noclick_(\d+)_from_/i;
      const reEv = /^click_(\d+)_from_/i;
      let maxNoNum = 0;
      let maxEvNum = -1;
      for (const f of existing) {
        const mNo = reNo.exec(f);
        if (mNo) {
          const n = parseInt(mNo[1], 10);
          if (Number.isFinite(n)) maxNoNum = Math.max(maxNoNum, n);
        }
        const mEv = reEv.exec(f);
        if (mEv) {
          const n = parseInt(mEv[1], 10);
          if (Number.isFinite(n)) maxEvNum = Math.max(maxEvNum, n);
        }
      }
      if (maxNoNum > 0) noclickCounter = maxNoNum + 1;
      if (maxEvNum >= 0) clickCounter = maxEvNum + 1;
    } catch (_) {}

    let totalTimeRemovedSeconds = 0.0;
    let lastLogicalEndTime = 0.0; // Track the end of the last logical segment

    for (let i = 0; i < allClips.length; i++) {
      const seg = allClips[i];
      
      const start = Math.max(0, seg.start);
      const fileSafe = (v) => String(v.toFixed(3)).replace(/[^0-9.]/g, "");
      const end = Math.min(durationSeconds, seg.end);
      const dur = Math.max(0, end - start);
      
      // Map physical start/end to logical (pre-pause) time to get the correct logical DURATION.
      const mapToLogical = (t) =>
        Math.max(
          0,
          t - cumulativePauseSecondsUpTo(t) - getClickFreezeTimeSecondsUpTo(t)
        );

      const mappedStart = mapToLogical(start);
      const mappedEnd = mapToLogical(end);
      const logicalDur = Math.max(0, mappedEnd - mappedStart);

    /*   if (
        (seg.type === "click" && !includeClick) ||
        (seg.type === "noclick" && !includeNoClick)
      ) {
        if (preserveGapDuration) {
           lastLogicalEndTime += logicalDur;
        }
        continue;
      }

      if (dur < 0.1) continue; */

      // The logical START of this segment is the logical END of the previous one.
      const logicalStart = lastLogicalEndTime;

      // Click and noclick segments (no capping/splitting of long noclicks)
      // Allow choosing which timeline to reflect in filenames: logical (default) vs visual
      const baseName =
        seg.type === "noclick"
          ? `noclick_${noclickCounter}_from_${fileSafe(
              logicalStart
            )}s_to_${fileSafe(logicalStart + logicalDur)}s_raw.mp4`
          : `click_${clickCounter}_from_${fileSafe(
              logicalStart
            )}s_to_${fileSafe(logicalStart + logicalDur)}s.mp4`;
      const outPath = path.join(outputDir, baseName);
      console.log(
        `[BEACON-CUT] Trimming ${seg.type} segment ${i}: start=${start.toFixed(
          3
        )}s (logical=${logicalStart.toFixed(3)}s), duration=${dur.toFixed(
          3
        )}s -> ${baseName}`
      );
      const trimOk = await trimVideo(resolvedVideoPath, outPath, start, dur);
      if (!trimOk) {
        console.warn(`[BEACON-CUT] Failed to trim segment ${i}`);
        continue;
      }
      let processedPath = null;
      let finalDuration = dur;
      if (
        removeBeaconFrames &&
        (seg.type === "click" || seg.type === "noclick")
      ) {
        try {
          // Instead of re-detecting beacons on the small clip, filter the master beacon list.
          // This is more reliable, especially for beacons at the very edge of a clip.
          const masterBeaconsForSegment = (beacons || []).filter(
            (b) => b.startTime <= seg.end && b.endTime >= seg.start
          );

          const segBeacons = masterBeaconsForSegment.map((b) => ({
            ...b,
            startTime: Math.max(0, b.startTime - seg.start),
            endTime: b.endTime - seg.start,
          }));

          const processedOut =
            seg.type === "noclick"
              ? outPath.replace(/_raw\.mp4$/i, ".mp4")
              : outPath.replace(/\.mp4$/i, "_processed.mp4");
          let ok = false;
          if (Array.isArray(segBeacons) && segBeacons.length > 0) {
            ok = await postProcessClip(outPath, processedOut, segBeacons);
          } else if (seg.type === "noclick") {
            // For noclick clips_folder with no beacons, just copy the original to the processed path.
            console.log(
              `[BEACON-CUT] No beacons in noclick segment ${i}, creating copy as processed.`
            );
            await fs.copyFile(outPath, processedOut);
            ok = true;
          }

          if (ok) {
            console.log(
              `[BEACON-CUT] Created processed segment ${i}: ${path.basename(
                processedOut
              )}`
            );
            processedPath = processedOut;
            // Trim trailing frames identical to the last frame
            console.log(
              `[BEACON-CUT] Tail-trim check on segment ${i}: ${path.basename(
                processedPath
              )}`
            );
            try {
              const { removedFrames } = await trimTrailingDuplicateFrames(
                processedPath
              );
              if (removedFrames > 0) {
                console.log(
                  `[BEACON-CUT] Tail-trim removed ${removedFrames} duplicate frame(s) from segment ${i}.`
                );
              } else {
                console.log(
                  `[BEACON-CUT] Tail-trim found no trailing duplicates for segment ${i}.`
                );
              }
            } catch (e) {
              console.warn(
                `[BEACON-CUT] Tail-trim warning (segment ${i}): ${
                  e?.message || e
                }`
              );
            }
            const processedDuration =
              getVideoDurationSecondsSync(processedPath);
            if (processedDuration) {
              finalDuration = processedDuration;
              const removedSeconds = dur - processedDuration;
              totalTimeRemovedSeconds += removedSeconds;
            }
            if (addAudio) {
              try {
                const tempProcessed = processedOut.replace(
                  /\.mp4$/i,
                  ".temp.mp4"
                );
                const ok2 = await addSilentAudioTrack(
                  processedOut,
                  tempProcessed
                );
                if (ok2) await fs.rename(tempProcessed, processedOut);
                else {
                  try {
                    await fs.unlink(tempProcessed);
                  } catch (_) {}
                }
              } catch (e) {
                console.warn(
                  `[BEACON-CUT] Audio add warning (processed segment ${i}): ${
                    e?.message || e
                  }`
                );
              }
            }
          }
        } catch (e) {
          console.warn(
            `[BEACON-CUT] Post-process warning (segment ${i}): ${
              e?.message || e
            }`
          );
        }
      }
      if (addAudio) {
        try {
          const tempOut = outPath.replace(/\.mp4$/i, ".temp.mp4");
          const ok = await addSilentAudioTrack(outPath, tempOut);
          if (ok) {
            await fs.rename(tempOut, outPath);
          } else {
            try {
              await fs.unlink(tempOut);
            } catch (_) {}
          }
        } catch (e) {
          console.warn(
            `[BEACON-CUT] Audio add warning (segment ${i}): ${e?.message || e}`
          );
        }
      }
      const adjustedEnd = logicalStart + finalDuration;

      // ---- NEW: Rename files to reflect final duration and update paths ----
      const finalBaseName =
        seg.type === "noclick"
          ? `noclick_${noclickCounter}_from_${fileSafe(
              logicalStart
            )}s_to_${fileSafe(adjustedEnd)}s.mp4`
          : `click_${clickCounter}_from_${fileSafe(
              logicalStart
            )}s_to_${fileSafe(adjustedEnd)}s.mp4`;

      let finalRawPath = path.join(outputDir, finalBaseName);
      if (seg.type === "noclick") {
        // For noclick, the "raw" version gets the _raw suffix
        finalRawPath = finalRawPath.replace(/\.mp4$/, "_raw.mp4");
      }

      let finalProcessedPath = null;
      if (processedPath) {
        finalProcessedPath = path.join(outputDir, finalBaseName);
        if (seg.type === "click") {
          // For click, the "processed" version gets the _processed suffix
          finalProcessedPath = finalProcessedPath.replace(
            /\.mp4$/,
            "_processed.mp4"
          );
        }
      }

      try {
        // Rename the raw file (outPath) to its final name
        if (fs2.existsSync(outPath) && outPath !== finalRawPath) {
          await fs.rename(outPath, finalRawPath);
        }
        // Rename the processed file (processedPath) to its final name
        if (
          processedPath &&
          fs2.existsSync(processedPath) &&
          processedPath !== finalProcessedPath
        ) {
          await fs.rename(processedPath, finalProcessedPath);
        }
      } catch (renameError) {
        console.warn(
          `[BEACON-CUT] Failed to rename clip files for segment ${i}, using original names. Error: ${renameError.message}`
        );
        // If rename fails, revert to original paths to avoid breaking downstream processing
        finalRawPath = outPath;
        finalProcessedPath = processedPath;
      }
      // ---- END NEW ----

      segmentsOut.push({
        type: seg.type,
        start: logicalStart,
        end: adjustedEnd,
        path: finalRawPath,
        processedPath: finalProcessedPath || null,
      });

      lastLogicalEndTime = adjustedEnd; // Update the tracker for the next iteration

      if (seg.type === "noclick") {
        noclickCounter++;
      } else {
        clickCounter++;
      }
    }

    console.log(
      `[BEACON-CUT] Finished. Segments created: ${segmentsOut.length}`
    );
    return { segments: segmentsOut };
  } catch (e) {
    console.error("[BEACON-CUT] Error:", e);
    throw e;
  }
}

async function embedBeaconSubtitlesIntoVideo(videoPath, sessionId) {
  try {
    const resolved = path.resolve(videoPath);
    const dir = path.dirname(resolved);
    const { name, ext } = path.parse(resolved);
    const tempMuxInput = path.join(dir, `${name}.pre_sub${ext || ".mp4"}`);
    const tempSrtPath = path.join(dir, `${name}_beacons.srt`);

    // Detect beacons from the final clip (post-trim/post-freeze-removal)
    const beacons = await findBeaconFrames_TrueStream(resolved);
    if (!Array.isArray(beacons) || beacons.length === 0) {
      return { added: false, reason: "no_beacons" };
    }

    const idMap = await writeBeaconSrt(beacons, tempSrtPath, sessionId);
    if (!idMap || idMap.length === 0) {
      try {
        await fs.unlink(tempSrtPath);
      } catch (_) {}
      return { added: false, reason: "no_yellow" };
    }

    // Rename original to temp, mux SRT into new output at original path
    await fs.rename(resolved, tempMuxInput);
    try {
      await muxSrtIntoMp4(tempMuxInput, tempSrtPath, resolved);
    } catch (e) {
      // Restore original on failure
      try {
        await fs.rename(tempMuxInput, resolved);
      } catch (_) {}
      try {
        await fs.unlink(tempSrtPath);
      } catch (_) {}
      throw e;
    }
    // Clean up temp files
    try {
      await fs.unlink(tempMuxInput);
    } catch (_) {}
    try {
      await fs.unlink(tempSrtPath);
    } catch (_) {}
    return { added: true, count: idMap.length };
  } catch (e) {
    console.warn(
      `[SUB-MUX] Failed to embed beacon subtitles: ${e?.message || e}`
    );
    return { added: false, error: e?.message || String(e) };
  }
}
export async function recordRrwebEvents(
  events,
  overlayInstructions,
  customOutputVideoPath,
  sessionId,
  apiKeys
) {
  let browser;
  const outputVideoPath = customOutputVideoPath || DEFAULT_OUTPUT_VIDEO_PATH;

  // Use sessionId for unique temp files. Fallback to random string if no ID given.
  const uniqueId = sessionId || crypto.randomBytes(6).toString("hex");

  // Use OS temporary directory for robust file creation
  const tempDir = os.tmpdir();
  const tempFilePath = path.join(tempDir, `temp_player_${uniqueId}.html`);
  const tempEventsPath = path.join(tempDir, `temp_events_${uniqueId}.json`);

  // Variables to track clip file size and timing
  let videoSizeBytes = 0;
  let videoSizeMB = 0;
  let finalFps = null;
  let timingOffsetMs = 0; // Initialize timing offset variable

  // Mark all click events to trigger the beacon/freeze effect in the main video.
  const eventsWithClicksMarked = events.map((event) => {
    if (
      event.type === 3 && // IncrementalSnapshot
      event.data?.source === 2 && // MouseInteraction
      event.data?.type === 2 // Click or TouchEnd
    ) {
      return {
        ...event,
        data: {
          ...event.data,
          isTargetClick: true,
          freezeMarker: "INSTANT_FREEZE_HERE",
        },
      };
    }
    return event;
  });

  // Calculate the extra time needed for all the freeze/beacon sequences.
  const clickCountForFreeze = eventsWithClicksMarked.filter(
    (e) => e.data?.isTargetClick
  ).length;
  const freezeTimePerClickMs = 3900; // 450ms green + 3000ms freeze + 450ms red
  const totalFreezeTimeMs = clickCountForFreeze * freezeTimePerClickMs;
  console.log(
    `[FREEZE-LOGIC] Detected ${clickCountForFreeze} clicks. Adding ${
      totalFreezeTimeMs / 1000
    }s of freeze time to total duration.`
  );

  const clickEventsForFreeze = eventsWithClicksMarked.filter(
    (e) => e.data?.isTargetClick
  );

  // ===== TESTING SHORTCUT =====
  // If you only want to test the downstream Gemini prompt and already have a
  // pre-generated clip (with audio) stored at `outputVideoPath`, set the
  // environment variable `SKIP_RECORDING` to "true" before running the script.
  // This will bypass the main clip recording but still allow clip generation.
  if (process.env.SKIP_RECORDING === "true") {
    /*   if (true) { */
    // Determine if this is a main clip or a clip based on overlayInstructions
    // Main videos have overlay instructions, clips_folder have empty array
    const isMainVideo = overlayInstructions && overlayInstructions.length > 0;
    const isClip = !isMainVideo && outputVideoPath.includes("clip");

    if (isMainVideo) {
      console.log(
        `[recordRrwebEvents] SKIP_RECORDING flag detected – using existing MAIN clip at "${outputVideoPath}" and skipping main recording.`
      );
      try {
        const stats = await fs.stat(outputVideoPath);
        videoSizeBytes = stats.size;
        videoSizeMB = (videoSizeBytes / (1024 * 1024)).toFixed(2);
        console.log(
          `[recordRrwebEvents] Existing main clip size: ${videoSizeMB} MB (${videoSizeBytes} bytes).`
        );
      } catch (err) {
        console.warn(
          `[recordRrwebEvents] Could not stat existing main clip file: ${err.message}`
        );
      }
      return { success: true, videoSizeBytes, videoSizeMB, fps: null };
    } else if (isClip) {
      console.log(
        `[recordRrwebEvents] SKIP_RECORDING flag detected but this is a CLIP – proceeding with clip recording: "${outputVideoPath}"`
      );
      // Continue with normal clip recording below
    } else {
      console.log(
        `[recordRrwebEvents] SKIP_RECORDING flag detected – using existing clip at "${outputVideoPath}" and skipping recording entirely.`
      );
      try {
        const stats = await fs.stat(outputVideoPath);
        videoSizeBytes = stats.size;
        videoSizeMB = (videoSizeBytes / (1024 * 1024)).toFixed(2);
        console.log(
          `[recordRrwebEvents] Existing clip size: ${videoSizeMB} MB (${videoSizeBytes} bytes).`
        );
      } catch (err) {
        console.warn(
          `[recordRrwebEvents] Could not stat existing clip file: ${err.message}`
        );
      }
      return { success: true, videoSizeBytes, videoSizeMB, fps: null };
    }
  }

  try {
    if (!events || events.length === 0) {
      console.log("No events for display. Video will not be created.");
      return false;
    }
    // overlayInstructions can be empty, that's fine.
    // Save eventsWithClicksMarked to a local file for debugging
    /*     try {
      const localEventsPath = `eventsWithClicksMarked_${uniqueId}.json`;
      await fs.writeFile(
        localEventsPath,
        JSON.stringify(eventsWithClicksMarked, null, 2),
        "utf-8"
      );
      console.log(
        `Saved eventsWithClicksMarked to local directory: ${localEventsPath}`
      );
    } catch (e) {
      console.warn("Could not save local eventsWithClicksMarked file", e);
    } */

    // Write events to a temporary file to avoid passing large data to puppeteer
    await fs.writeFile(
      tempEventsPath,
      JSON.stringify(eventsWithClicksMarked),
      "utf-8"
    );

    console.log(`Temporary events JSON written to ${tempEventsPath}`);

    const firstProcessedEventTimestamp = eventsWithClicksMarked[0].timestamp;
    const lastProcessedEventTimestamp =
      eventsWithClicksMarked[eventsWithClicksMarked.length - 1].timestamp;
    const recordingDurationMs =
      lastProcessedEventTimestamp - firstProcessedEventTimestamp;

    // DEBUG: Log expected event timeline for main session video
    try {
      console.log(
        `[MAIN-DEBUG] Events timeline firstTs=${firstProcessedEventTimestamp} (${new Date(
          firstProcessedEventTimestamp
        ).toISOString()}), lastTs=${lastProcessedEventTimestamp} (${new Date(
          lastProcessedEventTimestamp
        ).toISOString()}), durationMs=${recordingDurationMs}`
      );
      const ovCount = Array.isArray(overlayInstructions)
        ? overlayInstructions.length
        : 0;
      if (ovCount > 0) {
        const firstOv = overlayInstructions[0];
        const lastOv = overlayInstructions[ovCount - 1];
        console.log(
          `[MAIN-DEBUG] Overlay instructions: count=${ovCount}, first(showAt=${firstOv?.showAt}, hideAt=${firstOv?.hideAt}), last(showAt=${lastOv?.showAt}, hideAt=${lastOv?.hideAt})`
        );
      } else {
        console.log(`[MAIN-DEBUG] Overlay instructions: none`);
      }
    } catch (_) {}

    // Extract viewport change events (type 4 = meta events with viewport data)
    const viewportEvents = events.filter(
      (e) => e.type === 4 && e.data && e.data.width && e.data.height
    );
    try {
      const first3 = viewportEvents.slice(0, 3).map((e) => ({
        ts: e.timestamp,
        iso: new Date(e.timestamp).toISOString(),
        w: e.data.width,
        h: e.data.height,
      }));
      const last3 = viewportEvents.slice(-3).map((e) => ({
        ts: e.timestamp,
        iso: new Date(e.timestamp).toISOString(),
        w: e.data.width,
        h: e.data.height,
      }));
      console.log(
        `[VIEWPORT_DEBUG][HOST] meta viewport events count=${viewportEvents.length}`
      );
      if (first3.length)
        console.log(`[VIEWPORT_DEBUG][HOST] first3 meta:`, first3);
      if (last3.length)
        console.log(`[VIEWPORT_DEBUG][HOST] last3  meta:`, last3);
    } catch (_) {}

    // FIXED OUTPUT VIDEO SIZE - Always 1920x1080
    const viewportWidth = 1920;
    const viewportHeight = 1080;

    // Get initial content dimensions from the first meta event
    let originalWidth = 1920; // Default fallback
    let originalHeight = 1080;

    if (viewportEvents.length > 0) {
      const firstViewportEvent = viewportEvents[0];
      originalWidth = firstViewportEvent.data.width;
      originalHeight = firstViewportEvent.data.height;
    }

    // Calculate initial scale to fit content within the fixed viewport
    // Never scale up (max scale = 1.0), only scale down if content is larger than viewport
    const scale = Math.min(
      viewportWidth / originalWidth,
      viewportHeight / originalHeight
    );
    try {
      console.log(
        `[VIEWPORT_DEBUG][HOST] fixedViewport=${viewportWidth}x${viewportHeight}, originalContent=${originalWidth}x${originalHeight}, initialScale=${scale}`
      );
    } catch (_) {}

    console.log("Launching browser for replay and overlay display...");

    let args = {
      // true = no browser shows
      // false = browser shows
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--allow-file-access-from-files",
        `--window-size=${viewportWidth},${viewportHeight}`,
        "--force-device-scale-factor=1",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--no-zygote",
      ],
      protocolTimeout: 0,
    };

    browser = await puppeteer.launch({ ...args });
    const page = await browser.newPage();
    // Attach early console and error handlers before any navigation or script runs
    page.on("console", (msg) => {
      try {
        const args = msg.args();
        const serialized = args && args.length ? ` | args=${args.length}` : "";
        console.log(`[PAGE CONSOLE ${msg.type()}]:`, msg.text() + serialized);
      } catch (_) {
        console.log(`[PAGE CONSOLE ${msg.type()}]:`, msg.text());
      }
    });
    await page.evaluateOnNewDocument(() => {
      try {
        window.addEventListener("error", function (e) {
          try {
            console.error(
              "[EARLY PAGE ERROR]",
              e.message,
              e.filename,
              e.lineno,
              e.colno
            );
            try {
              const src =
                (document &&
                  document.documentElement &&
                  document.documentElement.outerHTML) ||
                "";
              if (src && e && typeof e.lineno === "number" && e.lineno > 0) {
                const lines = src.split("\n");
                const idx = Math.min(lines.length - 1, e.lineno - 1);
                const line = lines[idx] || "";
                console.error(
                  "[EARLY PAGE ERROR][SOURCE] line=",
                  e.lineno,
                  "col=",
                  e.colno,
                  "=>",
                  line
                );
              }
            } catch (_) {}
          } catch (_) {}
        });
        window.addEventListener("unhandledrejection", function (e) {
          try {
            const reason =
              e && e.reason
                ? e.reason.message || String(e.reason)
                : "unknown reason";
            console.error("[EARLY UNHANDLED REJECTION]", reason);
          } catch (_) {}
        });
      } catch (_) {}
    });
    // Add event listeners for diagnostics
    page.on("close", () =>
      console.log(
        "[DIAGNOSTIC] Puppeteer page event: close. The page has been closed."
      )
    );
    page.on("error", (err) =>
      console.error(
        "[DIAGNOSTIC] Puppeteer page event: error (an error was emitted by the page)",
        err
      )
    );
    page.on("pageerror", (pageErr) => {
      // Captures uncaught exceptions from page's JavaScript
      console.error(
        `[DIAGNOSTIC] Puppeteer page event: pageerror. Uncaught JS exception in the page: ${pageErr.message}`,
        pageErr
      );
    });
    page.on("crash", () =>
      console.error(
        "[DIAGNOSTIC] Puppeteer page event: crash. The page renderer process has crashed."
      )
    );
    page.on("requestfailed", (request) => {
      if (request.url().startsWith("data:")) return; // Often not interesting
      console.warn(
        `[DIAGNOSTIC] Puppeteer page event: requestfailed. A network request made by the page failed: ${request.url()} (${
          request.failure()?.errorText
        })`
      );
    });

    const recordingWidth = viewportWidth;
    const recordingHeight = viewportHeight;

    await page.setViewport({
      width: viewportWidth,
      height: viewportHeight,
      deviceScaleFactor: 1,
    });

    console.log("Setting up recorder..."); // Re-enable recorder
    const recordingDurationSeconds = recordingDurationMs / 1000;

    // Get the dynamic settings
    const { recordingFps, videoCrf } = getRecordingSettings(
      recordingDurationSeconds
    );

    // Combine static settings with the fixed recording FPS
    const recorderOptions = {
      followNewTab: true,
      ffmpeg_Path: null,
      videoFrame: {
        width: recordingWidth,
        height: recordingHeight,
      },
      fps: recordingFps,
      videoCrf,
    };
    finalFps = recorderOptions.fps;

    console.log("Final recorder options:", recorderOptions);

    const recorder = new PuppeteerScreenRecorder(page, recorderOptions);

    const tempEventsUrlStr = "file://" + tempEventsPath.replace(/\\/g, "/");
    const htmlContent = `
            <html>
            <head>
                <meta charset="utf-8" />
                <title>Replay & Overlay Test</title>
                <link rel="stylesheet" href="${RRWEB_PLAYER_CSS_CDN}">
              <style>
            html, body {
                        width: 100vw; height: 100vh; margin: 0; padding: 0; background: black; overflow: hidden;
                    }


                    #frame {
                        width: ${viewportWidth}px;
                        height: ${viewportHeight}px;
                        background: black;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        overflow: hidden;
                        position: relative;
                    }
                    #player {
                        width: ${originalWidth}px;
                        height: ${originalHeight}px;
                        background: black;
                        position: absolute;
                        left: 50%;
                        top: 50%;
                        transform: translate(-50%, -50%) scale(${scale});
                        transform-origin: center;
                        border: 1px solid #444;
                        box-shadow: 0 0 25px rgba(0,0,0,0.8);
                        transition: all 0.3s ease;
                    }

                    .replayer-mouse.active:after {
  animation: none !important;
}
           

                         #externalSkipOverlayText {
            background: rgba(30, 30, 45, 0.7);
            padding: 25px 40px;
            border-radius: 12px;  
            border: 2px solid #5A5A62;
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
            min-width: 320px;
            min-height: 80px;
            display: inline-block;
          }

          /* ADD THIS NEW CSS RULE */
          #clip-marker-flash-overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            z-index: 2147483646; /* Very high z-index */
            display: none; /* Hidden by default */
            pointer-events: none; /* Don't interfere with other events */
            opacity: 0.7; /* Make it a semi-transparent flash */
          }


                          #beacon-overlay {
                      position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                      z-index: 2147483647; /* Max z-index */
                      display: none;
                      pointer-events: none;
                    }
                    /* Freeze overlay */
                    #freeze-overlay {
                          position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                        background: rgba(0,0,0,0.65); z-index: 999998; display: none;
                        pointer-events: none;
                    }
                    
                    #freeze-canvas {
                        display: none; position: fixed; top: 0; left: 0; z-index: 999997;
                    }
                    .rr-player {
                        width: 100% !important;
                        height: 100% !important;
                        background: black !important;
                    }

          .click-animation {
                        position: absolute;
                        border-radius: 50%;
                        width: 20px;
                        height: 20px;
                          background-color: rgba(0, 47, 255, 0.85);
                   /*      background-color: rgba(255, 0, 242, 1); */
                        transform: translate(-50%, -50%) scale(0);
                        animation: click-ripple 0.35s ease-out;
                        pointer-events: none;
                        z-index: 999999;
                    }
                    @keyframes click-ripple {
                        to {
                            transform: translate(-50%, -50%) scale(2);
                            opacity: 0;
                        }
                    }

                    /* Image preloading and optimization styles */
                    .image-placeholder {
                        transition: all 0.2s ease;
                        min-width: 40px;
                        min-height: 40px;
                    }
                    
                    .image-placeholder[data-status="loading"] {
                        background: linear-gradient(45deg, #f0f0f0 25%, transparent 25%), 
                                    linear-gradient(-45deg, #f0f0f0 25%, transparent 25%), 
                                    linear-gradient(45deg, transparent 75%, #f0f0f0 75%), 
                                    linear-gradient(-45deg, transparent 75%, #f0f0f0 75%);
                        background-size: 20px 20px;
                        background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
                        animation: placeholderShimmer 2s linear infinite;
                    }
                    
                    @keyframes placeholderShimmer {
                        0% { background-position: 0 0, 0 10px, 10px -10px, -10px 0px; }
                        100% { background-position: 20px 20px, 20px 30px, 30px 10px, 10px 20px; }
                    }
                    
                    /* Optimized image loading - ensure immediate display when ready */
                    img[data-optimized="true"] {
                        transition: opacity 0.1s ease;
                        will-change: opacity;
                    }
                    
                    img[data-optimized="true"][data-loaded="false"] {
                        opacity: 0;
                    }
                    
                    img[data-optimized="true"][data-loaded="true"] {
                        opacity: 1;
                    }
                    
                    /* Faster rendering hints for all images in replay */
                    .rr-player img {
                        image-rendering: -webkit-optimize-contrast;
                        image-rendering: optimize-contrast;
                        image-rendering: crisp-edges;
                        transform: translateZ(0); /* Force GPU acceleration */
                        backface-visibility: hidden;
                    }
                    
                    /* Background image optimization */
                    [data-bg-optimized="true"] {
                        transition: background-image 0.1s ease;
                        will-change: background-image;
                    }
                    
                    #externalSkipOverlay {
                        display: none; /* Start hidden */
                        position: fixed;
                        top: 0; left: 0; width: 100vw; height: 100vh;
                        right: 0; bottom: 0;
                        z-index: 2147483647;
                        background: rgba(0,0,0,0.3);
                        color: white;
                        font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif;
                        font-size: 1em;
                        font-weight: bold;
                        text-align: center;
                        pointer-events: none;
                        align-items: center;
                        justify-content: center;
                    }
                    #externalSkipOverlay span {
                        background: rgba(30, 30, 45, 0.7);
                        padding: 25px 40px;
                        border-radius: 12px;  
                        border: 2px solid #5A5A62;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
                        min-width: 320px;
                        min-height: 80px;
                        display: inline-block;
                    }
                        .click-indicator {
                        position: fixed !important;
                        width: 5px !important;
                        height: 5px !important;
                        border-radius: 50% !important;
                        background-color: #ff00ff !important;
                        box-shadow: 0 0 0px 0px rgb(252, 185, 252) !important;
                        z-index: 999999 !important;
                        pointer-events: none !important;
                        transform: translate(-50%, -50%) !important;
                    }
          </style>
            </head>
            <body>
                <script>
                  // Early global error logger to capture syntax and runtime error details
                  window.addEventListener('error', function(e) {
                    try {
                      console.error('[EARLY PAGE ERROR]', e.message, e.filename, e.lineno, e.colno);
                    } catch(_) {}

                    });
                    window.sessionClickVerifications = [];
                </script>
                <div id="frame">
                    <div id="player"></div>
                </div>
                     <div id="beacon-overlay"></div>
                <div id="freeze-overlay"></div>
                <canvas id="freeze-canvas"></canvas>
                <div id="externalSkipOverlay"><span id="externalSkipOverlayText"></span></div>
                  <div id="clip-marker-flash-overlay"></div>
                <script src="${rrwebPlayerFileUrl}"></script>
                <!-- ================================================================================= -->
                <!-- START: ROBUST PLAYER HOOK (subscribe to event-cast or wrap emit) -->
                <!-- ================================================================================= -->
                <script>
                  (function() {
                    console.log('[PLAYER-PATCH] Attempting to patch rrwebPlayer prototype...');

                    function attachMarkerFlash(core) {
                      if (!core) return false;
                      let isShowingMarkerOverlay = false; // State for pause markers

                      const handleEventCast = (eventData) => {
                        const flashOverlay = document.getElementById(
                          "clip-marker-flash-overlay"
                        );

                     

                        
// If a pause overlay is showing, hide it on the next non-marker event.
if (isShowingMarkerOverlay) {
    const isAnyMarker =
        eventData &&
        eventData.type === 5 &&
        eventData.data?.tag === "CLIP_MARKER";

    if (!isAnyMarker) {
        // The next "real" event has arrived. The pause is over. Hide the overlay immediately.
        if (flashOverlay) {
            console.log(
                "[MARKER OVERLAY] Hiding overlay on next non-marker event."
            );
            flashOverlay.style.display = "none";
        }
        isShowingMarkerOverlay = false;

        // Also clear any pending timeout to be safe
        const state = window.__markerOverlay || {};
        if (state.hideTimer) {
            clearTimeout(state.hideTimer);
            state.hideTimer = null;
            state.hideAt = null;
        }
    }
}

                        if (
                          eventData &&
                          eventData.type === 5 &&
                          eventData.data?.tag === "CLIP_MARKER"
                        ) {
                          const markerPayload = eventData.data.payload;
                          if (flashOverlay && markerPayload && markerPayload.color) {
                            // Pause for yellow and blue beacons by showing an overlay.
                            // The actual pause is created by shifting timestamps in the event stream.
                          if (
                              markerPayload.color === "yellow" ||
                              markerPayload.color === "blue"
                            ) {
                              const durRaw = markerPayload.displayDurationMs;
                              const hasExplicitDuration = Number.isFinite(Number(durRaw));
                              const displayMs = hasExplicitDuration
                                ? Number(durRaw)
                                : 2000;
                              try {
                                
                              } catch (_) {}
                              flashOverlay.style.backgroundColor = markerPayload.color;
                              flashOverlay.style.display = "block";
                              isShowingMarkerOverlay = true;
                              if (hasExplicitDuration) {
                                // Track hide schedule only when explicitly requested
                                if (!window.__markerOverlay) window.__markerOverlay = {};
                                const state = window.__markerOverlay;
                                state.hideAt = Date.now() + displayMs;
                                if (state.hideTimer) {
                                  try { clearTimeout(state.hideTimer); } catch (_) {}
                                }
                                state.hideTimer = setTimeout(() => {
                                  try { flashOverlay.style.display = "none"; } catch (_) {}
                                  isShowingMarkerOverlay = false;
                                }, displayMs);
                              }
                            } else {
                              // Original flash logic for any other colors
                              try {
                                console.log(
                                  "[MARKER FLASH] Detected",
                                  markerPayload.markerType,
                                  markerPayload.color
                                );
                              } catch (_) {}
                              flashOverlay.style.backgroundColor =
                                markerPayload.color;
                              flashOverlay.style.display = "block";
                              setTimeout(() => {
                                flashOverlay.style.display = "none";
                              }, 150);
                            }
                          }
                        }
                      };


                      if (typeof core.on === 'function') {
                        try { core.on('event-cast', handleEventCast); } catch(_) { return false; }
                        console.log('[PLAYER-PATCH] Subscribed to coreReplayer.on("event-cast").');
                        return true;
                      }

                      if (typeof core.emit === 'function') {
                        const originalEmit = core.emit;
                        core.emit = (event, payload) => {
                          originalEmit.call(core, event, payload);
                          if (event === 'event-cast') handleEventCast(payload);
                        };
                        console.log('[PLAYER-PATCH] Wrapped coreReplayer.emit to observe event-cast.');
                        return true;
                      }

                      return false;
                    }

                    // Wait for rrwebPlayer to be defined on the window
                    const interval = setInterval(() => {
                      if (typeof window.rrwebPlayer === 'function') {
                        clearInterval(interval);
                        console.log('[PLAYER-PATCH] rrwebPlayer is available. Patching now.');

                        const OriginalPlayer = window.rrwebPlayer;

                        // Create a new constructor that wraps the original
                        window.rrwebPlayer = function(config) {
                          const playerInstance = new OriginalPlayer(config);
                          console.log('[PLAYER-PATCH] Original rrwebPlayer instance created.');

                          try {
                            let core = null;
                            if (playerInstance && typeof playerInstance.getReplayer === 'function') {
                              try { core = playerInstance.getReplayer(); } catch(_) {}
                            }
                            // Fallbacks for different builds
                            core = core || playerInstance.replayer || playerInstance._replayer || window.coreReplayer || null;

                            if (!attachMarkerFlash(core)) {
                              console.warn('[PLAYER-PATCH] coreReplayer not ready; will retry shortly.');
                              const retryId = setInterval(() => {
                                try {
                                  const c = window.coreReplayer
                                    || (playerInstance && typeof playerInstance.getReplayer === 'function' ? playerInstance.getReplayer() : null)
                                    || playerInstance.replayer
                                    || playerInstance._replayer
                                    || null;
                                  if (attachMarkerFlash(c)) clearInterval(retryId);
                                } catch(_) {}
                              }, 50);
                              setTimeout(() => { try { clearInterval(retryId); } catch(_) {} }, 5000);
                            }
                          } catch (e) {
                            console.warn('[PLAYER-PATCH] Non-fatal: Unable to hook into replayer immediately:', e && e.message ? e.message : e);
                          }

                          return playerInstance; // return the wrapped instance
                        };
                      }
                    }, 10); // Check every 10ms
                  })();
                </script>

                <script>
                                    window.EVENTS_URL = "${tempEventsUrlStr}";
                    window.originalWidth = ${originalWidth};
                    window.originalHeight = ${originalHeight};
                </script>
                <script>
                  // Early bootstrap to ensure startRrwebReplay exists even if later scripts fail

                      // Verification logic ported from single-clip recorder
       function getReplayDocAndMirrors() {
            const iframe = document.querySelector(".replayer-wrapper > iframe");
            const doc =
              iframe && iframe.contentDocument
                ? iframe.contentDocument
                : document;
            const mirrors = [
              window.coreReplayer && window.coreReplayer.mirror,
              iframe &&
                iframe.contentWindow &&
                iframe.contentWindow.__rrweb &&
                iframe.contentWindow.__rrweb.mirror,
              window.__rrweb && window.__rrweb.mirror,
              window.__rrwebMirror,
            ].filter(Boolean);
            return { doc, mirrors, iframe };
          }

          
          function getNodeByRrwebId(rrwebId) {
            const { doc, mirrors } = getReplayDocAndMirrors();
            for (const m of mirrors) {
              try {
                if (m && typeof m.getNode === "function") {
                  const n = m.getNode(rrwebId);
                  if (n) return n;
                }
              } catch (_) {
                /* ignore */
              }
            }
            try {
              const q = doc.querySelector('[data-rr-id="' + rrwebId + '"]');
              if (q) return q;
            } catch (_) {
              /* ignore */
            }
            return null;
          }


        function performClickVerification(clickEvent) {
            const claimedTargetId = clickEvent?.data?.id ?? null;
            try {
              const result = {
                claimedTargetId: claimedTargetId ?? null,
                coords: null,
                context: null,
                visualId: null,
                visualElement: null,
                match: true,
                matchType: null,
                error: null,
                timestamp: clickEvent?.timestamp
              };

              // Prefer liveCoordinateData; fallback to indicator center
              let x = null;
              let y = null;
              let coordSource = 'none';
              if (window.liveCoordinateData?.finalCoordinates) {
                x = window.liveCoordinateData.finalCoordinates.x;
                y = window.liveCoordinateData.finalCoordinates.y;
                coordSource = 'liveCoordinateData';
              } else {
                const indicator = document.getElementById("click-indicator");
                if (indicator) {
                    const r = indicator.getBoundingClientRect();
                    x = r.left + r.width / 2;
                    y = r.top + r.height / 2;
                    coordSource = 'indicatorCenter';
                }
              }
              try { console.log('[PINK_DOT_DEBUG][VERIFY] coordsChosen', { source: coordSource, x, y }); } catch (_) {}

              if (x == null || y == null) {
                result.error = "no_coordinates";
                window.clickVerificationResult = result;
                return result;
              }

              result.coords = { x, y };

              // Ensure overlays don't intercept the hit test
              [
                "freeze-canvas",
                "freeze-overlay",
                "beacon-overlay",
                "click-indicator",
              ].forEach((id) => {
                const e = document.getElementById(id);
                if (e) e.style.pointerEvents = "none";
              });

                 // Detect rrweb iframe, if present
              const iframe = document.querySelector(
                ".replayer-wrapper > iframe"
              );
              let el = null;
              let doc = document;
              let iframeRectGlobal = null;
              // Determine any host-level scale applied to the rrweb player
              let hostScaleX = 1;
                            let hostScaleY = 1;
                try {
                  const playerContainerEl =
                    document.getElementById("player");
                  const playerEl = playerContainerEl
                    ? playerContainerEl.querySelector(".rr-player")
                    : document.querySelector(".rr-player");
                  if (playerEl) {
                    const playerRectHost = playerEl.getBoundingClientRect();
                    const ow = playerEl.offsetWidth || playerRectHost.width || 0;
                    const oh =
                      playerEl.offsetHeight || playerRectHost.height || 0;
                    if (ow) hostScaleX = playerRectHost.width / ow;
                    if (oh) hostScaleY = playerRectHost.height / oh;
                  }
                } catch (_) {}
  
                if (iframe && iframe.contentDocument) {
                  const iframeRect = iframe.getBoundingClientRect();
                  iframeRectGlobal = iframeRect;
                  // Map host coordinates back into iframe's local coordinate space,
                  // compensating for any host-level scale applied to the rrweb player.
                  let relX = (x - iframeRect.left) / hostScaleX;
                  let relY = (y - iframeRect.top) / hostScaleY;
                  console.log('[VERIFY][COORDS] host x=' + x + ', y=' + y + '; iframeRect left=' + iframeRect.left + ', top=' + iframeRect.top + '; hostScaleX=' + hostScaleX + ', hostScaleY=' + hostScaleY + '; relX=' + relX + ', relY=' + relY);
                  el = iframe.contentDocument.elementFromPoint(relX, relY);
                  doc = iframe.contentDocument;
                  result.context = "iframe";
                } else {
                  el = document.elementFromPoint(x, y);
                  result.context = "top";
                }

              
              // Resolve rrweb mirror and ids
              const iframeWin =
                iframe && iframe.contentWindow ? iframe.contentWindow : null;
              const mirrors = [
                window.coreReplayer && window.coreReplayer.mirror,
                iframeWin && iframeWin.__rrweb && iframeWin.__rrweb.mirror,
                window.__rrweb && window.__rrweb.mirror,
                window.__rrwebMirror,
              ].filter(Boolean);

              const getIdViaMirror = (node) => {
                for (const m of mirrors) {
                  try {
                    if (m && typeof m.getId === "function") {
                      const id = m.getId(node);
                      if (typeof id === "number") return id;
                    }
                  } catch (_) {}
                }
                return null;
              };

              // Fallback: walk up to find data-rr-id
              const getIdViaAttribute = (node) => {
                let cur = node;
                while (cur && cur.nodeType === 1) {
                  if (cur.getAttribute) {
                    const val = cur.getAttribute("data-rr-id");
                    if (val != null) return parseInt(val, 10);
                  }
                  cur = cur.parentElement;
                }
                return null;
              };

            const describe = (n) => {
                if (!n) return null;
                const classes = n.classList?.length
                  ? [...n.classList]
                      .filter((c) => !c.startsWith(':'))
                      .join('.')
                  : '';
                return (n.tagName?.toLowerCase() || '') +
                  (n.id ? '#' + n.id : '') +
                  (classes ? '.' + classes : '');
              };

              let visualId = null;
              if (el) {
                visualId = getIdViaMirror(el);
                if (visualId == null) visualId = getIdViaAttribute(el);
              }
              result.visualId = visualId;
              result.visualElement = describe(el);
              if (!el) {
                console.warn('[VERIFY][COORDS] elementFromPoint returned null. context=' + result.context);
                if (iframeRectGlobal) {
                  console.warn('[VERIFY][COORDS] iframeRect: left=' + iframeRectGlobal.left + ', top=' + iframeRectGlobal.top + ', w=' + iframeRectGlobal.width + ', h=' + iframeRectGlobal.height);
                }
              }

              let matchType = null;
              let claimedEl = null;

                           if (claimedTargetId != null) {
                claimedEl = getNodeByRrwebId(claimedTargetId);
              }
   if (claimedTargetId != null && visualId === claimedTargetId) {
                matchType = "exact";
              } else if (claimedTargetId != null && el) {
                if (claimedEl && claimedEl.contains(el)) {
                   matchType = "descendant";
                }
              }

              result.matchType = matchType;
              result.match = Boolean(matchType);

              const getRectOrNull = (node) =>
                node && typeof node.getBoundingClientRect === "function"
                  ? node.getBoundingClientRect()
                  : null;

              let claimedNodeForCheck = claimedEl;
              if (claimedNodeForCheck && claimedNodeForCheck.nodeType === 3) {
                claimedNodeForCheck = claimedNodeForCheck.parentElement;
              }
              const baseRect = getRectOrNull(claimedNodeForCheck);
              const claimedRect = baseRect
                ? iframeRectGlobal
                  ? {
                      left: iframeRectGlobal.left + baseRect.left * hostScaleX,
                      top: iframeRectGlobal.top + baseRect.top * hostScaleY,
                      right:
                        iframeRectGlobal.left + baseRect.right * hostScaleX,
                      bottom:
                        iframeRectGlobal.top + baseRect.bottom * hostScaleY,
                      width: baseRect.width * hostScaleX,
                      height: baseRect.height * hostScaleY,
                    }
                  : {
                      left: baseRect.left,
                      top: baseRect.top,
                      right: baseRect.right,
                      bottom: baseRect.bottom,
                      width: baseRect.width,
                      height: baseRect.height,
                    }
                : null;

              // Mirror click-clip containment logic
              const geometryTolerance = 0; // strict
              let pointInClaimedByGeometry = false;
              let containingTightRect = null;
              try {
                const getClientRectsArray = (node) => {
                  try {
                    if (node && typeof node.getClientRects === 'function') {
                      return Array.from(node.getClientRects());
                    }
                  } catch (_) {}
                  return [];
                };
                let claimedNodeForCheckEffective = claimedNodeForCheck;
                const tightRects = getClientRectsArray(claimedNodeForCheckEffective).map((r) => ({
                  left: iframeRectGlobal ? iframeRectGlobal.left + r.left * hostScaleX : r.left,
                  top: iframeRectGlobal ? iframeRectGlobal.top + r.top * hostScaleY : r.top,
                  right: iframeRectGlobal ? iframeRectGlobal.left + r.right * hostScaleX : r.right,
                  bottom: iframeRectGlobal ? iframeRectGlobal.top + r.bottom * hostScaleY : r.bottom,
                  width: iframeRectGlobal ? r.width * hostScaleX : r.width,
                  height: iframeRectGlobal ? r.height * hostScaleY : r.height,
                }));
                if (tightRects.length > 0) {
                  for (const r of tightRects) {
                    const inside =
                      x >= r.left - geometryTolerance &&
                      x <= r.right + geometryTolerance &&
                      y >= r.top - geometryTolerance &&
                      y <= r.bottom + geometryTolerance;
                    if (inside) {
                      pointInClaimedByGeometry = true;
                      containingTightRect = r;
                      break;
                    }
                  }
                } else {
                  pointInClaimedByGeometry = claimedRect
                    ? x >= claimedRect.left - geometryTolerance &&
                      x <= claimedRect.right + geometryTolerance &&
                      y >= claimedRect.top - geometryTolerance &&
                      y <= claimedRect.bottom + geometryTolerance
                    : false;
                }
              } catch (_) {}

              let forcedOutside = false;
              const effectiveRect = containingTightRect || claimedRect;
              if (effectiveRect) {
                const dxOutside =
                  x < effectiveRect.left
                    ? effectiveRect.left - x
                    : x > effectiveRect.right
                    ? x - effectiveRect.right
                    : 0;
                const dyOutside =
                  y < effectiveRect.top
                    ? effectiveRect.top - y
                    : y > effectiveRect.bottom
                    ? y - effectiveRect.bottom
                    : 0;
                forcedOutside = dxOutside > 0 || dyOutside > 0;
              }
              let pointInClaimed = pointInClaimedByGeometry && !forcedOutside;

              if (!matchType && pointInClaimed) {
                matchType = 'geometric';
                result.match = true;
                result.matchType = 'geometric';
              }

              result.pointInClaimed = pointInClaimed;
              const persistedRect = containingTightRect || effectiveRect;
              result.claimedRect = persistedRect || null;
              if (persistedRect) {
                  const dx =
                    x < persistedRect.left
                      ? persistedRect.left - x
                      : x > persistedRect.right
                      ? x - persistedRect.right
                      : 0;
                  const dy =
                    y < persistedRect.top
                      ? persistedRect.top - y
                      : y > persistedRect.bottom
                      ? y - persistedRect.bottom
                      : 0;
                  result.distanceToClaimedRect = { dx, dy };
                  if (
                    x >= persistedRect.left &&
                    x <= persistedRect.right &&
                    y >= persistedRect.top &&
                    y <= persistedRect.bottom &&
                    persistedRect.width &&
                    persistedRect.height
                  ) {
                    result.relativePositionInClaimed = {
                      x: (x - persistedRect.left) / persistedRect.width,
                      y: (y - persistedRect.top) / persistedRect.height,
                    };
                  }
              }
              
              try {
                const viewportW = window.innerWidth || 0;
                const viewportH = window.innerHeight || 0;
                if (viewportW > 0 && viewportH > 0 && (result.claimedRect)) {
                  const cr = result.claimedRect;
                  const rectLeft = typeof cr.left === "number" ? cr.left : 0;
                  const rectTop = typeof cr.top === "number" ? cr.top : 0;
                  const rectRight = typeof cr.right === "number" ? cr.right : rectLeft + (cr.width || 0);
                  const rectBottom = typeof cr.bottom === "number" ? cr.bottom : rectTop + (cr.height || 0);

                  const interLeft = Math.max(0, rectLeft);
                  const interTop = Math.max(0, rectTop);
                  const interRight = Math.min(viewportW, rectRight);
                  const interBottom = Math.min(viewportH, rectBottom);

                  const interW = Math.max(0, interRight - interLeft);
                  const interH = Math.max(0, interBottom - interTop);
                  const visibleAreaPx = interW * interH;
                  const elementAreaPx = Math.max(0, ((cr.width || rectRight - rectLeft) * (cr.height || rectBottom - rectTop)));
                  const percentVisible = elementAreaPx > 0 ? visibleAreaPx / elementAreaPx : 0;

                  result.highlightVisibility = {
                    visibleInViewport: visibleAreaPx > 0,
                    percentVisible,
                    visibleAreaPx,
                    elementAreaPx,
                    viewport: { width: viewportW, height: viewportH },
                  };
                }
              } catch (_) {}

              return result;
            } catch (e) {
              const err = { error: String(e?.message || e), timestamp: clickEvent?.timestamp };
              console.warn("[VERIFY] Error during click verification", e);
              return err;
            }
          }
                  window.startRrwebReplay = window.startRrwebReplay || (async function() {
                    try {
                      let events = window.RRWEB_EVENTS;
                      if (!events || !Array.isArray(events) || events.length === 0) {
                        if (window.EVENTS_URL) {
                          console.log('[BOOT] Fetching events from', window.EVENTS_URL);
                          try {
                            const resp = await fetch(window.EVENTS_URL);
                            events = await resp.json();
                            window.RRWEB_EVENTS = events;
                          } catch (e) {
                            console.warn('[BOOT] Fetch failed, will rely on injected RRWEB_EVENTS if available.', e && e.message ? e.message : e);
                          }
                        }
                      }
                      if (!Array.isArray(events) || events.length === 0) {
                        console.error('[BOOT] No events available to start replay.');
                        return;
                      }
                      try {
                        const metas = Array.isArray(events)
                          ? events.filter((e) => e && e.type === 4 && e.data && e.data.width && e.data.height)
                          : [];
                        const firstMeta = metas[0];
                        const lastMeta = metas[metas.length - 1];
                        console.log(
                          "[VIEWPORT_DEBUG][PAGE] metaCount=" + metas.length
                        );
                        if (firstMeta)
                          console.log(
                            '[VIEWPORT_DEBUG][PAGE] firstMeta:',
                            {
                              ts: firstMeta.timestamp,
                              iso: new Date(firstMeta.timestamp).toISOString(),
                              w: firstMeta.data.width,
                              h: firstMeta.data.height,
                            }
                          );
                        if (lastMeta && lastMeta !== firstMeta)
                          console.log(
                            '[VIEWPORT_DEBUG][PAGE] lastMeta:',
                            {
                              ts: lastMeta.timestamp,
                              iso: new Date(lastMeta.timestamp).toISOString(),
                              w: lastMeta.data.width,
                              h: lastMeta.data.height,
                            }
                          );
                        console.log(
                          '[VIEWPORT_DEBUG][PAGE] dpr/window:',
                          {
                            devicePixelRatio: window.devicePixelRatio,
                            innerWidth: window.innerWidth,
                            innerHeight: window.innerHeight,
                          }
                        );
                      } catch (_) {}
                      const playerElement = document.getElementById('player');
                      if (!playerElement) {
                        console.error('[BOOT] Player element not found.');
                        return;
                      }
                    const replayer = new rrwebPlayer({
                        target: playerElement,
                        props: {
                          events: events,
                          width: ${originalWidth},
                          height: ${originalHeight},
                          autoPlay: false,
                          mouseTail: true,
                          speed: 1,
                          skipInactive: false,
                          showController: false,
                          iframeSandbox: 'allow-scripts allow-same-origin'
                        }
                      });
                     window.replayer = replayer;
                       console.log('[BOOT] rrwebPlayer initialized');
 
                        try {
                        if (typeof replayer.getReplayer === 'function') {
                           window.coreReplayer = replayer.getReplayer();


  // Listen for rrweb events to move the custom cursor dot
                            window.coreReplayer.on('event-cast', (event) => {

 try {
                        if (
                          event &&
                          event.type === 3 && // Incremental snapshot
                          event.data &&
                          event.data.source === 2 && // MouseInteraction
                          event.data.type === 2 && // Click
                          event.data.isTargetClick === true
                        ) {
                          const x = event.data.x;
                          const y = event.data.y;
                          const playerEl = document.getElementById('player');
                          
                          if (playerEl && typeof x === 'number' && typeof y === 'number') {
                            const animationEl = document.createElement('div');
                            animationEl.className = 'click-animation';
                            animationEl.style.left = x + "px";
                            animationEl.style.top = y + "px";
                            playerEl.appendChild(animationEl);
                            
                            setTimeout(() => {
                              if (animationEl && animationEl.parentNode) {
                                animationEl.parentNode.removeChild(animationEl);
                              }
                            }, 500); // Match animation duration
                          }
                        }
                      } catch (e) {
                        console.error('Error creating click animation:', e);
                      }
})


                           // Define the freeze function and its helpers
                         window.freezePlayerAtClick = async function(clickEvent) {
                               if (window.isPlayerFrozen) return;
                               window.isPlayerFrozen = true;
                               if (clickEvent && clickEvent.data && typeof clickEvent.data.x === 'number' && typeof clickEvent.data.y === 'number') {
                                   try {
                                       const rawClickX = clickEvent.data.x;
                                       const rawClickY = clickEvent.data.y;
                                       let finalClickX = 640;
                                       let finalClickY = 360;
                                       let coordinateSource = "default-center";

                                       try {
                                           console.log('[PINK_DOT_DEBUG][MAIN] freezePlayerAtClick invoked');
                                           console.log('[PINK_DOT_DEBUG][MAIN] Raw rrweb coords:', rawClickX, rawClickY);
                                           console.log('[PINK_DOT_DEBUG][MAIN] Original content size:', window.originalWidth, 'x', window.originalHeight);
                                           console.log('[PINK_DOT_DEBUG][MAIN] Viewport:', window.innerWidth, 'x', window.innerHeight);
                                       } catch(_) {}

                                       if (rawClickX !== null && rawClickY !== null) {
                                           const playerContainer = document.getElementById("player-container");
                                           const playerElement = (playerContainer && playerContainer.querySelector(".rr-player"))
                                               || document.querySelector("#player .rr-player")
                                               || document.querySelector(".rr-player");
                                           if (playerElement) {
                                               const contentSelectors = ["iframe", ".replayer-wrapper", ".replayer-mirror", ".rr-player__frame", ".rr-player > div", ".rr-player"];
                                               let contentElement = null;
                                               // Prefer querying inside the rrweb player; fall back to document if needed
                                               const searchRoots = [playerElement, document];
                                               for (const root of searchRoots) {
                                                   if (!root) continue;
                                                   for (const selector of contentSelectors) {
                                                       const element = (root === document ? document : root).querySelector(selector);
                                                       if (element) { contentElement = element; break; }
                                                   }
                                                   if (contentElement) break;
                                               }
                                               if (!contentElement) contentElement = playerElement || document.querySelector("#player") || document.body;

                                               try {
                                                   const playerBox = document.getElementById('player');
                                                   const tf = playerBox ? getComputedStyle(playerBox).transform : null;
                                                   console.log('[PINK_DOT_DEBUG][MAIN] #player CSS transform:', tf);
                                               } catch (_) {}
                                               const playerRect = playerElement.getBoundingClientRect();
                                               const contentRect = contentElement.getBoundingClientRect();
                                               try {
                                                   const desc = (el) => {
                                                       if (!el) return 'null';
                                                       const t = (el.tagName || '').toLowerCase();
                                                       const id = el.id ? '#' + el.id : '';
                                                       let cls = '';
                                                       try {
                                                           cls = el.classList && el.classList.length ? '.' + [...el.classList].join('.') : '';
                                                       } catch (_) {}
                                                       return t + id + cls;
                                                   };
                                                   console.log('[PINK_DOT_DEBUG][MAIN] Using contentElement:', desc(contentElement));
                                               } catch (_) {}

                                   try {
                                                   console.log('[PINK_DOT_DEBUG][MAIN] Player rect:', JSON.stringify({left: playerRect.left, top: playerRect.top, width: playerRect.width, height: playerRect.height}));
                                                   console.log('[PINK_DOT_DEBUG][MAIN] Content rect:', JSON.stringify({left: contentRect.left, top: contentRect.top, width: contentRect.width, height: contentRect.height}));
                                              const hasLetterboxing = contentRect.width !== playerRect.width || contentRect.height !== playerRect.height || contentRect.left !== playerRect.left || contentRect.top !== playerRect.top;
                                                 console.log('[PINK_DOT_DEBUG][MAIN] Letterboxing detected:', hasLetterboxing);
                                             } catch(_) {}
                                             
                                             let workingOriginalW = window.originalWidth || 1920;
                                             let workingOriginalH = window.originalHeight || 1080;
                                             
                                             // Use the live resolution if available, which accounts for mid-clip changes
                                             try {
                                                 if (window.currentActiveResolution && typeof window.currentActiveResolution.width === 'number' && typeof window.currentActiveResolution.height === 'number') {
                                                     workingOriginalW = window.currentActiveResolution.width;
                                                     workingOriginalH = window.currentActiveResolution.height;
                                                 }
                                             } catch (_) {}
                                             
                                             // CORRECTED LOGIC: Use player's scale, but content's offset.
                                             // This correctly handles letterboxing when content is centered.
                                             const playerScaleX = playerRect.width / workingOriginalW;
                                             const playerScaleY = playerRect.height / workingOriginalH;
    
                                             // Use contentRect's offset to account for letterboxing (content centered in player)
                                             const contentOffsetX = contentRect.left;
                                             const contentOffsetY = contentRect.top;
                                             
                                             finalClickX = contentOffsetX + rawClickX * playerScaleX;
                                             finalClickY = contentOffsetY + rawClickY * playerScaleY;
                                             coordinateSource = "live-browser-calculation";
    
                                             try {
                                                 console.log('[PINK_DOT_DEBUG][MAIN] Final calculated position:', JSON.stringify({x: finalClickX, y: finalClickY, coordinateSource}));
                                                 console.log('[PINK_DOT_DEBUG][MAIN] Position as viewport %:', JSON.stringify({xPct: (finalClickX / window.innerWidth) * 100, yPct: (finalClickY / window.innerHeight) * 100}));
                                                   } catch(_) {}

                                               const liveData = {
                                                   playerRect: { left: playerRect.left, top: playerRect.top, width: playerRect.width, height: playerRect.height },
                                                   contentRect: { left: contentRect.left, top: contentRect.top, width: contentRect.width, height: contentRect.height },
                                                   finalCoordinates: { x: finalClickX, y: finalClickY },
                                               };
                                               window.liveCoordinateData = liveData;
                                               try { console.log('[PINK_DOT_DEBUG][MAIN] Stored liveCoordinateData'); } catch(_) {}
                                           }
                                       }

                                       const indicator = document.createElement('div');
                                       indicator.className = 'click-indicator';
                                       indicator.style.left = finalClickX + 'px';
                                       indicator.style.top = finalClickY + 'px';
                                       indicator.id = 'click-indicator';
                                       document.body.appendChild(indicator);

                                       try {
                                           console.log('[PINK_DOT_DEBUG][MAIN] Indicator appended at:', finalClickX, finalClickY, 'source=', coordinateSource);
                                           setTimeout(() => {
                                               try {
                                                   const r = indicator.getBoundingClientRect();
                                                   const actualCenter = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
                                                   console.log('[PINK_DOT_DEBUG][MAIN] Indicator DOM rect:', JSON.stringify({left: r.left, top: r.top, width: r.width, height: r.height}));
                                                   console.log('[PINK_DOT_DEBUG][MAIN] Intended vs actual center:', JSON.stringify({intended: {x: finalClickX, y: finalClickY}, actual: actualCenter, dx: actualCenter.x - finalClickX, dy: actualCenter.y - finalClickY}));
                                               } catch(_) {}
                                           }, 10);
                                       } catch(_) {}
                                   } catch (e) {
                                       console.error('[PINK_DOT_DEBUG] Error adding click indicator:', e);
                                   }
                               }
                               const beaconOverlay = document.getElementById("beacon-overlay");

                               const freezeOverlay = document.getElementById("freeze-overlay");
                               const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
                               
                               const showBeacon = async (color) => {
                                   if (beaconOverlay) {
                                       beaconOverlay.style.backgroundColor = color;
                                       beaconOverlay.style.display = "block";
                                       await new Promise(r => requestAnimationFrame(r));
                                   }
                               };
                               const hideBeacon = () => {
                                   if (beaconOverlay) beaconOverlay.style.display = "none";
                               };
                               try {
                                   // --- ASYNC/AWAIT FREEZE SEQUENCE ---
                                   await showBeacon("rgba(0, 255, 0, 1)"); // GREEN
                                   await sleep(450);
                                   hideBeacon();
                                  if (freezeOverlay) freezeOverlay.style.display = "block";
                                   
                                   // Attempt click verification during the freeze hold
                                   let claimedNodeForHighlight = null;
                                   try {
                                       if (typeof performClickVerification === 'function') {
                                           const verifyResultDuringHold = performClickVerification(clickEvent);
                                           try { window.sessionClickVerifications = window.sessionClickVerifications || []; } catch(_) {}
                                           try { window.sessionClickVerifications.push(verifyResultDuringHold); } catch(_) {}
                                           try { console.log('[VERIFY][MAIN] Verification during freeze:', JSON.stringify(verifyResultDuringHold)); } catch(_) {}
                                           
                                           // NEW: If containment fails, highlight the claimed element's border
                                           if (verifyResultDuringHold && verifyResultDuringHold.pointInClaimed === false && verifyResultDuringHold.claimedTargetId != null) {
                                               console.log('[VERIFY][MAIN] pointInClaimed is false. Highlighting claimed element with ID:', verifyResultDuringHold.claimedTargetId);
                                               if (typeof getNodeByRrwebId === 'function') {
                                                   claimedNodeForHighlight = getNodeByRrwebId(verifyResultDuringHold.claimedTargetId);
                                                   if (claimedNodeForHighlight) {
                                                        if (claimedNodeForHighlight.nodeType === 3) { // If it's a text node, get parent
                                                            claimedNodeForHighlight = claimedNodeForHighlight.parentElement;
                                                        }
                                                        if (claimedNodeForHighlight && claimedNodeForHighlight.style) {
                                                            claimedNodeForHighlight.style.outline = '3px solid orange';
                                                            claimedNodeForHighlight.style.outlineOffset = '2px';
                                                            claimedNodeForHighlight.style.boxShadow = '0 0 15px orange';
                                                        }
                                                   }
                                               }
                                           }
                                       } else {
                                           console.warn('[VERIFY][MAIN] performClickVerification is not available in page context');
                                       }
                                   } catch (verErr) {
                                       console.warn('[VERIFY][MAIN] Verification attempt failed during freeze:', verErr);
                                   }

                                   await sleep(3000); // 3-second freeze
                                   
                                   // NEW: Remove highlight after freeze
                                   if (claimedNodeForHighlight && claimedNodeForHighlight.style) {
                                       claimedNodeForHighlight.style.outline = '';
                                       claimedNodeForHighlight.style.outlineOffset = '';
                                       claimedNodeForHighlight.style.boxShadow = '';
                                   }
                                   
                                   if (freezeOverlay) freezeOverlay.style.display = "none";
                                   await showBeacon("rgba(255, 0, 0, 1)"); // RED
                                   await sleep(450);
                               } finally {
                                   hideBeacon();
                                   const indicator = document.getElementById('click-indicator');
                                   if (indicator) {
                                       indicator.remove();
                                   }
                                   window.isPlayerFrozen = false;
                               }
                           };
                           
                          // PROACTIVE FREEZE LOGIC FOR MAIN SESSION VIDEO
                           let pendingClickEvents = [...(window.clickEventsForFreeze || [])].sort((a, b) => a.timestamp - b.timestamp);
                            let isProactivelyFreezing = false;
                            const FREEZE_LEAD_TIME_MS = 350; // Freeze 250 before the click

                            // High-frequency check for upcoming clicks using requestAnimationFrame for performance
                           function checkForUpcomingClicks() {
                                if (isProactivelyFreezing || pendingClickEvents.length === 0 || !window.coreReplayer || !window.RRWEB_EVENTS || window.RRWEB_EVENTS.length === 0) {
                                    // If no more clicks, stop the loop.
                                    if (pendingClickEvents.length === 0) return;
                                    requestAnimationFrame(checkForUpcomingClicks);
                                    return;
                                }

                                try {
                                    const currentTime = window.coreReplayer.getCurrentTime();
                                    const firstEventTs = window.RRWEB_EVENTS[0].timestamp;
                                    const absolutePlayerTime = firstEventTs + currentTime;
                                    
                                    const nextClickEvent = pendingClickEvents[0];
                                    const nextClickTs = nextClickEvent.timestamp;
                                    const timeToClick = nextClickTs - absolutePlayerTime;

                                    if (timeToClick > 0 && timeToClick <= FREEZE_LEAD_TIME_MS) {
                                        isProactivelyFreezing = true;
                                        const clickToProcess = pendingClickEvents.shift(); // Consume the click
                                        
                                        (async () => {
                                            try {
                                                if (window.coreReplayer.speedService) {
                                                    window.coreReplayer.speedService.send({ type: 'SET_SPEED', payload: { speed: 0 } });
                                                } else if (typeof window.coreReplayer.setConfig === 'function') {
                                                    window.coreReplayer.setConfig({ speed: 0 });
                                                }
                                                
                                                await window.freezePlayerAtClick(clickToProcess);

                                            } catch (e) {
                                                console.error('Proactive freeze sequence failed:', e);
                                            } finally {
                                                 try {
                                                    if (window.coreReplayer.speedService) {
                                                        window.coreReplayer.speedService.send({ type: 'SET_SPEED', payload: { speed: 1 } });
                                                    } else if (typeof window.coreReplayer.setConfig === 'function') {
                                                        window.coreReplayer.setConfig({ speed: 1 });
                                                    }
                                                } catch(e) { console.error('Failed to resume player after proactive freeze:', e); }
                                                isProactivelyFreezing = false;
                                            }
                                        })();
                                    }
                                } catch(e) {
                                    // Could fail if player is not ready, just ignore
                                }

                                // Continue the loop
                                requestAnimationFrame(checkForUpcomingClicks);
                            }
                            // Start the check loop
                            requestAnimationFrame(checkForUpcomingClicks);



                         } else {
                           console.error('[FREEZE-SETUP] replayer.getReplayer is not a function!');
                         }
                       } catch (e) {
                          console.error('[FREEZE-SETUP] Error getting coreReplayer or setting up freeze logic:', e);
                       }

                       try{ 
                        if (!window.__speedLockIntervalMain) {
                          const enforceSpeed1x = () => {
                            try {
                              const r = window.coreReplayer || window.replayer || null;
                              if (!r) return;
                              const currentSpeed = r.config.speed;
                              if (currentSpeed !== 1) {
                              
                                if (r.speedService) {
                                  r.speedService.send({ type: 'SET_SPEED', payload: { speed: 1 } });
                                } else if (typeof r.setConfig === 'function') {
                                  r.setConfig({ speed: 1 });
                                }
                              }
                            } catch (_) {}
                          };

                          // Initial application and periodic re-application
                          enforceSpeed1x();
                          window.__speedLockIntervalMain = setInterval(enforceSpeed1x, 1500);
                        }
                      } catch (_) {}
                    } catch (e) {
                      console.error('[BOOT] Error in startRrwebReplay:', e && e.message ? e.message : e);
                    }
                  });
                </script>
                
              
            </body>
            </html>
        `;

    await fs.writeFile(tempFilePath, htmlContent, "utf-8");
    console.log(`Temporary player HTML written to ${tempFilePath}`);

    const tempFileUrl = "file://" + tempFilePath.replace(/\\/g, "/");
    const tempEventsUrl = "file://" + tempEventsPath.replace(/\\/g, "/");

    console.log("[NAV] About to navigate to local player HTML:", tempFileUrl);
    try {
      await page.goto(tempFileUrl, {
        timeout: 6000000,
        waitUntil: "networkidle0", // wait for all network requests to finish
      });
      console.log("Navigated to local player HTML file.");
    } catch (navErr) {
      console.error("[NAV] Navigation failed:", navErr?.message || navErr);
      throw navErr;
    }

    // DIAGNOSTIC: Log viewport and window size from inside Puppeteer
    const viewportDiagnostics = await page.evaluate(() => {
      return {
        windowInnerWidth: window.innerWidth,
        windowInnerHeight: window.innerHeight,
        docClientWidth: document.documentElement.clientWidth,
        docClientHeight: document.documentElement.clientHeight,
        bodyClientWidth: document.body.clientWidth,
        bodyClientHeight: document.body.clientHeight,
      };
    });
    console.log(
      "[DIAGNOSTIC] Viewport/Window sizes as seen by browser:",
      viewportDiagnostics
    );

    // Force html/body to fill viewport and remove margin/padding
    await page.addStyleTag({
      content: `html, body { width: 100vw !important; height: 100vh !important; margin: 0 !important; padding: 0 !important; overflow: hidden !important; background: black !important; }`,
    });

    // pageerror already handled above

    // Wait longer to ensure CDN script is fully loaded (especially for clips_folder)
    const isClip = !overlayInstructions || overlayInstructions.length === 0;
    const waitTime = isClip ? 5000 : 2000; // Wait longer for clips_folder
    console.log(
      `Waiting ${waitTime}ms for CDN loading (${
        isClip ? "CLIP" : "MAIN VIDEO"
      })...`
    );
    await page.waitForTimeout(waitTime);

    // Verify rrweb is actually loaded with more details
    const rrwebLoaded = await page.evaluate(() => {
      try {
        const keys = Object.keys(window).filter((k) =>
          k.toLowerCase().includes("rrweb")
        );
        return {
          rrwebPlayerExists: typeof rrwebPlayer !== "undefined",
          rrwebPlayerType: typeof rrwebPlayer,
          hasStartFn: typeof window.startRrwebReplay === "function",
          rrwebEventsType: Array.isArray(window.RRWEB_EVENTS)
            ? "array"
            : typeof window.RRWEB_EVENTS,
          rrwebEventsLen: Array.isArray(window.RRWEB_EVENTS)
            ? window.RRWEB_EVENTS.length
            : 0,
          windowRrwebKeys: keys,
          docReadyState: document.readyState,
        };
      } catch (e) {
        return { error: e && e.message ? e.message : String(e) };
      }
    });

    console.log(`[DIAGNOSTIC] rrweb loading check:`, rrwebLoaded);

    if (!rrwebLoaded.rrwebPlayerExists) {
      console.error(
        `[DIAGNOSTIC] rrwebPlayer not loaded from CDN after ${waitTime}ms wait`
      );
      console.error(
        `[DIAGNOSTIC] Available rrweb-related globals:`,
        rrwebLoaded.windowRrwebKeys
      );
    }

    console.log("Setting up rrweb player and overlay cycle...");

    // Read the events from the JSON file
    const eventsData = await fs.readFile(tempEventsPath, "utf-8");
    const eventsToReplay = JSON.parse(eventsData);

    // Inject events into the page context in chunks to avoid oversized payload and token issues
    try {
      await page.evaluate(() => {
        window.RRWEB_EVENTS = [];
      });
      const __chunkSize = 100;
      for (let __i = 0; __i < eventsToReplay.length; __i += __chunkSize) {
        const __chunk = eventsToReplay.slice(__i, __i + __chunkSize);
        await page.evaluate(
          (chunk, startIndex, total) => {
            try {
              if (!Array.isArray(window.RRWEB_EVENTS)) window.RRWEB_EVENTS = [];
              for (const ev of chunk) window.RRWEB_EVENTS.push(ev);
              if (startIndex === 0) {
                console.log("[INJECT] Initialized RRWEB_EVENTS array.");
              }
              if (startIndex + chunk.length >= total) {
                console.log(
                  "[INJECT] Completed events injection. total=",
                  total
                );
              }
            } catch (e) {
              console.error(
                "[INJECT] Error while pushing chunk:",
                e && e.message ? e.message : e
              );
            }
          },
          __chunk,
          __i,
          eventsToReplay.length
        );
        if (__i === 0 || __i + __chunkSize >= eventsToReplay.length) {
          console.log(
            `[DIAGNOSTIC] Injecting events chunk: start=${__i} size=${__chunk.length}`
          );
        }
      }
      const verify = await page.evaluate(() => ({
        type: Array.isArray(window.RRWEB_EVENTS)
          ? "array"
          : typeof window.RRWEB_EVENTS,
        len: Array.isArray(window.RRWEB_EVENTS)
          ? window.RRWEB_EVENTS.length
          : 0,
      }));
      console.log("[DIAGNOSTIC] Post-injection verify:", verify);
      console.log(
        `[DIAGNOSTIC] Injected RRWEB_EVENTS into page context in ${Math.ceil(
          eventsToReplay.length / __chunkSize
        )} chunks (total=${eventsToReplay.length}).`
      );
    } catch (injectErr) {
      console.warn(
        "[DIAGNOSTIC] Failed to inject RRWEB_EVENTS into page (chunked):",
        injectErr?.message || injectErr
      );
    }

    // Enhanced event validation before setup
    console.log(`[DEBUG] Events validation before setup:`);
    console.log(`  - Total events: ${eventsToReplay.length}`);
    console.log(`  - First event type: ${eventsToReplay[0]?.type}`);
    console.log(
      `  - Last event type: ${eventsToReplay[eventsToReplay.length - 1]?.type}`
    );
    console.log(
      `  - Has FullSnapshot: ${eventsToReplay.some((e) => e.type === 2)}`
    );
    console.log(
      `  - Event types: ${[...new Set(eventsToReplay.map((e) => e.type))].join(
        ", "
      )}`
    );

    // Check for common issues that could cause rrweb player to fail
    const hasRequiredFields = eventsToReplay.every(
      (event) =>
        event.type !== undefined &&
        event.timestamp !== undefined &&
        event.data !== undefined
    );

    if (!hasRequiredFields) {
      console.error(
        `[DEBUG] Events are missing required fields (type, timestamp, data)`
      );
      const invalidEvent = eventsToReplay.find(
        (event) =>
          event.type === undefined ||
          event.timestamp === undefined ||
          event.data === undefined
      );
      console.error(
        `[DEBUG] Sample invalid event:`,
        JSON.stringify(invalidEvent, null, 2)
      );
    }
    // MODIFIED: Events are now injected, so we only pass instructions and timestamp
    const setupResult = await page.evaluate(
      async (instructions, firstTs, clickEventsForFreeze) => {
        try {
          console.log("[PUPPETEER CONTEXT] Starting rrweb player setup...");
          window.clickEventsForFreeze = clickEventsForFreeze;
          console.log(
            "[PUPPETEER CONTEXT] rrwebPlayer available:",
            typeof rrwebPlayer
          );
          try {
            console.log("[PUPPETEER CONTEXT] Pre-setup globals:", {
              hasStartFn: typeof window.startRrwebReplay === "function",
              rrwebEventsType: Array.isArray(window.RRWEB_EVENTS)
                ? "array"
                : typeof window.RRWEB_EVENTS,
              rrwebEventsLen: Array.isArray(window.RRWEB_EVENTS)
                ? window.RRWEB_EVENTS.length
                : 0,
              keys: Object.keys(window)
                .filter((k) => k.toLowerCase().includes("rrweb"))
                .slice(0, 10),
            });
          } catch (_) {}

          window.replayStarted = false; // Initialize flag

          if (window.startRrwebReplay) {
            console.log("[PUPPETEER CONTEXT] Calling startRrwebReplay...");
            await window.startRrwebReplay(); // Events are now on window.RRWEB_EVENTS
            console.log("[PUPPETEER CONTEXT] startRrwebReplay completed");
          } else {
            console.error(
              "[PUPPETEER CONTEXT] window.startRrwebReplay not defined."
            );
            return {
              error: "window.startRrwebReplay not defined",
            };
          }

          // Debug helper: dump key layout metrics
          function __dumpRect(el) {
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {
              left: r.left,
              top: r.top,
              width: r.width,
              height: r.height,
            };
          }
          function __dumpStyle(el) {
            if (!el) return null;
            const cs = window.getComputedStyle(el);
            return {
              position: cs.position,
              left: cs.left,
              top: cs.top,
              width: cs.width,
              height: cs.height,
              transform: cs.transform,
              transformOrigin: cs.transformOrigin,
              margin: cs.margin,
              padding: cs.padding,
            };
          }
          window.logLayout = function (label) {
            try {
              const pc = document.getElementById("player-container");
              const rr = pc ? pc.querySelector(".rr-player") : null;
              const wrapper = pc ? pc.querySelector(".replayer-wrapper") : null;
              const iframe = pc ? pc.querySelector("iframe") : null;
              const vp = {
                width: window.innerWidth,
                height: window.innerHeight,
              };
              const center = { x: vp.width / 2, y: vp.height / 2 };
              const rrRect = rr ? rr.getBoundingClientRect() : null;
              const rrCenter = rrRect
                ? {
                    x: rrRect.left + rrRect.width / 2,
                    y: rrRect.top + rrRect.height / 2,
                  }
                : null;
              const deltas = rrCenter
                ? {
                    dx: Math.round(rrCenter.x - center.x),
                    dy: Math.round(rrCenter.y - center.y),
                  }
                : null;
              console.log(
                "[LAYOUT]",
                JSON.stringify({
                  label,
                  viewport: vp,
                  center,
                  deltas,
                  elements: {
                    playerContainer: {
                      rect: __dumpRect(pc),
                      style: __dumpStyle(pc),
                    },
                    rrPlayer: { rect: __dumpRect(rr), style: __dumpStyle(rr) },
                    replayerWrapper: {
                      rect: __dumpRect(wrapper),
                      style: __dumpStyle(wrapper),
                    },
                    iframe: {
                      rect: __dumpRect(iframe),
                      style: __dumpStyle(iframe),
                    },
                  },
                })
              );
            } catch (e) {
              console.log("[LAYOUT] error", e);
            }
          };

          // Wait a bit for the async operations to complete and check for replayer
          let retries = 0;
          while (!window.replayer && retries < 30) {
            await new Promise((resolve) => setTimeout(resolve, 200));
            retries++;
            console.log(
              "[PUPPETEER CONTEXT] Waiting for replayer... attempt " +
                retries +
                "/30"
            );

            // Additional debugging every 5 attempts
            if (retries % 5 === 0) {
              console.log(
                "[PUPPETEER CONTEXT] Current window properties:",
                Object.keys(window).filter(
                  (k) => k.includes("replay") || k.includes("rrweb")
                )
              );
              console.log(
                "[PUPPETEER CONTEXT] Player element exists:",
                !!document.getElementById("player")
              );
              console.log(
                "[PUPPETEER CONTEXT] rrwebPlayer function type:",
                typeof rrwebPlayer
              );
            }
          }

          if (!window.replayer) {
            console.error(
              "[PUPPETEER CONTEXT] rrweb player instance not found after setup (waited " +
                retries * 200 +
                "ms)."
            );
            // Enhanced debugging information
            console.error(
              "[PUPPETEER CONTEXT] Available on window:",
              Object.keys(window)
                .filter((k) => k.includes("replay") || k.includes("rrweb"))
                .join(", ")
            );
            console.error(
              "[PUPPETEER CONTEXT] typeof rrwebPlayer:",
              typeof rrwebPlayer
            );
            console.error(
              "[PUPPETEER CONTEXT] Player element innerHTML:",
              document.getElementById("player")?.innerHTML?.substring(0, 200)
            );
            console.error(
              "[PUPPETEER CONTEXT] DOM ready state:",
              document.readyState
            );
            return {
              error: "replayer not available after setup",
              debugInfo: {
                rrwebPlayerType: typeof rrwebPlayer,
                playerElementExists: !!document.getElementById("player"),
                domReadyState: document.readyState,
                windowProps: Object.keys(window).filter(
                  (k) => k.includes("replay") || k.includes("rrweb")
                ),
              },
            };
          }

          // Setup image preloading before overlay cycle starts
          if (window.setupImagePreloading) {
            console.log("[PUPPETEER CONTEXT] Setting up image preloading...");
            try {
              await window.setupImagePreloading();
              console.log(
                "[PUPPETEER CONTEXT] Image preloading setup completed"
              );
            } catch (e) {
              console.warn(
                "[PUPPETEER CONTEXT] Image preloading failed:",
                e.message
              );
              // Continue with replay even if image preloading fails
            }
          } else {
            console.warn(
              "[PUPPETEER CONTEXT] window.setupImagePreloading not available"
            );
          }

          // Fallback: define a minimal overlay cycle if not provided by the page
          if (typeof window.runOverlayCycle !== "function") {
            console.log(
              "[PUPPETEER CONTEXT] Defining fallback runOverlayCycle"
            );
            window.runOverlayCycle = function (instructions, firstTs) {
              try {
                const overlayDiv = document.getElementById(
                  "externalSkipOverlay"
                );
                const overlayText = document.getElementById(
                  "externalSkipOverlayText"
                );
                if (!overlayDiv || !overlayText) {
                  console.warn(
                    "[PUPPETEER CONTEXT] Overlay elements missing; fallback overlay disabled"
                  );
                  return;
                }
                if (!Array.isArray(instructions)) instructions = [];
                let currentInstruction = null;
                let instructionIdx = 0;

                function nowReplayTime() {
                  try {
                    const r = window.coreReplayer || window.replayer;
                    const t =
                      r && typeof r.getCurrentTime === "function"
                        ? r.getCurrentTime()
                        : 0;
                    const looksAbs = t > firstTs && t - firstTs < 600000;
                    const abs = looksAbs ? t : (firstTs || 0) + t;
                    const rel = abs - (firstTs || 0);
                    return { raw: t, abs, rel, looksAbs };
                  } catch (e) {
                    return { raw: null, abs: null, rel: null, looksAbs: null };
                  }
                }

                function checkStatus() {
                  if (window.replayStarted) {
                    const ts = nowReplayTime();
                    if (
                      currentInstruction &&
                      ts.abs != null &&
                      ts.abs >= currentInstruction.hideAt - 1
                    ) {
                      overlayDiv.style.display = "none";
                      overlayText.textContent = "";
                      try {
                        console.log(
                          `[OVERLAY][HIDE] raw=${ts.raw} abs=${ts.abs} rel=${
                            ts.rel
                          } targetHide=${currentInstruction.hideAt} diff=${
                            ts.abs - currentInstruction.hideAt
                          }`
                        );
                      } catch (_) {}
                      currentInstruction = null;
                    }
                    if (
                      !currentInstruction &&
                      instructionIdx < instructions.length &&
                      ts.abs != null &&
                      ts.abs >= instructions[instructionIdx].showAt
                    ) {
                      currentInstruction = instructions[instructionIdx];
                      overlayText.textContent =
                        currentInstruction.message || "";
                      overlayDiv.style.display = "flex";
                      try {
                        console.log(
                          `[OVERLAY][SHOW] raw=${ts.raw} abs=${ts.abs} rel=${
                            ts.rel
                          } targetShow=${currentInstruction.showAt} diff=${
                            ts.abs - currentInstruction.showAt
                          }`
                        );
                      } catch (_) {}
                      instructionIdx++;
                    }
                  }
                  requestAnimationFrame(checkStatus);
                }
                requestAnimationFrame(checkStatus);
              } catch (e) {
                console.error(
                  "[PUPPETEER CONTEXT] Fallback runOverlayCycle error:",
                  e && e.message ? e.message : e
                );
              }
            };
          }

          if (window.runOverlayCycle) {
            window.runOverlayCycle(instructions, firstTs); // Sets up the overlay check loop
          } else {
            console.error(
              "[PUPPETEER CONTEXT] window.runOverlayCycle not defined."
            );
            return {
              error: "window.runOverlayCycle not defined",
            };
          }

          console.log("[PUPPETEER CONTEXT] Setup completed successfully");
          return {
            success: true,
          };
        } catch (e) {
          console.error(
            "[PUPPETEER CONTEXT] Error during setup:",
            e.message,
            e.stack
          );
          return {
            error: e.message,
            stack: e.stack,
          };
        }
      },
      overlayInstructions,
      firstProcessedEventTimestamp,
      clickEventsForFreeze
    );

    if (setupResult && setupResult.error) {
      console.error(
        `[DIAGNOSTIC] Error from page.evaluate during setup: ${setupResult.error}`,
        setupResult.stack || ""
      );

      if (setupResult.debugInfo) {
        console.error(
          `[DIAGNOSTIC] Additional debug information:`,
          setupResult.debugInfo
        );
      }

      // Enhanced error context for clips_folder vs main videos
      const isClip = !overlayInstructions || overlayInstructions.length === 0;
      console.error(
        `[DIAGNOSTIC] Recording type: ${isClip ? "CLIP" : "MAIN VIDEO"}`
      );
      console.error(`[DIAGNOSTIC] Events count: ${eventsToReplay.length}`);
      console.error(`[DIAGNOSTIC] Output path: ${outputVideoPath}`);
      console.error(`[DIAGNOSTIC] Session ID: ${sessionId}`);

      // Log events summary for debugging
      const eventTypeCounts = eventsToReplay.reduce((acc, event) => {
        acc[event.type] = (acc[event.type] || 0) + 1;
        return acc;
      }, {});
      console.error(`[DIAGNOSTIC] Event type distribution:`, eventTypeCounts);

      throw new Error(`Failed to set up rrweb player: ${setupResult.error}`);
    }
    console.log("rrweb player setup complete.");

    await recorder.start(outputVideoPath); // Start recorder
    const recordingStartTime = Date.now(); // Record when clip recording actually starts
    console.log(
      "Screen recording started. Output will be saved to:",
      outputVideoPath
    );

    // Mark precise recorder start time in the page for freeze interval calculations
    try {
      await page.evaluate(() => {
        // Anchor all freeze timings to when the recorder actually started
        window.videoRecordingStartedWallTime = performance.now();
        try {
          console.log(
            `[RECORDER] videoRecordingStartedWallTime=${window.videoRecordingStartedWallTime.toFixed(
              1
            )}`
          );
        } catch (_) {}
      });
    } catch (_) {}

    console.log("Starting rrweb replay...");
    const playResult = await page.evaluate((recordingStartTime) => {
      try {
        if (window.replayer && typeof window.replayer.play === "function") {
          const replayStartTime = Date.now(); // Record when replay actually starts
          window.replayer.play();
          window.replayStarted = true; // Signal that replay has started
          const timingOffset = replayStartTime - recordingStartTime; // Calculate offset
          console.log(
            "[PUPPETEER CONTEXT] Called replayer.play() and set replayStarted=true."
          );
          console.log(
            "[TIMING] Recording started at: " +
              recordingStartTime +
              ", Replay started at: " +
              replayStartTime +
              ", Offset: " +
              timingOffset +
              "ms"
          );
          return {
            success: true,
            timingOffset: timingOffset, // Return the measured offset
          };
        } else {
          console.error(
            "[PUPPETEER CONTEXT] window.replayer.play is not a function or replayer is missing."
          );
          return {
            error: "replayer.play not available",
          };
        }
      } catch (e) {
        console.error(
          "[PUPPETEER CONTEXT] Error starting replay:",
          e.message,
          e.stack
        );
        return {
          error: e.message,
          stack: e.stack,
        };
      }
    }, recordingStartTime);

    if (playResult && playResult.error) {
      console.error(
        `[DIAGNOSTIC] Error from page.evaluate during play: ${playResult.error}`,
        playResult.stack || ""
      );
      // Stop recording and clean up if play fails.
      await recorder.stop();
      throw new Error(`Failed to start rrweb replay: ${playResult.error}`);
    }

    // Extract the timing offset from the play result
    timingOffsetMs = playResult.timingOffset || 0;
    console.log(`[TIMING] Measured clip recording delay: ${timingOffsetMs}ms`);

    console.log("Replay started.");

    // DEBUG: Periodic playback heartbeat from within the page
    try {
      await page.evaluate(
        (firstTs, lastTs) => {
          try {
            const r = window.coreReplayer || window.replayer;
            window.__MAIN_DEBUG = window.__MAIN_DEBUG || {};
            window.__MAIN_DEBUG.firstTs = firstTs;
            window.__MAIN_DEBUG.lastTs = lastTs;
            window.__MAIN_DEBUG.durationMs = lastTs - firstTs;
            if (window.__mainDebugInterval) {
              clearInterval(window.__mainDebugInterval);
            }
            let loggedEnd = false;
            window.__mainDebugInterval = setInterval(() => {
              try {
                const raw =
                  r && typeof r.getCurrentTime === "function"
                    ? r.getCurrentTime()
                    : null;
                const looksAbs =
                  raw != null && raw > firstTs && raw - firstTs < 600000;
                const abs = looksAbs ? raw : raw != null ? firstTs + raw : null;
                const rel = abs != null ? abs - firstTs : null;
                const duration = window.__MAIN_DEBUG.durationMs;
                const pct =
                  rel != null && duration > 0
                    ? ((rel / duration) * 100).toFixed(1)
                    : null;
                const sp = r?.config?.speed;
                const svc =
                  (r?.service &&
                    (r.service.status || r.service.state?.value)) ||
                  null;
                console.log(
                  `[MAIN-TICK] raw=${raw} abs=${abs} rel=${rel}/${duration}ms pct=${pct}% speed=${sp} state=${svc}`
                );
                if (
                  !loggedEnd &&
                  rel != null &&
                  duration != null &&
                  rel >= duration - 50
                ) {
                  loggedEnd = true;
                  console.log(
                    `[MAIN-TICK] Reached end threshold rel=${rel}ms duration=${duration}ms`
                  );
                }
              } catch (e) {
                console.log("[MAIN-TICK] error", (e && e.message) || String(e));
              }
            }, 1000);
          } catch (_) {}
        },
        firstProcessedEventTimestamp,
        lastProcessedEventTimestamp
      );
    } catch (_) {}

    // Start main-thread freeze monitor for the full session video
    try {
      await page.evaluate(() => {
        if (window.__freezeMonitorStarted) return;
        window.__freezeMonitorStarted = true;
        console.log(
          "[DIAGNOSTIC] Starting main thread responsiveness monitor (MAIN VIDEO)..."
        );
        window.mainThreadFreezes = Array.isArray(window.mainThreadFreezes)
          ? window.mainThreadFreezes
          : [];
        let lastFrameTime = performance.now();
        function checkResponsiveness() {
          const now = performance.now();
          const delta = now - lastFrameTime;
          if (delta > 200 && window.videoRecordingStartedWallTime) {
            const freezeStartInVideo =
              now - delta - window.videoRecordingStartedWallTime;
            const freezeEndInVideo = now - window.videoRecordingStartedWallTime;
            if (freezeStartInVideo >= 0) {
              window.mainThreadFreezes.push({
                start: freezeStartInVideo / 1000,
                end: freezeEndInVideo / 1000,
              });
              try {
                console.log(
                  `[DIAGNOSTIC][MAIN] Freeze detected: ${delta.toFixed(
                    0
                  )}ms at ${freezeStartInVideo.toFixed(0)}ms`
                );
              } catch (_) {}
            }

            if (window.coreReplayer && window.coreReplayer.speedService) {
              console.log(
                "[DIAGNOSTIC] Main thread stall detected. Pausing and resuming player to reset timers."
              );
              window.coreReplayer.speedService.send({
                type: "SET_SPEED",
                payload: { speed: 0 },
              });
              setTimeout(() => {
                window.coreReplayer.speedService.send({
                  type: "SET_SPEED",
                  payload: { speed: 1 },
                });
                console.log("[DIAGNOSTIC] Player resumed at normal speed.");
              }, 50);
            }
          }
          lastFrameTime = now;
          requestAnimationFrame(checkResponsiveness);
        }
        requestAnimationFrame(checkResponsiveness);
      });
    } catch (_) {}

    // The baseline time to wait is the duration of the replay PLUS the initial delay before the replay started.
    // We then extend if playback hasn't actually reached the end (to account for stalls).
    const totalWaitTimeMs =
      recordingDurationMs + timingOffsetMs + totalFreezeTimeMs;
    console.log(
      `Replay and overlay test running. Waiting for ${Math.round(
        totalWaitTimeMs / 1000
      )}s (recording duration + timing offset + freeze time)...`
    );
    await new Promise((resolve) => setTimeout(resolve, totalWaitTimeMs));

    // After baseline wait, verify progress. If not at end, extend wait until end threshold.
    try {
      const progressAfterBaseline = await page.evaluate((firstTs) => {
        try {
          const r = window.coreReplayer || window.replayer;
          const t =
            r && typeof r.getCurrentTime === "function"
              ? r.getCurrentTime()
              : 0;
          const looksAbs = t > firstTs && t - firstTs < 600000;
          const abs = looksAbs ? t : firstTs + t;
          const rel = abs - firstTs;
          const duration =
            (window.__MAIN_DEBUG && window.__MAIN_DEBUG.durationMs) || 0;
          return { rel, duration };
        } catch (e) {
          return {
            rel: null,
            duration: null,
            error: (e && e.message) || String(e),
          };
        }
      }, firstProcessedEventTimestamp);
      console.log(
        `[MAIN-DEBUG] Progress after baseline wait rel=${progressAfterBaseline?.rel} / ${progressAfterBaseline?.duration}ms`
      );
      const needMs =
        progressAfterBaseline &&
        typeof progressAfterBaseline.rel === "number" &&
        typeof progressAfterBaseline.duration === "number"
          ? Math.max(
              0,
              progressAfterBaseline.duration - progressAfterBaseline.rel
            )
          : 0;
      if (needMs > 50) {
        console.log(
          `[MAIN-DEBUG] Replay not finished after baseline wait. Extending until end threshold (need ~${Math.round(
            needMs
          )}ms).`
        );
        // Prefer an in-page completion condition to avoid over/under waiting.
        try {
          await page.waitForFunction(
            (firstTs) => {
              try {
                const r = window.coreReplayer || window.replayer;
                if (!r || typeof r.getCurrentTime !== "function") return false;
                const t = r.getCurrentTime();
                const looksAbs = t > firstTs && t - firstTs < 600000;
                const abs = looksAbs ? t : firstTs + t;
                const rel = abs - firstTs;
                const duration =
                  (window.__MAIN_DEBUG && window.__MAIN_DEBUG.durationMs) || 0;
                return duration > 0 && rel >= duration - 33; // within ~1-2 frames at 30-60fps
              } catch (_) {
                return false;
              }
            },
            { timeout: Math.min(20000, needMs + 5000) },
            firstProcessedEventTimestamp
          );
          console.log(
            `[MAIN-DEBUG] End threshold reached. Proceeding to stop recorder.`
          );
        } catch (e) {
          console.warn(
            `[MAIN-DEBUG] waitForFunction end-threshold timed out: ${
              (e && e.message) || e
            }. Proceeding to stop.`
          );
        }
      }
    } catch (e) {
      console.warn(
        `[MAIN-DEBUG] Post-wait progress check failed: ${(e && e.message) || e}`
      );
    }

    // One last probe before stopping
    try {
      const finalProbe = await page.evaluate((firstTs) => {
        try {
          const r = window.coreReplayer || window.replayer;
          const t =
            r && typeof r.getCurrentTime === "function"
              ? r.getCurrentTime()
              : 0;
          const looksAbs = t > firstTs && t - firstTs < 600000;
          const abs = looksAbs ? t : firstTs + t;
          const rel = abs - firstTs;
          const duration =
            (window.__MAIN_DEBUG && window.__MAIN_DEBUG.durationMs) || 0;
          return { rel, duration };
        } catch (e2) {
          return {
            rel: null,
            duration: null,
            error: (e2 && e2.message) || String(e2),
          };
        }
      }, firstProcessedEventTimestamp);
      console.log(
        `[MAIN-DEBUG] Final probe before stop rel=${finalProbe?.rel} / ${finalProbe?.duration}ms`
      );
    } catch (_) {}

    await recorder.stop();
    console.log("Screen recording stopped.");

    const sessionVerifications = await page.evaluate(() => {
      try {
        const arr = Array.isArray(window.sessionClickVerifications)
          ? window.sessionClickVerifications
          : [];
        console.log(
          "[VERIFY][MAIN] Collected sessionClickVerifications length:",
          arr.length
        );
        return arr;
      } catch (_) {
        return [];
      }
    });
    if (sessionVerifications.length > 0) {
      console.log(
        `\n==== SESSION-WIDE CLICK VERIFICATION RESULTS (${sessionVerifications.length} clicks) ====`
      );
      sessionVerifications.forEach((result, index) => {
        console.log(
          `[Click ${index + 1}] Claimed Target ID: ${
            result.claimedTargetId
          }, Visual Target ID: ${result.visualId}, Match: ${result.match} (${
            result.matchType
          })`
        );
        if (!result.match) {
          console.log(`   - Mismatch Details:`, result);
        }
      });

      const clipsDir = path.join(
        process.cwd(),
        "clips_folder",
        `session_${sessionId}`
      );
      await fs.mkdir(clipsDir, { recursive: true });

      for (let i = 0; i < sessionVerifications.length; i++) {
        const verification = sessionVerifications[i];
        const verificationFileName = `click_${i}_verification.json`;
        const verificationFilePath = path.join(clipsDir, verificationFileName);
        try {
          await fs.writeFile(
            verificationFilePath,
            JSON.stringify(verification, null, 2)
          );
          console.log(
            `✅ Verification result for click ${i} saved to: ${verificationFilePath}`
          );
        } catch (e) {
          console.error(
            `❌ Failed to save session verification for click ${i}: ${e.message}`
          );
        }
      }
    }

    // Trim the initial delay from the clip to synchronize it with the events.
    if (timingOffsetMs > 100) {
      // Only trim if delay is significant
      const {
        dir: outDir,
        name: outName,
        ext: outExt,
      } = path.parse(outputVideoPath);
      const untrimmedPath = path.join(
        outDir,
        `${outName}.untrimmed${outExt || ".mp4"}`
      );

      try {
        console.log(
          `Attempting to trim initial ${timingOffsetMs}ms from video.`
        );
        await fs.rename(outputVideoPath, untrimmedPath);
        const trimSuccess = await trimVideo(
          untrimmedPath,
          outputVideoPath,
          timingOffsetMs / 1000
        );
        if (trimSuccess) {
          console.log("Video trimmed successfully. Deleting untrimmed source.");
          await fs.unlink(untrimmedPath);
        } else {
          console.warn(
            "Failed to trim video. Restoring original, which will have an initial delay."
          );
          await fs.rename(untrimmedPath, outputVideoPath);
        }
      } catch (trimError) {
        console.error(
          "An error occurred during the clip trimming process:",
          trimError
        );
        // Try to restore if the untrimmed file still exists
        try {
          await fs.access(untrimmedPath);
          await fs.rename(untrimmedPath, outputVideoPath);
        } catch (restoreError) {
          console.error("Failed to restore untrimmed video.", restoreError);
        }
      }
    }
    // Remove detected main-thread freezes from the MAIN clip before adding audio
    try {
      const freezeIntervalsRaw = await page.evaluate(
        () => window.mainThreadFreezes || []
      );
      if (freezeIntervalsRaw && freezeIntervalsRaw.length > 0) {
        console.log(
          `[POST-PROCESS][MAIN] Found ${freezeIntervalsRaw.length} main thread freeze interval(s). Preparing removal...`
        );

        // Adjust intervals if we trimmed an initial timing offset earlier
        const trimSeconds = timingOffsetMs > 100 ? timingOffsetMs / 1000 : 0;
        const adjustedIntervals = freezeIntervalsRaw
          .map((iv) => ({
            start: Math.max(0, Number(iv.start) - trimSeconds),
            end: Math.max(0, Number(iv.end) - trimSeconds),
          }))
          .filter(
            (iv) =>
              Number.isFinite(iv.start) &&
              Number.isFinite(iv.end) &&
              iv.end > iv.start
          );

        if (adjustedIntervals.length > 0) {
          const {
            dir: outDir2,
            name: outName2,
            ext: outExt2,
          } = path.parse(outputVideoPath);
          const tempFreezeInputPath = path.join(
            outDir2,
            `${outName2}.with_freezes${outExt2 || ".mp4"}`
          );

          // Preserve a raw copy before removal
          const rawCopyPath = path.join(
            outDir2,
            `${outName2}_raw${outExt2 || ".mp4"}`
          );
          try {
            await fs.rename(outputVideoPath, tempFreezeInputPath);
            await fs.copyFile(tempFreezeInputPath, rawCopyPath);
            console.log(
              `[POST-PROCESS][MAIN] Saved raw copy to: ${rawCopyPath}`
            );
          } catch (copyErr) {
            console.warn(
              "[POST-PROCESS][MAIN] Couldn't prepare raw copy for freeze removal:",
              copyErr
            );
          }

          const removalSuccess = await removeSegmentsFromVideo(
            tempFreezeInputPath,
            outputVideoPath,
            adjustedIntervals
          );

          if (removalSuccess) {
            console.log(
              "[POST-PROCESS][MAIN] Freeze intervals removed successfully."
            );
            try {
              await fs.unlink(tempFreezeInputPath);
            } catch (_) {}
          } else {
            console.warn(
              "[POST-PROCESS][MAIN] Freeze removal failed. Restoring original video."
            );
            try {
              await fs.rename(tempFreezeInputPath, outputVideoPath);
            } catch (_) {}
          }
        } else {
          console.log(
            "[POST-PROCESS][MAIN] No valid freeze intervals after adjustment. Skipping removal."
          );
        }
      } else {
        console.log(
          "[POST-PROCESS][MAIN] No main-thread freezes detected. Skipping removal."
        );
      }
    } catch (freezeErr) {
      console.warn(
        "[POST-PROCESS][MAIN] Error during freeze removal step (skipped):",
        freezeErr
      );
    }

    // Add silent audio track to the video
    // Ensure the temporary file name is created by inserting `.temp` before the file extension
    const {
      dir: outDir,
      name: outName,
      ext: outExt,
    } = path.parse(outputVideoPath);
    const tempVideoPath = path.join(
      outDir,
      `${outName}.temp${outExt || ".mp4"}`
    );

    // Rename the original clip to temp
    try {
      await fs.rename(outputVideoPath, tempVideoPath);
      console.log(`Renamed clip to temporary file: ${tempVideoPath}`);

      // Add silent audio track
      const audioAddedSuccessfully = await addSilentAudioTrack(
        tempVideoPath,
        outputVideoPath
      );

      if (!audioAddedSuccessfully) {
        // If adding audio failed, rename the temp file back
        console.warn(
          "Failed to add audio track. Using original clip without audio."
        );
        await fs.rename(tempVideoPath, outputVideoPath);
      } else {
        // Clean up the temporary file
        try {
          await fs.unlink(tempVideoPath);
          console.log(`Temporary clip file ${tempVideoPath} deleted.`);
        } catch (unlinkError) {
          console.warn(`Could not delete temporary clip file:`, unlinkError);
        }
      }
    } catch (renameError) {
      console.error("Error processing clip for audio track:", renameError);
      // Continue with the original clip if rename failed
    }

    // Get the final clip file size
    try {
      const videoStats = await fs.stat(outputVideoPath);
      videoSizeBytes = videoStats.size;
      videoSizeMB = (videoSizeBytes / (1024 * 1024)).toFixed(2);
      console.log(
        `Final clip file size: ${videoSizeBytes} bytes (${videoSizeMB} MB)`
      );
      // DEBUG: Probe final media duration to compare with expected timeline
      try {
        const ffprobeCommand = ffprobeStatic?.path || "ffprobe";
        const pr = spawnSync(
          ffprobeCommand,
          [
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            outputVideoPath,
          ],
          { encoding: "utf-8" }
        );
        if (pr.status === 0 && pr.stdout) {
          const seconds = parseFloat(pr.stdout.trim());
          if (Number.isFinite(seconds)) {
            const finalDurationMs = Math.round(seconds * 1000);
            console.log(
              `[MAIN-DEBUG][FFPROBE] Final clip durationMs=${finalDurationMs}, expectedRecordingMs=${recordingDurationMs}, diffMs=${
                finalDurationMs - recordingDurationMs
              }`
            );
          }
        } else {
          console.warn(
            `[MAIN-DEBUG][FFPROBE] Failed to probe final clip duration. stderr=${pr.stderr}`
          );
        }
      } catch (e) {
        console.warn(
          `[MAIN-DEBUG][FFPROBE] Exception probing final duration: ${e.message}`
        );
      }
    } catch (statError) {
      console.error("Error getting clip file size:", statError);
    }

    // Embed hidden beacon subtitles (mov_text) for yellow markers
    try {
      const subResult = await embedBeaconSubtitlesIntoVideo(
        outputVideoPath,
        sessionId
      );
      if (subResult?.added) {
        console.log(
          `[SUB-MUX] Embedded ${subResult.count} yellow beacon subtitles (hidden).`
        );
      } else {
        console.log(
          `[SUB-MUX] No beacon subtitles embedded (${
            subResult?.reason || subResult?.error || "unknown"
          }).`
        );
      }
    } catch (e) {
      console.warn(`[SUB-MUX] Skipped embedding subtitles: ${e?.message || e}`);
    }

    // Optionally cut the master clip into segments based on beacon flashes
    try {
      const shouldCutByBeacons =
        String(
          process.env.CUT_VIDEO_BASED_ON_BEACONS || "true"
        ).toLowerCase() === "true";
      if (shouldCutByBeacons && process.env.ONLY_ANALYSIS !== "true") {
        const sessionDir = path.join(
          process.cwd(),
          "clips_folder",
          `session_${sessionId}`
        );
        console.log(
          `[BEACON-CUT] Starting beacon-based cutting into: ${sessionDir}`
        );
        const cutResult = await cutVideoBasedOnBeacons(
          outputVideoPath,
          sessionDir,
 {
            includeClick: true,
            includeNoClick: true,
            removeBeaconFrames: true,
            addAudio: true,
            sessionId: sessionId,
          }

         /*  {
            includeClick: true,
            includeNoClick: NOCLICK_FROM_GAPS
              ? false
              : INCLUDE_NOCLICK_SEGMENTS,
            preserveGapDuration: NOCLICK_FROM_GAPS,
            removeBeaconFrames: true,
            addAudio: true,
            sessionId: sessionId,
          } */
        );
        try {
          const indexPath = path.join(
            sessionDir,
            "beacon_segments",
            "segments_index.json"
          );
          await fs.mkdir(path.dirname(indexPath), { recursive: true });
          await fs.writeFile(
            indexPath,
            JSON.stringify({ segments: cutResult.segments }, null, 2)
          );
          console.log(`[BEACON-CUT] Wrote segments index: ${indexPath}`);
        } catch (_) {}
      } else {
        console.log(
          "[BEACON-CUT] Skipped (CUT_VIDEO_BASED_ON_BEACONS != true)"
        );
      }
    } catch (e) {
      console.warn(
        "[BEACON-CUT] Cutting step failed (non-fatal):",
        e?.message || e
      );
    }

    // ---- GCS UPLOAD START ----
    // This is the new section that uploads the result.

    // this section uploads the entire video to GCS
    if (process.env.NODE_ENV === "production") {
      // Define a unique name for the file in the GCS bucket.
      const destinationFileName = `clips_folder/${sessionId}.mp4`;

      // Chelper to upload the file and get the public URL.
      const publicUrl = await uploadToGCS(outputVideoPath, destinationFileName);

      console.log("Successfully generated and uploaded recording.");
      // ---- GCS UPLOAD END ----
    }

    try {
      await fs.unlink(tempFilePath);
      console.log(`Temporary player HTML file ${tempFilePath} deleted.`);
      await fs.unlink(tempEventsPath);
      console.log(`Temporary events JSON file ${tempEventsPath} deleted.`);
    } catch (unlinkError) {
      console.warn(`Could not delete temporary files:`, unlinkError);
    }
  } catch (error) {
    return { success: false, videoSizeBytes: 0, videoSizeMB: 0, fps: null }; // Indicate failure
    console.error("An error occurred during recordRrwebEvents:", error);
  } finally {
    if (browser) {
      console.log("Closing browser...");
      await browser.close();
      console.log("Browser closed.");
    }
  }
  console.log(`Script finished. Video should be at ${outputVideoPath}`);
  return {
    success: true,
    videoSizeBytes,
    videoSizeMB,
    fps: finalFps,
    timingOffsetMs,
  }; // Return success with file size info and timing offset
}

// const pako = require('pako'); // If using CommonJS with require

// rrweb event type constants
const EventType = {
  DomContentLoaded: 0,
  Load: 1,
  FullSnapshot: 2,
  IncrementalSnapshot: 3,
  Meta: 4,
  Custom: 5,
  Plugin: 6,
};

// rrweb incremental source constants
const IncrementalSource = {
  Mutation: 0,
  MouseMove: 1,
  MouseInteraction: 2,
  Scroll: 3,
  ViewportResize: 4,
  Input: 5,
  StyleSheetRule: 6,
  MediaInteraction: 7,
  Font: 8,
  Log: 9,
  Drag: 10,
  StyleDeclaration: 11,
};

/**
 * Helper function to fetch an image and convert it to a data URI.
 * @param {string} url The URL of the image to fetch.
 * @returns {Promise<string|null>} A data URI string or null if fetching fails.
 */
async function fetchImageAsDataURI(url) {
  if (!url || url.startsWith("data:") || !url.startsWith("http")) {
    return null;
  }
  try {
    // Normalize Next.js optimized image URLs to the original source when possible
    const normalizeNextImageUrl = (inputUrl) => {
      try {
        const parsed = new URL(inputUrl);
        if (
          parsed.pathname === "/_next/image" &&
          parsed.searchParams.has("url")
        ) {
          let inner = parsed.searchParams.get("url");
          try {
            inner = decodeURIComponent(inner);
          } catch (_) {}
          if (inner && inner.startsWith("/")) {
            return parsed.origin + inner;
          }
          if (inner && inner.startsWith("http")) {
            return inner;
          }
        }
      } catch (_) {}
      return inputUrl;
    };

    const finalUrl = normalizeNextImageUrl(url);

    const response = await axios.get(finalUrl, {
      responseType: "arraybuffer",
      timeout: 8000,
      maxRedirects: 2,
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        // Some CDNs require a Referer; use the origin of the URL if parsable
        Referer: (() => {
          try {
            return new URL(finalUrl).origin;
          } catch (_) {
            return undefined;
          }
        })(),
      },
    });
    const mimeType = response.headers["content-type"] || "image/png";
    if (!mimeType.startsWith("image/")) {
      console.warn(
        `[Image Pre-fetch] Skipped non-image content at ${url} (MIME: ${mimeType})`
      );
      return null;
    }
    const base64 = Buffer.from(response.data, "binary").toString("base64");
    return `data:${mimeType};base64,${base64}`;
  } catch (error) {
    console.warn(
      `[Image Pre-fetch] Failed to fetch image ${url}: ${error.message}`
    );
    return null;
  }
}
/**
 * Finds all image URLs in rrweb events, pre-fetches them,
 * and rewrites the events to use self-contained data URIs.
 * @param {Array} events The original array of rrweb events.
 * @returns {Promise<Array>} A new array of rrweb events with images embedded as data URIs.
 */
async function prefetchAllImagesAndRewriteEvents(events) {
  try {
    console.log("🖼️  Starting image pre-fetching and event rewriting...");
    const imageUrls = new Set();

    const findImageUrlsInNode = (node) => {
      if (!node) return;
      if (
        (node.tagName === "img" || node.tagName === "source") &&
        node.attributes
      ) {
        if (node.attributes.src) imageUrls.add(node.attributes.src);
        if (node.attributes.srcset) {
          node.attributes.srcset.split(",").forEach((part) => {
            const url = part.trim().split(" ")[0];
            if (url) imageUrls.add(url);
          });
        }
      }
      // Inline style background images
      if (node.attributes && typeof node.attributes.style === "string") {
        const style = node.attributes.style;
        const urlRegex = /url\((['"]?)(.*?)\1\)/g;
        let match;
        while ((match = urlRegex.exec(style)) !== null) {
          const candidate = match[2];
          if (candidate && candidate.startsWith("http")) {
            imageUrls.add(candidate);
          }
        }
      }
      if (node.childNodes) {
        node.childNodes.forEach(findImageUrlsInNode);
      }
    };

    for (const event of events) {
      if (event.type === EventType.FullSnapshot && event.data?.node) {
        findImageUrlsInNode(event.data.node);
      } else if (
        event.type === EventType.IncrementalSnapshot &&
        event.data?.source === IncrementalSource.Mutation
      ) {
        if (event.data.adds) {
          event.data.adds.forEach((add) => findImageUrlsInNode(add.node));
        }
        if (event.data.attributes) {
          event.data.attributes.forEach((attr) => {
            if (attr.attributes.src) imageUrls.add(attr.attributes.src);
            if (attr.attributes.srcset) {
              attr.attributes.srcset.split(",").forEach((part) => {
                const url = part.trim().split(" ")[0];
                if (url) imageUrls.add(url);
              });
            }
          });
        }
      }
    }

    const uniqueUrls = Array.from(imageUrls).filter(
      (url) => url && !url.startsWith("data:") && url.startsWith("http")
    );
    if (uniqueUrls.length === 0) {
      console.log("🖼️  No external image URLs found to pre-fetch.");
      return events;
    }

    console.log(
      `🖼️  Found ${uniqueUrls.length} unique image URLs to pre-fetch.`
    );

    const urlToDataURIMap = new Map();
    // Fetch with concurrency cap to avoid hammering hosts
    const concurrency = 8;
    let index = 0;
    async function worker() {
      while (index < uniqueUrls.length) {
        const current = uniqueUrls[index++];

        if (process.env.PROCESS_IMAGES !== "false") {
          const dataURI = await fetchImageAsDataURI(current);
          if (dataURI) urlToDataURIMap.set(current, dataURI);
        }
      }
    }
    const workers = Array.from(
      { length: Math.min(concurrency, uniqueUrls.length) },
      worker
    );
    await Promise.all(workers);

    if (urlToDataURIMap.size === 0) {
      console.log("🖼️  Failed to fetch any images, returning original events.");
      return events;
    }
    console.log(
      `🖼️  Successfully pre-fetched and converted ${urlToDataURIMap.size} images to data URIs.`
    );

    // As an added safety, ensure images still render even if some URLs failed:
    // For <img> with missing/blocked src, set a tiny transparent placeholder if nothing loaded during playback.
    const ONE_BY_ONE_PNG =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==";

    const newEvents = JSON.parse(JSON.stringify(events));

    const rewriteUrlsInNode = (node) => {
      if (!node) return;
      if (node.tagName === "img" && node.attributes) {
        // Force eager load/paint for faster clip rendering
        if (node.attributes.loading) delete node.attributes.loading;
        if (node.attributes.decoding) delete node.attributes.decoding;
        node.attributes.fetchpriority = "high";
        if (node.attributes.src && urlToDataURIMap.has(node.attributes.src)) {
          node.attributes.src = urlToDataURIMap.get(node.attributes.src);
        }
        if (node.attributes.srcset) {
          const newSrcset = node.attributes.srcset
            .split(",")
            .map((part) => {
              const url = part.trim().split(" ")[0];
              if (urlToDataURIMap.has(url)) {
                return part.replace(url, urlToDataURIMap.get(url));
              }
              return part;
            })
            .join(", ");
          node.attributes.srcset = newSrcset;
        }
        // If no src and no srcset mapped, keep original. Player may still load it.
        // Optionally, one could fallback to a tiny PNG placeholder if desired.
      }
      if (node.tagName === "source" && node.attributes) {
        if (node.attributes.srcset) {
          const newSrcset = node.attributes.srcset
            .split(",")
            .map((part) => {
              const url = part.trim().split(" ")[0];
              if (urlToDataURIMap.has(url)) {
                return part.replace(url, urlToDataURIMap.get(url));
              }
              return part;
            })
            .join(", ");
          node.attributes.srcset = newSrcset;
        }
      }
      // Inline style background images
      if (node.attributes && typeof node.attributes.style === "string") {
        const urlRegex = /url\((['"]?)(.*?)\1\)/g;
        let replaced = node.attributes.style;
        replaced = replaced.replace(urlRegex, (full, quote, innerUrl) => {
          if (urlToDataURIMap.has(innerUrl)) {
            return `url(${urlToDataURIMap.get(innerUrl)})`;
          }
          return full;
        });
        node.attributes.style = replaced;
      }
      if (node.childNodes) {
        node.childNodes.forEach(rewriteUrlsInNode);
      }
    };

    for (const event of newEvents) {
      if (event.type === EventType.FullSnapshot && event.data?.node) {
        rewriteUrlsInNode(event.data.node);
      } else if (
        event.type === EventType.IncrementalSnapshot &&
        event.data?.source === IncrementalSource.Mutation
      ) {
        if (event.data.adds) {
          event.data.adds.forEach((add) => rewriteUrlsInNode(add.node));
        }
        if (event.data.attributes) {
          event.data.attributes.forEach((attr) => {
            if (
              attr.attributes.src &&
              urlToDataURIMap.has(attr.attributes.src)
            ) {
              attr.attributes.src = urlToDataURIMap.get(attr.attributes.src);
            }
            if (attr.attributes.srcset) {
              const newSrcset = attr.attributes.srcset
                .split(",")
                .map((part) => {
                  const url = part.trim().split(" ")[0];
                  if (urlToDataURIMap.has(url)) {
                    return part.replace(url, urlToDataURIMap.get(url));
                  }
                  return part;
                })
                .join(", ");
              attr.attributes.srcset = newSrcset;
            }
          });
        }
      }
    }

    console.log("🖼️  Finished rewriting events with pre-fetched image data.");
    return newEvents;
  } catch (e) {
    console.warn(`[Image Pre-fetch] Unexpected error: ${e.message}`);
    return events;
  }
}

/**
 * Converts a JavaScript string (where char codes represent byte values) to a Uint8Array.
 */
function stringToUint8Array(str) {
  const arr = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) {
    arr[i] = str.charCodeAt(i) & 0xff; // Ensure byte values (0-255)
  }
  return arr;
}

/**
 * Decompresses a GZIP compressed string (where char codes are byte values)
 * using pako and parses the result as JSON.
 */
function unzip(compressedStringData) {
  // Check if the input is actually a string and looks like it might be gzipped
  // The first two bytes of a GZIP stream are 0x1f and 0x8b.
  if (
    typeof compressedStringData !== "string" ||
    compressedStringData.length < 2 ||
    compressedStringData.charCodeAt(0) !== 0x1f ||
    compressedStringData.charCodeAt(1) !== 0x8b
  ) {
    // If not a string or doesn't have GZIP magic numbers, assume it's not compressed
    // or it's already an object/array (e.g., uncompressed parts of incremental snapshots)
    return compressedStringData;
  }

  try {
    const byteArray = stringToUint8Array(compressedStringData);
    const decompressedUtf8String = pako.inflate(byteArray, {
      to: "string",
    });
    return JSON.parse(decompressedUtf8String);
  } catch (e) {
    console.error("Pako unzip/JSON.parse failed for a payload:", e);
    console.error(
      "Problematic payload (first 50 chars):",
      compressedStringData.substring(0, 50)
    );
    throw e; // Re-throw to be caught by decompressSingleEvent, or handle differently
  }
}

/**
 * Checks if an event is potentially compressed based on PostHog's 'cv' field.
 */
function isPotentiallyCompressedByCV(event) {
  return (
    typeof event === "object" && event !== null && typeof event.cv === "string"
  );
}
/**
 * Decompresses a single PostHog session recording event if it matches known compression formats.
 */
function decompressSingleEvent(event) {
  // If the event doesn't have the 'cv' field indicating compression version,
  // but its data is a string (which could be GZIP), we might still try to unzip.
  // However, the provided TypeScript strictly checks 'cv'. We will adapt.

  if (!isPotentiallyCompressedByCV(event)) {
    // If no 'cv' field, but data is a string and type is FullSnapshot, it might still be compressed
    // This is an assumption if 'cv' might be missing on some compressed events.
    // For now, let's stick closer to the user-provided TS logic which checks 'cv'.
    return event;
  }

  try {
    if (event.cv === "2024-10") {
      if (event.type === EventType.FullSnapshot) {
        // For FullSnapshot, the 'data' field is the compressed payload
        const decompressedData = unzip(event.data);
        return { ...event, data: decompressedData };
      } else if (
        event.type === EventType.IncrementalSnapshot &&
        typeof event.data === "object" &&
        event.data !== null
      ) {
        const newData = { ...event.data }; // Shallow copy the data object
        let changed = false; // Flag to see if any part was decompressed

        const fieldsToDecompressInMutation = [
          "adds",
          "removes",
          "texts",
          "attributes",
        ];
        const fieldsToDecompressInStyleSheet = ["adds", "removes"];

        if (event.data.source === IncrementalSource.StyleSheetRule) {
          fieldsToDecompressInStyleSheet.forEach((field) => {
            if (typeof newData[field] === "string") {
              newData[field] = unzip(newData[field]);
              changed = true;
            }
          });
        } else if (event.data.source === IncrementalSource.Mutation) {
          fieldsToDecompressInMutation.forEach((field) => {
            if (typeof newData[field] === "string") {
              newData[field] = unzip(newData[field]);
              changed = true;
            }
          });
        } else if (event.data.source === IncrementalSource.Font) {
          fieldsToDecompressInStyleSheet.forEach((field) => {
            if (typeof newData[field] === "string") {
              newData[field] = unzip(newData[field]);
              changed = true;
            }
          });
        }
        // If any sub-field was changed (decompressed), return the event with the new data object
        return changed ? { ...event, data: newData } : event;
      }
      // If compression version is '2024-10' but doesn't match known compressible structures, return as is.
      return event;
    } else {
      console.warn(
        `Unknown or unhandled compressed event version: ${event.cv}. Event not decompressed.`,
        event
      );
      return event; // Unknown compression version, return as is
    }
  } catch (e) {
    console.error(
      "Error during decompressSingleEvent for event:",
      e,
      JSON.stringify(event).substring(0, 500) + "..."
    );
    return event; // Return original event on error during decompression
  }
}

// Normalize navigation pairs where a Page view and Page leave for the same URL
// end up with effectively the same timestamp (e.g., inference + matching artifacts).
// In such cases, prefer keeping the Page view and drop the simultaneous Page leave.
function normalizeSimultaneousSameUrlNavs(events, thresholdMs = 5) {
  if (!Array.isArray(events) || events.length === 0) return events;

  const navInfo = (text) => {
    if (typeof text !== "string") return null;
    const m = text.match(
      /^(?:\[PRE-RECORDING \(mention in the output\)\]\s*)?(Page view|Page leave):\s*(.+)$/
    );
    if (!m) return null;
    return { kind: m[1] === "Page view" ? "view" : "leave", url: m[2].trim() };
  };

  const out = [];
  for (let i = 0; i < events.length; i++) {
    const a = events[i];
    const aInfo = navInfo(a?.text);
    const next = events[i + 1];
    const bInfo = navInfo(next?.text);

    if (
      a &&
      next &&
      aInfo &&
      bInfo &&
      aInfo.url === bInfo.url &&
      ((aInfo.kind === "view" && bInfo.kind === "leave") ||
        (aInfo.kind === "leave" && bInfo.kind === "view"))
    ) {
      const aMs = typeof a.adjustedMs === "number" ? a.adjustedMs : 0;
      const bMs = typeof next.adjustedMs === "number" ? next.adjustedMs : 0;
      if (Math.abs(aMs - bMs) <= thresholdMs) {
        // Drop the leave, keep the view
        const keep = aInfo.kind === "view" ? a : next;
        // Push the kept one once
        out.push(keep);
        // Skip over the paired one
        i += 1;
        continue;
      }
    }
    out.push(a);
  }
  return out;
}

function insertInferredPageLeaves(eventLog, initialPageUrl = null) {
  console.log("\n--- [DEBUG] ENTERING insertInferredPageLeaves ---");
  console.log(`[DEBUG] initialPageUrl: ${initialPageUrl}`);
  console.log(`[DEBUG] Incoming eventLog length: ${eventLog.length}`);
  if (eventLog.length > 0 && eventLog.length < 20) {
    console.log(
      "[DEBUG] Incoming eventLog content:",
      JSON.stringify(
        eventLog.map((e) => ({
          adjustedMs: e.adjustedMs,
          text: e.text,
          source: e.source,
        })),
        null,
        2
      )
    );
  }

  if (!eventLog || eventLog.length < 1) {
    console.log("[DEBUG] EXITING insertInferredPageLeaves (empty log).");
    return eventLog;
  }

  const result = [];
  let lastUrl = initialPageUrl;
  let hasSeenFirstPageView = Boolean(initialPageUrl);

  const urlRegex =
    /(?:Page view|Page leave):\s*(?:\[PRE-RECORDING \(mention in the output\)\]\s*)?(.+)/;
  const shiftOneMsEarlier = (value) =>
    typeof value === "number" ? Math.max(0, value - 1) : value;

  for (const [index, event] of eventLog.entries()) {
    const isPageView = event.text?.includes("Page view:");
    const isPageLeave = event.text?.includes("Page leave:");

    if (isPageView) {
      const match = event.text.match(urlRegex);
      const newUrl = match ? match[1].trim() : null;
      console.log(`[DEBUG]   - Event is a PageView.`);
      console.log(
        `[DEBUG]   - State before check: lastUrl='${lastUrl}', newUrl='${newUrl}', hasSeenFirstPageView=${hasSeenFirstPageView}`
      );

      if (lastUrl && newUrl) {
        if (hasSeenFirstPageView) {
          console.log(
            `[DEBUG]   -> CONDITION MET: Inserting page leave for "${lastUrl}".`
          );
          const inferredAdjustedMs = shiftOneMsEarlier(event.adjustedMs);
          const inferredLeaveEvent = {
            adjustedMs: inferredAdjustedMs,
            originalMs: shiftOneMsEarlier(event.originalMs),
            originalAbsoluteMs: shiftOneMsEarlier(event.originalAbsoluteMs),
            text: `Page leave: ${lastUrl}`,
            isPreRecording: event.isPreRecording,
            eventIndex: `inferred-leave-${event.eventIndex}`,
            source: "system-inferred-page-leave",
            originalEvent: null,
          };
          result.push(inferredLeaveEvent);
        } else {
          console.log(
            `[DEBUG]   -> CONDITION NOT MET: hasSeenFirstPageView is false.`
          );
        }
      } else if (lastUrl && newUrl && newUrl === lastUrl) {
        console.log(
          `[DEBUG]   -> SKIP inferring leave: newUrl equals lastUrl (no transition)`
        );
      } else {
        console.log(
          `[DEBUG]   -> CONDITION NOT MET for inserting leave. Reason: lastUrl='${lastUrl}', newUrl='${newUrl}'`
        );
      }
      result.push(event);
      lastUrl = newUrl;
      hasSeenFirstPageView = true;
      if (newUrl) {
        console.log(
          `[DEBUG]   - State after: lastUrl updated to '${lastUrl}', hasSeenFirstPageView=true`
        );
      }
    } else if (isPageLeave) {
      console.log(`[DEBUG]   - Event is a PageLeave.`);
      result.push(event);
      const match = event.text.match(urlRegex);
      const leaveUrl = match ? match[1].trim() : null;
      if (leaveUrl && leaveUrl === lastUrl) {
        console.log(
          `[DEBUG]   - Clearing lastUrl because a matching leave was found for "${leaveUrl}".`
        );
        lastUrl = null;
      }
    } else {
      result.push(event);
    }
  }

  // Final normalization to avoid confusing simultaneous view/leave pairs for the same URL
  console.log("\n[DEBUG] --- Final result from insertInferredPageLeaves ---");
  if (result.length > 0 && result.length < 20) {
    console.log(
      JSON.stringify(
        result.map((e) => ({
          adjustedMs: e.adjustedMs,
          text: e.text,
          source: e.source,
        })),
        null,
        2
      )
    );
  }
  console.log(`[DEBUG] Final result length: ${result.length}`);
  console.log("--- [DEBUG] EXITING insertInferredPageLeaves ---\n");
  return result;
}

function processPostHogRecording(jsonData) {
  try {
    // jsonData is now the already parsed PostHog data

    let posthogData = jsonData;

    const decompressedSnapshots = posthogData.map((snapshot) =>
      decompressSingleEvent(snapshot)
    );

    console.log("Transforming to standard rrweb event format...");
    const processedEvents = decompressedSnapshots
      .filter((event) => event && event.type !== EventType.Plugin)
      .map((event) => {
        if (
          typeof event.type !== "number" ||
          typeof event.timestamp !== "number" ||
          typeof event.data === "undefined"
        ) {
          console.warn(
            "Warning: Event missing core fields after decompression, skipping:",
            event
          );
          return null;
        }
        return {
          type: event.type,
          data: event.data,
          timestamp: event.timestamp,
        };
      });

    console.log("Sorting events by timestamp...");
    // CRITICAL FIX: The events must be sorted by timestamp before processing.
    // PostHog can return snapshot chunks that are not.

    processedEvents.sort((a, b) => a.timestamp - b.timestamp);
    /*   try {


  // outputFiles are the reason why the build has > 7mb! They are not neccesarry, comment this out to reduce the build size
      fs2.writeFileSync(
        "outputFile.json",
        JSON.stringify(processedEvents, null, 2)
      ); // null, 2 for pretty printing
    } catch (error) {
      console.error("An error occurred during writing to file:", error);
    }
 */
    return processedEvents;
  } catch (error) {
    console.error("An error occurred during pre-processing:", error);
    if (error instanceof SyntaxError) {
      // This error is less likely now as parsing is done outside
      console.error(
        "This might be due to an issue with the JSON format if it was manually constructed."
      );
    }
  }
}

const apiKeys = [process.env.GEMINI_API_KEY_0];

console.log(`🔑 Configured ${apiKeys.length} API keys for rotation`);

const ai = new GoogleGenAI({
  apiKey: apiKeys[0],
});
function splitAndReconstructArrayStrings(inputString) {
  // The delimiter used to split the input string.
  const delimiter = "]\n[";

  // Split the input string into parts using the delimiter.
  // For example, if inputString is "[val1]\n[val2]",
  // parts will be ["[val1", "val2]"].
  const parts = inputString.split(delimiter);

  // If the input string is empty, split() on an empty string yields [""].
  // If the input string doesn't contain the delimiter, it returns the original string in an array.
  // The mapping logic below handles these cases correctly.

  // Map over the parts to reconstruct them.
  const reconstructedParts = parts.map((part, index, array) => {
    let currentPart = part;

    // If this is not the first part (index > 0),
    // it means the opening "[" was removed by the split.
    // So, we prepend "[" to this part.
    if (index > 0) {
      currentPart = "[" + currentPart;
    }

    // If this is not the last part (index < array.length - 1),
    // it means the closing "]" was removed by the split.
    // So, we append "]" to this part.
    if (index < array.length - 1) {
      currentPart = currentPart + "]";
    }

    return currentPart;
  });

  return reconstructedParts;
}

async function fetchWithRetry(url, config, maxRetries = 3) {
  let attempt = 0;
  const cfg = { ...(config || {}) };
  if (cfg.timeout == null) {
    cfg.timeout = 60000; // default 60s request timeout
  }
  while (attempt < maxRetries) {
    try {
      const response = await axios(url, cfg);
      return response;
    } catch (error) {
      if (error.response && error.response.status === 429) {
        attempt++;
        const retryAfter = error.response.headers["retry-after"];
        const waitTime = retryAfter
          ? parseInt(retryAfter, 10) * 1000
          : Math.pow(2, attempt) * 1000;
        if (waitTime >= 60000) {
          console.error(
            `API retry-after time (${waitTime}ms) is >= 60 seconds. Not waiting.`
          );
          throw new Error(
            `Rate limit retry-after >= 60 seconds. Aborting this request.`
          );
        }
        console.log(
          `Request throttled. Retrying after ${waitTime} ms. Attempt ${attempt} of ${maxRetries}.`
        );
        await new Promise((resolve) => setTimeout(resolve, waitTime));
      } else {
        throw error;
      }
    }
  }
  throw new Error(
    `Failed to fetch data from ${url} after ${maxRetries} attempts.`
  );
}

// Attempts to extract a server-suggested retry delay (in ms) from Gemini API errors
function getRetryDelayMsFromError(error) {
  try {
    const asString = `${error?.message || ""} ${String(error || "")}`;
    const match = asString.match(/"retryDelay"\s*:\s*"(\d+)s"/);
    if (match && match[1]) {
      const seconds = parseInt(match[1], 10);
      if (!Number.isNaN(seconds)) return seconds * 1000;
    }
  } catch (e) {
    // ignore
  }
  return null;
}

function isEmptyMetadataValue(value) {
  if (value === null || value === undefined) {
    return true;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed === "" || trimmed.toUpperCase() === "N/A") {
      return true;
    }
  }
  return false;
}

async function fetchSessionMetadataFromQuery(
  sessionReplayId,
  projectId,
  apiKey,
  API_BASE_URL
) {
  const queryBody = {
    query: {
      kind: "HogQLQuery",
      query: `
        SELECT * FROM events 
        WHERE properties.$session_id = '${sessionReplayId}'
        ORDER BY timestamp ASC
      `,
    },
  };

  const endpoints = [];
  if (API_BASE_URL) {
    endpoints.push(`${API_BASE_URL}/projects/${projectId}/query/`);
  }

  if (endpoints.length === 0) {
    return null;
  }

  let lastError = null;

  for (const endpoint of endpoints) {
    try {
      const response = await fetchWithRetry(endpoint, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        data: queryBody,
      });

      const columns = response?.data?.columns || [];
      const results = response?.data?.results || [];

      if (
        !Array.isArray(columns) ||
        columns.length === 0 ||
        !Array.isArray(results) ||
        results.length === 0
      ) {
        continue;
      }

      const events = results.map((row) => {
        const eventObject = {};
        columns.forEach((col, index) => {
          eventObject[col] = row[index];
        });
        if (typeof eventObject.properties === "string") {
          try {
            eventObject.properties = JSON.parse(eventObject.properties);
          } catch (_) {
            // Ignore JSON parse errors and leave properties as-is
          }
        }
        return eventObject;
      });

      const getTimestampMs = (event) => {
        if (!event || !event.timestamp) {
          return null;
        }
        const parsed = Date.parse(event.timestamp);
        return Number.isFinite(parsed) ? parsed : null;
      };

      const firstTimestampMs = getTimestampMs(events[0]);
      const lastTimestampMs = getTimestampMs(events[events.length - 1]);

      const findPropValue = (keys) => {
        for (const event of events) {
          const props = event?.properties;
          if (!props || typeof props !== "object") {
            continue;
          }
          for (const key of keys) {
            const value = props[key];
            if (!isEmptyMetadataValue(value)) {
              return value;
            }
          }
        }
        return null;
      };

      const findEventValue = (keys) => {
        for (const event of events) {
          for (const key of keys) {
            const value = event?.[key];
            if (!isEmptyMetadataValue(value)) {
              return value;
            }
          }
        }
        return null;
      };

      const recordingDurationSeconds =
        firstTimestampMs !== null &&
        lastTimestampMs !== null &&
        lastTimestampMs >= firstTimestampMs
          ? String(Math.max(0, (lastTimestampMs - firstTimestampMs) / 1000))
          : null;

      const clickCount = events.filter(
        (event) =>
          event?.event === "$autocapture" &&
          event?.properties &&
          typeof event.properties === "object" &&
          event.properties.$event_type === "click"
      ).length;

      let serverStartUrl =
        findPropValue([
          "$session_entry_url",
          "$current_url",
          "$initial_current_url",
        ]) || null;

      if (isEmptyMetadataValue(serverStartUrl)) {
        const protocolCandidate = findPropValue([
          "$session_entry_protocol",
          "$protocol",
        ]);
        const hostCandidate = findPropValue(["$session_entry_host", "$host"]);
        const pathCandidate = findPropValue([
          "$session_entry_pathname",
          "$pathname",
        ]);
        if (
          !isEmptyMetadataValue(hostCandidate) &&
          !isEmptyMetadataValue(pathCandidate)
        ) {
          const normalizedProtocol =
            typeof protocolCandidate === "string" &&
            protocolCandidate.length > 0
              ? protocolCandidate.replace(/:$/, "")
              : "https";
          serverStartUrl = `${normalizedProtocol}://${hostCandidate}${pathCandidate}`;
        }
      }

      return {
        serverStartUrl,
        recording_duration: recordingDurationSeconds,
        click_count: String(clickCount),
        datetime:
          firstTimestampMs !== null
            ? new Date(firstTimestampMs).toISOString()
            : null,
        person_id: findEventValue([
          "person_id",
          "event_person_id",
          "distinct_id",
        ]),
        browser: findPropValue(["$browser"]),
        osVar: findPropValue(["$initial_os", "$os", "$os_name"]),
        osVersion: findPropValue(["$os_version"]),
        device_type: findPropValue(["$initial_device_type", "$device_type"]),
        country: findPropValue(["$geoip_country_name"]),
        referrer: findPropValue(["$session_entry_referrer", "$referrer"]),
        idVar:
          findEventValue(["id", "uuid"]) ||
          findPropValue(["$session_id"]) ||
          sessionReplayId,
        user_agent: findPropValue(["$raw_user_agent"]),
      };
    } catch (error) {
      lastError = error;
      if (error?.message && error.message.includes("Session terminated")) {
        throw error;
      }
      console.log(
        `Failed to fetch session metadata from ${endpoint}:`,
        error?.message || error
      );
    }
  }

  if (lastError) {
    console.log(
      "Unable to backfill session metadata from PostHog /query endpoint."
    );
  }

  return null;
}

async function getEntireSessionReplay(
  projectId,
  sessionId,
  apiKey,
  API_BASE_URL,
  API_BASE_URL2
) {
  let allSnapshots = [];
  let startUrl = "N/A";
  let serverStartUrl = null;
  let active_seconds = null;
  let recording_duration = "N/A";
  let click_count = "N/A";
  let end_time = "N/A";
  let datetime = "N/A";
  let processedAt = null;
  let person_id = "N/A";
  let browser = "N/A";
  let osVar = "N/A";
  let osVersion = "N/A";
  let device_type = "N/A";
  let country = "N/A";
  let referrer = "N/A";
  let idVar = "N/A";
  let user_agent = "";

  let initialResponseData; // To store data from either initial or fallback call

  try {
    try {
      console.log("fetching additional data for start_url");
      let response;
      let response2;
      try {
        console.log("hereeee");
        response = await fetchWithRetry(
          `${API_BASE_URL}/projects/${projectId}/session_recordings/${sessionId}`,
          {
            headers: {
              Authorization: `Bearer ${apiKey}`,
              "Content-Type": "application/json",
            },
          }
        );
      } catch (error) {
        if (error.message.includes("Session terminated")) {
          throw error;
        }
        console.log("Error fetching additional data for start_url: ", error);
        response = await fetchWithRetry(
          `${API_BASE_URL2}/projects/${projectId}/session_recordings/${sessionId}`,
          {
            headers: {
              Authorization: `Bearer ${apiKey}`,
              "Content-Type": "application/json",
            },
          }
        );

        API_BASE_URL = API_BASE_URL2;
        console.log("hereeeee");
      }

      console.log("fjewnfjfwejfwenfjwne ", response.data);
      serverStartUrl = response.data?.start_url || "";
      active_seconds = response?.data?.active_seconds;
      recording_duration = response.data?.recording_duration || "";
      click_count = response.data?.click_count || "";
      datetime = response.data?.start_time || "";
      end_time = response.data?.end_time || "";
      person_id = response.data?.person?.id || "";
      browser = response.data?.person?.properties?.$browser || "";
      osVar = response.data?.person?.properties?.$initial_os || "";
      referrer = "";
      idVar = response.data?.id || "";
      osVersion = response.data?.person?.properties?.$os_version || "";
      device_type =
        response.data?.person?.properties?.$initial_device_type || "";
      country = response.data?.person?.properties?.$geoip_country_name || "";
      // If start_url wasn't present, retry once after a short delay
      if (!serverStartUrl) {
        console.log(
          "start_url not found in first attempt; retrying once after delay..."
        );
        await new Promise((resolve) => setTimeout(resolve, 2000));
        try {
          // Try the current primary base URL first
          response = await fetchWithRetry(
            `${API_BASE_URL}/projects/${projectId}/session_recordings/${sessionId}`,
            {
              headers: {
                Authorization: `Bearer ${apiKey}`,
                "Content-Type": "application/json",
              },
            }
          );
        } catch (retryErr) {
          await new Promise((resolve) => setTimeout(resolve, 2000));
          if (
            retryErr.message &&
            retryErr.message.includes("Session terminated")
          ) {
            throw retryErr;
          }
          try {
            // Try the current primary base URL first
            response = await fetchWithRetry(
              `${API_BASE_URL}/projects/${projectId}/session_recordings/${sessionId}`,
              {
                headers: {
                  Authorization: `Bearer ${apiKey}`,
                  "Content-Type": "application/json",
                },
              }
            );
          } catch (retryErr) {
            await new Promise((resolve) => setTimeout(resolve, 2000));

            if (
              retryErr.message &&
              retryErr.message.includes("Session terminated")
            ) {
              throw retryErr;
            }
            console.log(
              "Retry (primary) fetching additional data for start_url failed:",
              retryErr
            );
            // Fallback to secondary base URL
            response = await fetchWithRetry(
              `${API_BASE_URL}/projects/${projectId}/session_recordings/${sessionId}`,
              {
                headers: {
                  Authorization: `Bearer ${apiKey}`,
                  "Content-Type": "application/json",
                },
              }
            );

          
          }
        }

        // Re-assign fields from the retry response
        serverStartUrl = response.data?.start_url || "";
        active_seconds = response?.data?.active_seconds;


        recording_duration =
          response.data?.recording_duration || recording_duration || "";
        click_count = response.data?.click_count || click_count || "";
        datetime = response.data?.start_time || datetime || "";
        end_time = response.data?.end_time || "";

        person_id = response.data?.person?.id || person_id || "";
        browser = response.data?.person?.properties?.$browser || browser || "";
        osVar = response.data?.person?.properties?.$initial_os || osVar || "";
        referrer = "";
        idVar = response.data?.id || idVar || "";
        osVersion =
          response.data?.person?.properties?.$os_version || osVersion || "";
        device_type =
          response.data?.person?.properties?.$initial_device_type ||
          device_type ||
          "";
        country =
          response.data?.person?.properties?.$geoip_country_name ||
          country ||
          "";
      }
    } catch (error) {
      console.log("Error fetching additional data for start_url: ", error);
    }

console.log("active_secondsactive_secondsactive_seconds ", active_seconds)


    try {
     const properties = [
    {
      key: "$session_id",
      value: [sessionId],
      operator: "exact",
      type: "event",
    },
  ];


      const url = `${API_BASE_URL}/projects/${projectId}/session_recordings/`
/*    +
  "?kind=RecordingsQuery&properties=" +
  encodeURIComponent(JSON.stringify(body)) +
  "&operand=AND"; */
    // Fallback to secondary base URL
    let response2 = await fetchWithRetry(url
      ,
      {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        params: {
          kind: "RecordingsQuery",
          properties: JSON.stringify(properties), // <-- only the list
          operand: "AND",
        },
      }
    );

if(response2?.data.results[0]?.ongoing) {
    console.log("Paseeeeeed2")
    const reason = `Session is ongoing`;
    console.log(reason, sessionId);
    const error = new Error(reason);
      error.message = reason;
      throw error;
}
    } catch(errr) {
      console.log("erejnrewjnrew ", errr)
      throw errr
    }
    console.log("Paseeeeeed")
/*     return; */

    if (active_seconds) {
      if((process.env.RUN_PARTICULAR_SESSION==="true" && active_seconds > 1000) || (process.env.RUN_PARTICULAR_SESSION!="true" && active_seconds > 720)) {
        const reason = `Session contains more than 720 seconds (count=${active_seconds}), skipping session`;
        console.log(reason, sessionId);
        const error = new Error(reason);
      error.name = "TooManySecondsError";
      error.sessionId = sessionId;
      error.message = reason;
      throw error;
    }
    }





















    const metadataValues = {
      serverStartUrl,
      recording_duration,
      click_count,
      end_time,
      datetime,
      person_id,
      browser,
      osVar,
      referrer,
      idVar,
      osVersion,
      device_type,
      country,
      user_agent,
    };

    const needsMetadataFallback = Object.values(metadataValues).some((value) =>
      isEmptyMetadataValue(value)
    );

    if (needsMetadataFallback) {
      try {
        const fallbackMetadata = await fetchSessionMetadataFromQuery(
          sessionId,
          projectId,
          apiKey,
          API_BASE_URL
        );

        console.log("Fnkewfnwefnwjfnwe ", fallbackMetadata);
        if (fallbackMetadata) {
          const updatedFields = [];
          if (
            isEmptyMetadataValue(serverStartUrl) &&
            !isEmptyMetadataValue(fallbackMetadata.serverStartUrl)
          ) {
            serverStartUrl = fallbackMetadata.serverStartUrl;
            updatedFields.push("serverStartUrl");
          }
          if (
            isEmptyMetadataValue(recording_duration) &&
            !isEmptyMetadataValue(fallbackMetadata.recording_duration)
          ) {
            recording_duration = fallbackMetadata.recording_duration;
            updatedFields.push("recording_duration");
          }
          if (
            isEmptyMetadataValue(click_count) &&
            !isEmptyMetadataValue(fallbackMetadata.click_count)
          ) {
            click_count = fallbackMetadata.click_count;
            updatedFields.push("click_count");
          }
          if (
            isEmptyMetadataValue(datetime) &&
            !isEmptyMetadataValue(fallbackMetadata.datetime)
          ) {
            datetime = fallbackMetadata.datetime;
            updatedFields.push("datetime");
          }

          if (
            isEmptyMetadataValue(end_time) &&
            !isEmptyMetadataValue(fallbackMetadata.end_time)
          ) {
            end_time = fallbackMetadata.end_time;
            updatedFields.push("end_time");
          }
          if (
            isEmptyMetadataValue(person_id) &&
            !isEmptyMetadataValue(fallbackMetadata.person_id)
          ) {
            person_id = fallbackMetadata.person_id;
            updatedFields.push("person_id");
          }
          if (
            isEmptyMetadataValue(browser) &&
            !isEmptyMetadataValue(fallbackMetadata.browser)
          ) {
            browser = fallbackMetadata.browser;
            updatedFields.push("browser");
          }
          if (
            isEmptyMetadataValue(osVar) &&
            !isEmptyMetadataValue(fallbackMetadata.osVar)
          ) {
            osVar = fallbackMetadata.osVar;
            updatedFields.push("osVar");
          }
          if (
            isEmptyMetadataValue(referrer) &&
            !isEmptyMetadataValue(fallbackMetadata.referrer)
          ) {
            referrer = fallbackMetadata.referrer;
            updatedFields.push("referrer");
          }
          if (
            isEmptyMetadataValue(idVar) &&
            !isEmptyMetadataValue(fallbackMetadata.idVar)
          ) {
            idVar = fallbackMetadata.idVar;
            updatedFields.push("idVar");
          }
          if (
            isEmptyMetadataValue(osVersion) &&
            !isEmptyMetadataValue(fallbackMetadata.osVersion)
          ) {
            osVersion = fallbackMetadata.osVersion;
            updatedFields.push("osVersion");
          }
          if (
            isEmptyMetadataValue(device_type) &&
            !isEmptyMetadataValue(fallbackMetadata.device_type)
          ) {
            device_type = fallbackMetadata.device_type;
            updatedFields.push("device_type");
          }
          if (
            isEmptyMetadataValue(country) &&
            !isEmptyMetadataValue(fallbackMetadata.country)
          ) {
            country = fallbackMetadata.country;
            updatedFields.push("country");
          }
          if (
            isEmptyMetadataValue(user_agent) &&
            !isEmptyMetadataValue(fallbackMetadata.user_agent)
          ) {
            user_agent = fallbackMetadata.user_agent;
            updatedFields.push("user_agent");
          }

          if (updatedFields.length > 0) {
            console.log(
              `Backfilled session metadata from /query for fields: ${updatedFields.join(
                ", "
              )}`
            );
          } else {
            console.log(
              "Attempted to backfill session metadata from /query but no additional fields were populated."
            );
          }
        } else {
          console.log(
            "No data returned from /query to backfill missing session metadata."
          );
        }
      } catch (fallbackError) {
        if (
          fallbackError?.message &&
          fallbackError.message.includes("Session terminated")
        ) {
          throw fallbackError;
        }
        console.log(
          "Failed to backfill session metadata from PostHog /query endpoint:",
          fallbackError?.message || fallbackError
        );
      }
    }
    console.log("serverStartUrlserverStartUrlserverStartUrl ", serverStartUrl);
    const numericClickCount = Number(click_count);
      if(!Number.isNaN(numericClickCount) && (process.env.RUN_PARTICULAR_SESSION==="true" && numericClickCount > 150) || (process.env.RUN_PARTICULAR_SESSION!="true" &&numericClickCount > 60)) {
      const reason = `Session contains more than 60 clicks (count=${numericClickCount}), skipping session`;
      console.log(reason, sessionId);
      const error = new Error(reason);
      error.name = "TooManyClicksError";
      error.sessionId = sessionId;
      error.message = reason;
      throw error;
    }

    // 1. Initial call to get sources (using blob_v2)
    const initialUrl = `${API_BASE_URL}/environments/${projectId}/session_recordings/${sessionId}/snapshots?blob_v2=1`;
    console.log("Fetching sources from (attempt 1 - blob_v2):", initialUrl);
    try {
      const initialResponse = await fetchWithRetry(initialUrl, {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      });
      if (initialResponse.status !== 200) {
        throw new Error(
          `Error fetching sources (blob_v2): ${initialResponse} ${initialResponse}`
        );
      }
      initialResponseData = initialResponse.data;

    console.log("nfjfwfjenfjwn ", initialResponse)

      console.log("Successfully fetched sources using blob_v2.");
    } catch (error) {
      if (error.message.includes("Session terminated")) {
        throw error;
      }
      console.warn(
        `Failed to fetch sources using blob_v2: ${error}. Trying fallback...`
      );
      const fallbackUrl = `${API_BASE_URL}/environments/${projectId}/session_recordings/${sessionId}/snapshots`;
      console.log("Fetching sources from (attempt 2 - fallback):", fallbackUrl);
      const fallbackResponse = await fetchWithRetry(fallbackUrl, {
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
      });
      if (fallbackResponse.status !== 200) {
        throw new Error(
          `Error fetching sources (fallback): ${fallbackResponse.status} ${fallbackResponse.statusText}`
        );
      }
      initialResponseData = fallbackResponse.data;
      console.log("Successfully fetched sources using fallback.");
    }

    const sourcesData = initialResponseData; // Use the data from whichever call succeeded
    let sourcesToFetch = sourcesData.sources || [];

    if (sourcesToFetch.length > 2000) {
      const reason = `Session contains more than 2000 blobs (count=${sourcesToFetch.length}), skipping session`;
      console.log(reason, sessionId);
      const error = new Error(reason);
      error.name = "TooManyBlobsError";
      error.sessionId = sessionId;
      error.message = reason;
      throw error;
    }

    // If a realtime source is present, the session is still active and we should skip it.
    const hasRealtimeSource = sourcesToFetch.some(
      (s) => s.source === "realtime"
    );
    if (hasRealtimeSource) {
      console.log(
        `Session ${sessionId} has a 'realtime' source, indicating it is ongoing. Skipping.`
      );
      return null;
    }

    // 2. Filter sources - Prefer blob_v2 if available (though initial fetch already tried this preference)
    const blobV2Sources = sourcesToFetch.filter((s) => s.source === "blob_v2");


    if (blobV2Sources.length > 0) {
      console.log("Using blob_v2 sources.");
      sourcesToFetch = blobV2Sources;
    } else {
      // Fallback logic if no blob_v2 sources (e.g., use 'blob' and potentially 'realtime')
      // For an "entire" replay, you'd ideally wait until 'realtime' is no longer a source.
      // If only 'realtime' is present, the session is very new and not fully processed.
      console.log(
        "No blob_v2 sources found. Using available sources (blob/realtime)."
      );
      const blobSources = sourcesToFetch.filter((s) => s.source === "blob");

      const realtimeSource = sourcesToFetch.find(
        (s) => s.source === "realtime"
      );
      sourcesToFetch = [...blobSources]; // Prioritize blob sources

      // If you must get data for a very recent session and only realtime is available,
      // you might add it, but be aware it's not a complete historical record.
      // For complete replay, polling for new sources is needed if realtime is still active or
      // if you expect more blob_v2 sources to appear for an ongoing session.
      // The doc suggests polling the original snapshot API for new sources with blob_v2.
      if (realtimeSource && blobSources.length === 0) {
        console.warn(
          "Only realtime source available. Replay may be incomplete or still active. Consider waiting."
        );
        // sourcesToFetch.push(realtimeSource); // Not typically done for "entire" historical export
      }
    }

    if (sourcesToFetch.length === 0) {
      console.log(
        "No suitable sources found to fetch data. The recording might be too new or empty."
      );
      return [[], ""];
    }

    // 3. Fetch data for sources. For blob_v2, batch by ranges (max 20 keys per request)
    const isAllBlobV2 = sourcesToFetch.every((s) => s.source === "blob_v2");
    if (isAllBlobV2) {
      // Collect, sort, and batch consecutive blob_key values into ranges of up to 20
      const keys = sourcesToFetch
        .map((s) => Number(s.blob_key))
        .filter((n) => Number.isFinite(n))
        .sort((a, b) => a - b);

      // Helper: parse response and push into allSnapshots
      function pushFromResponse(dataResponse, label) {
        let data = [];
        if (typeof dataResponse.data === "string") {
          data = splitAndReconstructArrayStrings(dataResponse.data);
          try {
            data = data.map((value) => JSON.parse(value));
          } catch (error) {
            const lines = dataResponse.data.trim().split("\n");
            const snapshotEvents = [];
            for (const line of lines) {
              if (line.trim() === "") continue;
              try {
                snapshotEvents.push(JSON.parse(line));
              } catch (parseError) {
                console.warn(
                  `Skipping line due to JSON parse error: ${parseError.message} on line: ${line}`
                );
              }
            }

            if (Array.isArray(snapshotEvents)) {
              allSnapshots.push(...snapshotEvents);
              return;
            } else if (
              snapshotEvents &&
              Array.isArray(snapshotEvents.snapshot_data)
            ) {
              allSnapshots.push(...snapshotEvents.snapshot_data);
              return;
            } else {
              console.warn(
                `Unexpected data format from blob_v2 range ${label}:`,
                snapshotEvents
              );
              return;
            }
          }
        } else {
          data = dataResponse.data;
        }
        allSnapshots.push(data);
      }

      // Helper: fetch a range; on timeout, split and retry with smaller ranges
      async function fetchBlobV2Range(startInclusive, endInclusive) {
        const label = `${startInclusive}-${endInclusive}`;
        const count = endInclusive - startInclusive + 1;
        const rangeUrl = `${API_BASE_URL}/environments/${projectId}/session_recordings/${sessionId}/snapshots?source=blob_v2&start_blob_key=${startInclusive}&end_blob_key=${endInclusive}`;
        console.log(
          `Fetching blob_v2 range: start=${startInclusive}, end(inclusive)=${endInclusive}`
        );
        try {
          const dataResponse = await fetchWithRetry(rangeUrl, {
            headers: {
              Authorization: `Bearer ${apiKey}`,
              "Content-Type": "application/json",
            },
            timeout: 60000,
          });
          if (dataResponse.status !== 200) {
            throw new Error(
              `Error fetching blob_v2 range ${label}: ${dataResponse.status} ${dataResponse.statusText}`
            );
          }
          pushFromResponse(dataResponse, label);
        } catch (error) {
          const msg = String(error?.message || error || "");
          const isTimeout = error?.code === "ECONNABORTED" || /timeout/i.test(msg);
          const isGatewayTimeout = error?.response?.status === 504 || error?.response?.status === 408;
          if ((isTimeout || isGatewayTimeout) && count > 1) {
            const mid = Math.floor((startInclusive + endInclusive) / 2);
            console.warn(
              `[SNAPSHOTS] Range ${label} timed out. Retrying as smaller ranges: ${startInclusive}-${mid} and ${mid + 1}-${endInclusive}`
            );
            await fetchBlobV2Range(startInclusive, mid);
            await fetchBlobV2Range(mid + 1, endInclusive);
            return;
          }
          throw error;
        }
      }

      let i = 0;
      while (i < keys.length) {
        const startKey = keys[i];
        let count = 1;
        // extend the batch while keys are consecutive and under the 20-key limit
        while (
          i + count < keys.length &&
          keys[i + count] === startKey + count &&
          count < 20
        ) {
          count++;
        }
        const lastKey = keys[i + count - 1];
        await fetchBlobV2Range(startKey, lastKey);
        i += count;
      }
    } else {
      // Fallback: fetch each source individually (blob or others)
      for (const sourceInfo of sourcesToFetch) {
        if (sourceInfo.source === "realtime" && !sourceInfo.blob_key) {
          console.log(
            `Skipping realtime source for full replay export: ${JSON.stringify(
              sourceInfo
            )}`
          );
          continue;
        }
        if (!sourceInfo.blob_key) {
          console.warn(
            `Skipping source without blob_key: ${JSON.stringify(sourceInfo)}`
          );
          continue;
        }

        const dataUrl = `${API_BASE_URL}/environments/${projectId}/session_recordings/${sessionId}/snapshots?source=${sourceInfo.source}&blob_key=${sourceInfo.blob_key}`;
        console.log(`Fetching data from: ${dataUrl}`);
        const dataResponse = await fetchWithRetry(dataUrl, {
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
        });
        if (dataResponse.status !== 200) {
          throw new Error(
            `Error fetching data: ${dataResponse.status} ${dataResponse.statusText}`
          );
        }
        let data = [];
        if (typeof dataResponse.data === "string") {
          data = splitAndReconstructArrayStrings(dataResponse.data);
          try {
            data = data.map((value) => JSON.parse(value));
          } catch (error) {
            const lines = dataResponse.data.trim().split("\n");
            const snapshotEvents = [];
            for (const line of lines) {
              if (line.trim() === "") continue;
              try {
                snapshotEvents.push(JSON.parse(line));
              } catch (parseError) {
                console.warn(
                  `Skipping line due to JSON parse error: ${parseError.message} on line: ${line}`
                );
              }
            }

            if (Array.isArray(snapshotEvents)) {
              allSnapshots.push(...snapshotEvents);
            } else if (
              snapshotEvents &&
              Array.isArray(snapshotEvents.snapshot_data)
            ) {
              allSnapshots.push(...snapshotEvents.snapshot_data);
            } else {
              console.warn(
                `Unexpected data format from source ${sourceInfo.blob_key}:`,
                snapshotEvents
              );
            }
          }
        } else {
          data = dataResponse.data;
        }

        allSnapshots.push(data);
      }
    }

    return [
      allSnapshots,
      [
        serverStartUrl,
        recording_duration,
        click_count,
        end_time,
        datetime,
        person_id,
        browser,
        osVar,
        osVersion,
        device_type,
        country,
        referrer,
        idVar,
        user_agent,
      ],
    ];
  } catch (error) {
    console.error("An error occurred:", error);
    return [[], ""];
  }
}
async function fetchPostHogSessionRecordings(
  projectId,
  apiKey,
  API_BASE_URL,
  API_BASE_URL2,
  dateFrom = null
) {
  let allRecordings = [];
  console.log(
    `fetchPostHogSessionRecordings called with dateFrom: ${dateFrom}`
  );

  // Use the provided dateFrom timestamp if available, otherwise default to 2 hours ago.
  let dateFromFilter;
  let firstRunWindowStart = null;
  let firstRunWindowEnd = null;

  if (dateFrom) {
    dateFromFilter = new Date(dateFrom).toISOString();
  } else {
    firstRunWindowEnd = new Date();
    firstRunWindowStart = new Date(
      firstRunWindowEnd.getTime() - 7 * 24 * 60 * 60 * 1000
    );
    dateFromFilter = firstRunWindowStart.toISOString();
  }
  console.log(
    `Using ISO8601 (UTC) date for PostHog API filter: ${dateFromFilter}`
  );
  let nextUrl = `${API_BASE_URL}/projects/${projectId}/session_recordings?date_from=${dateFromFilter}&limit=1000&filter_test_accounts=true`;
  let nextUrl2 = `${API_BASE_URL2}/projects/${projectId}/session_recordings?date_from=${dateFromFilter}&limit=1000&filter_test_accounts=true`;

  // FOR TESTING
  console.log("fnewkfnwkfnw ", process.env.NODE_ENV);
  if (process.env.NODE_ENV === "development") {
    // Sometimes this didnt fetch anything! Maybe too much data at once?
    nextUrl = `${API_BASE_URL}/projects/${projectId}/session_recordings?date_from=2025-12-04T01:30:00.000Z&date_to=2025-12-06T23:30:00.000Z&limit=10000&filter_test_accounts=true`;
    nextUrl2 = `${API_BASE_URL2}/projects/${projectId}/session_recordings?date_from=2025-12-04T01:30:00.000Z&date_to=2025-12-06T23:30:00.000Z&limit=10000&filter_test_accounts=true`;


    /*     &filter_test_accounts=true
 */  }

  try {
    while (nextUrl) {
      let response;
      try {
        console.log(
          "efwfwefwnextUrl ",
          API_BASE_URL,
          "nextUrl2 ",
          API_BASE_URL2
        );
        console.log(`Fetching session recordings from: ${nextUrl}`);
        response = await fetchWithRetry(nextUrl, {
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
        });
      } catch (error) {
        if (error.message.includes("Session terminated")) {
          throw error;
        }
        console.log("Error fetching session recordings: ", error?.message);
        try {
          response = await fetchWithRetry(nextUrl2, {
            headers: {
              Authorization: `Bearer ${apiKey}`,
              "Content-Type": "application/json",
            },
          });
          API_BASE_URL = API_BASE_URL2;
        } catch (fallbackError) {
          console.log(
            "Error fetching session recordings: ",
            fallbackError.message
          );
          throw new Error(
            `Error fetching session recordings: ${fallbackError.message}`
          );
        }
      }

      const data = response.data;

      if (data.results && Array.isArray(data.results)) {
        allRecordings.push(...data.results);
      } else {
        allRecordings.push(data);
      }

      nextUrl = data.next;
    }

    console.log(
      `Fetched a total of ${allRecordings.length} session recording metadata objects since ${dateFromFilter}.`
    );

    const validRecordings = allRecordings
      .filter((rec) => rec?.active_seconds && rec?.active_seconds > 5)
      .map((rec) => ({
        id: rec.id,
        start_time: rec.start_time,
      }));

    if (!dateFrom && firstRunWindowStart && firstRunWindowEnd) {
      const windowFiltered = validRecordings.filter((rec) => {
        if (!rec?.start_time) return false;
        const start = new Date(rec.start_time);
        return start >= firstRunWindowStart && start <= firstRunWindowEnd;
      });

      const newestFirst = [...windowFiltered].sort(
        (a, b) => new Date(b.start_time) - new Date(a.start_time)
      );

      const limitedWindow =
        newestFirst.length > 90 ? newestFirst.slice(0, 90) : newestFirst;

      console.log(
        `[SESSION-FETCH] Initial fetch limited to ${limitedWindow.length} session(s) from the last 7 days (raw count: ${windowFiltered.length}).`
      );

      return limitedWindow.sort(
        (a, b) => new Date(a.start_time) - new Date(b.start_time)
      );
    }

    return validRecordings;
  } catch (error) {
    /*  for (let i = 0; i < 10; i++) {
console.log("Probably Posthog api is down!")
    } */
    console.error(
      "An error occurred while fetching session recordings:",
      error.message
    );
    if (error.response) {
      console.error("Response Data:", error.response.data);
      console.error("Response Status:", error.response.status);
    }
    return [];
  }
}
function extractAllVisitedUrls(events) {
  const visitedUrls = [];

  events.forEach((event, index) => {
    const timestamp = new Date(event.timestamp);

    // Meta events (type 4) - actual page loads
    if (event.type === 4 && event.data?.href) {
      visitedUrls.push({
        url: event.data.href,
        timestamp: event.timestamp,
        time: timestamp.toISOString(),
        type: "META_EVENT",
        eventIndex: index,
        viewport: `${event.data.width}x${event.data.height}`,
      });
    }

    // Custom events (type 5) with $pageview tag - SPA navigation
    if (
      event.type === 5 &&
      event.data?.tag === "$pageview" &&
      event.data?.payload?.href
    ) {
      visitedUrls.push({
        url: event.data.payload.href,
        timestamp: event.timestamp,
        time: timestamp.toISOString(),
        type: "SPA_NAVIGATION",
        eventIndex: index,
        viewport: "N/A",
      });
    }
  });

  // Sort by timestamp
  visitedUrls.sort((a, b) => a.timestamp - b.timestamp);

  console.log(
    `\n📊 COMPLETE NAVIGATION HISTORY (${visitedUrls.length} events):`
  );
  console.log("=".repeat(60));

  visitedUrls.forEach((visit, index) => {
    const icon = visit.type === "META_EVENT" ? "🌐" : "🔄";
    console.log(`${icon} ${index + 1}. ${visit.time}`);
    console.log(`   ${visit.url}`);
    console.log(
      `   Type: ${visit.type}, Event: ${visit.eventIndex}, Viewport: ${visit.viewport}`
    );
    console.log("");
  });

  // Extract unique URLs in order of first visit
  const uniqueUrls = [];
  const seenUrls = new Set();

  visitedUrls.forEach((visit) => {
    if (!seenUrls.has(visit.url)) {
      uniqueUrls.push(visit.url);
      seenUrls.add(visit.url);
    }
  });

  // Create chronological URLs without consecutive duplicates
  const chronologicalUrlsDeduped = [];
  visitedUrls.forEach((visit, index) => {
    // Add the URL if it's the first one or different from the previous one
    if (index === 0 || visit.url !== visitedUrls[index - 1].url) {
      chronologicalUrlsDeduped.push(visit.url);
    }
  });

  console.log("\n📋 UNIQUE URLS VISITED (in order):");
  console.log("=".repeat(40));
  uniqueUrls.forEach((url, index) => {
    console.log(`${index + 1}. ${url}`);
  });

  console.log("\n📋 CHRONOLOGICAL URLS (no consecutive duplicates):");
  console.log("=".repeat(50));
  chronologicalUrlsDeduped.forEach((url, index) => {
    console.log(`${index + 1}. ${url}`);
  });

  return chronologicalUrlsDeduped;
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
      `🔑 API key rotation enabled with ${keysToTry.length} keys available.`
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
          `🔑 Using API key ${keyIndex + 1}/${keysToTry.length} (attempt ${
            totalAttempts + 1
          }/${maxTotalAttempts})`
        );
        currentAi = new GoogleGenAI({ apiKey: currentKey });
      }

      for (let attempt = 1; attempt <= retriesPerKey; attempt++) {
        totalAttempts++;

        if (totalAttempts > maxTotalAttempts) {
          console.error(
            `❌ Maximum total attempts (${maxTotalAttempts}) reached. All API keys and retries failed.`
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
              `⚠️ Missing candidatesTokenCount in response (key ${
                keyIndex + 1
              }, attempt ${attempt}). Retrying...`
            );
            throw new Error(
              "Missing candidatesTokenCount in response - treating as retryable error"
            );
          }

          console.log(
            `✅ Content generation successful on key ${
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
                `⏳ Quota/rate limit encountered. Waiting ${suggestedWaitMs} ms before retrying same key...`
              );
              await new Promise((resolve) =>
                setTimeout(resolve, suggestedWaitMs)
              );
              // retry same key (continue inner loop)
              continue;
            }
            console.log(
              `🔑 API key ${
                keyIndex + 1
              } hit quota limit (no short retry delay). Moving to next key.`
            );
            break; // Exit inner loop and move to the next key
          }

          if (!isRetryable) {
            console.error("❌ Non-retryable error encountered. Aborting.");
            throw error; // For non-retryable errors, fail fast
          }

          if (attempt >= retriesPerKey) {
            console.warn(
              `⚠️  Max retries (${retriesPerKey}) reached for key ${
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
    `❌ All API keys and retries failed. Total attempts: ${totalAttempts}.`
  );
  throw lastError || new Error("All retry attempts failed.");
}

async function createGeminiBatchJob({
  inlinedRequests,
  sessionId,
  analysisId,
  displayNameHint = "session",
}) {
  if (!Array.isArray(inlinedRequests) || inlinedRequests.length === 0) {
    throw new Error(
      "[BATCH] No inlined requests supplied for Gemini batch job creation."
    );
  }

  const destUri = buildGeminiBatchOutputUri(
    sessionId,
    analysisId,
    displayNameHint
  );
  const displayName = buildGeminiBatchDisplayName(
    sessionId,
    analysisId,
    displayNameHint
  );

  console.log(
    `[BATCH] Creating Gemini batch job (${displayNameHint}) for session ${sessionId} -> ${destUri}`
  );

  const batchJob = await ai.batches.create({
    model: Gemini,
    src: {
      inlinedRequests: inlinedRequests.map((request) => {
        const normalizedConfig = { ...(request?.config || {}) };
        if (request?.systemInstruction) {
            return {
                model: request?.model || Gemini,
                contents: request?.contents,
                config: normalizedConfig,
                systemInstruction: request.systemInstruction
              };
        }
        return {
          model: request?.model || Gemini,
          contents: request?.contents,
          config: normalizedConfig,
        };
      }),
    },
    config: {
      displayName,
      /* dest: {
        format: "jsonl",
        gcsUri: destUri,
      }, */
    },
  });

  console.log(
    `[BATCH] Gemini batch job created (name: ${batchJob?.name || "unknown"})`
  );

  return { batchJob, destUri };
}

/**
 * Extracts a frame from the middle of the clip video (during the freeze period)
 * This is where the click indicator is visible and the screen state is captured
 * @param {string} videoPath - Path to the clip file
 * @returns {Promise<string>} Path to the extracted frame image
 */
async function extractMiddleFrame(videoPath) {
  try {
    const { dir, name } = path.parse(videoPath);
    const frameOutputPath = path.join(dir, `${name}_middle_frame.png`);

    let ffmpegCommand = ffmpegStatic || "ffmpeg";

    // In production, ffmpeg might be at a specific path
    if (process.env.NODE_ENV === "production") {
      try {
        execSync("ffmpeg -version", { stdio: "ignore" });
      } catch (error) {
        const possiblePaths = [
          "/usr/bin/ffmpeg",
          "/usr/local/bin/ffmpeg",
          "/opt/render/project/.render/ffmpeg/bin/ffmpeg",
        ];

        for (const path of possiblePaths) {
          try {
            execSync(`${path} -version`, { stdio: "ignore" });
            ffmpegCommand = path;
            console.log(`Using ffmpeg at: ${path}`);
            break;
          } catch (e) {
            // Continue to next path
          }
        }
      }
    }

    const resolvedVideoPath = path.resolve(videoPath);
    const resolvedFramePath = path.resolve(frameOutputPath);

    // Prefer extracting a frame from the middle of the FREEZE window
    // (between green START beacon and red END beacon). This avoids
    // capturing the post-freeze red beacon and any pre-click pause.
    let ssTimeSeconds = 7.5; // fallback default
    try {
      const beacons = await findBeaconFrames_TrueStream(resolvedVideoPath);
      if (Array.isArray(beacons) && beacons.length > 0) {
        const greenBeacon = beacons.find((b) => b.color === "green");
        const redBeacon = beacons.find((b) => b.color === "red");
        if (greenBeacon && redBeacon) {
          const freezeStart = Math.max(0, greenBeacon.startTime || 0);
          const freezeEnd = Math.max(
            freezeStart,
            (typeof redBeacon.endTime === "number"
              ? redBeacon.endTime
              : redBeacon.startTime) || freezeStart
          );
          if (freezeEnd > freezeStart) {
            // Middle of freeze window, with a tiny safety margin away from the red beacon
            const mid = (freezeStart + freezeEnd) / 2;
            ssTimeSeconds = Math.max(
              freezeStart + 0.25,
              Math.min(mid, freezeEnd - 0.2)
            );
            console.log(
              `Extracting analysis frame within freeze window: green=${freezeStart.toFixed(
                3
              )}s, red=${freezeEnd.toFixed(3)}s, ss=${ssTimeSeconds.toFixed(
                3
              )}s`
            );
          }
        } else if (greenBeacon) {
          // If only green was detected, pick shortly after it
          const freezeStart = Math.max(0, greenBeacon.startTime || 0);
          ssTimeSeconds = freezeStart + 1.0;
          console.log(
            `Extracting analysis frame after green beacon only: ss=${ssTimeSeconds.toFixed(
              3
            )}s`
          );
        }
      }
    } catch (beaconErr) {
      console.warn(
        `[ANALYSIS-FRAME] Beacon-aware seek failed, falling back to default 7.5s: ${beaconErr.message}`
      );
    }

    const ffArgs = [
      "-ss",
      String(ssTimeSeconds),
      "-i",
      resolvedVideoPath,
      "-frames:v",
      "1", // Extract exactly 1 frame
      "-q:v",
      "2", // High quality PNG
      "-y", // Overwrite output file
      resolvedFramePath,
    ];

    console.log(`Extracting middle frame from: ${videoPath}`);
    console.log(`Output frame: ${frameOutputPath}`);

    const { status, error, stderr } = spawnSync(ffmpegCommand, ffArgs, {
      encoding: "utf-8",
    });

    if (status !== 0) {
      throw new Error(
        `ffmpeg frame extraction failed with code ${status}. ${
          (error && error.message) || stderr
        }`
      );
    }

    console.log(`Successfully extracted middle frame: ${frameOutputPath}`);
    return frameOutputPath;
  } catch (error) {
    console.error("Error extracting middle frame:", error);
    throw error;
  }
}

/**
 * Creates zoomed versions of the frame around the click position
 * @param {string} frameImagePath - Path to the original frame image
 * @param {number} clickX - X coordinate of the click
 * @param {number} clickY - Y coordinate of the click
 * @param {number} viewportWidth - Viewport width (default: 1920)
 * @param {number} viewportHeight - Viewport height (default: 1080)
 * @returns {Promise<{zoomedPath: string, superZoomedPath: string}>} Paths to zoomed images
 */
async function createZoomedFrames(
  frameImagePath,
  clickX,
  clickY,
  viewportWidth = 1920,
  viewportHeight = 1080
) {
  try {
    const { dir, name, ext } = path.parse(frameImagePath);
    const zoomedPath = path.join(dir, `${name}_zoomed${ext}`);
    const superZoomedPath = path.join(dir, `${name}_super_zoomed${ext}`);

    // Ensure click coordinates are within bounds
    const boundedClickX = Math.max(0, Math.min(viewportWidth, clickX));
    const boundedClickY = Math.max(0, Math.min(viewportHeight, clickY));

    console.log(
      `Creating zoomed frames around click position: ${boundedClickX}, ${boundedClickY}`
    );

    // Create 1.4x zoomed version
    const zoomFactor1 = 1.4;
    const zoomSize1 = {
      width: Math.round(viewportWidth / zoomFactor1),
      height: Math.round(viewportHeight / zoomFactor1),
    };
    const zoomLeft1 = Math.max(
      0,
      Math.min(
        viewportWidth - zoomSize1.width,
        boundedClickX - zoomSize1.width / 2
      )
    );
    const zoomTop1 = Math.max(
      0,
      Math.min(
        viewportHeight - zoomSize1.height,
        boundedClickY - zoomSize1.height / 2
      )
    );

    await sharp(frameImagePath)
      .extract({
        left: Math.round(zoomLeft1),
        top: Math.round(zoomTop1),
        width: zoomSize1.width,
        height: zoomSize1.height,
      })
      .resize(viewportWidth, viewportHeight, { fit: "fill" }) // Scale back up to original size for better visibility
      .png()
      .toFile(zoomedPath);

    // Create 3x zoomed version
    const zoomFactor2 = 3;
    const zoomSize2 = {
      width: Math.round(viewportWidth / zoomFactor2),
      height: Math.round(viewportHeight / zoomFactor2),
    };
    const zoomLeft2 = Math.max(
      0,
      Math.min(
        viewportWidth - zoomSize2.width,
        boundedClickX - zoomSize2.width / 2
      )
    );
    const zoomTop2 = Math.max(
      0,
      Math.min(
        viewportHeight - zoomSize2.height,
        boundedClickY - zoomSize2.height / 2
      )
    );

    await sharp(frameImagePath)
      .extract({
        left: Math.round(zoomLeft2),
        top: Math.round(zoomTop2),
        width: zoomSize2.width,
        height: zoomSize2.height,
      })
      .resize(viewportWidth, viewportHeight, { fit: "fill" }) // Scale back up to original size for better visibility
      .png()
      .toFile(superZoomedPath);

    console.log(`Created zoomed frame: ${zoomedPath}`);
    console.log(`Created super zoomed frame: ${superZoomedPath}`);

    return { zoomedPath, superZoomedPath };
  } catch (error) {
    console.error("Error creating zoomed frames:", error);
    throw error;
  }
}
function createClipsFromRrwebClicks(processedRecording, skips) {
  console.log("=== Rrweb-Only Click-Based Clip & Log Creation ===");

  if (!processedRecording || processedRecording.length === 0) {
    return { matches: [], logEntries: [] };
  }

  // 1. Get all raw rrweb clicks (type: 0 for click, type: 9 for touchend)
  const allRrwebClicks = processedRecording.filter(
    (ev) => ev.type === 3 && ev.data?.source === 2 && ev.data?.type === 2
  );

  if (allRrwebClicks.length === 0) {
    console.log(
      "No rrweb click or touch events (MouseInteractionType.Click or TouchEnd) found."
    );
    return { matches: [], logEntries: [] };
  }

  // 2. Deduplicate/group nearby clicks to represent a single user action
  const uniqueClickGroups = [];
  let currentGroup = [];

  if (allRrwebClicks.length > 0) {
    currentGroup.push(allRrwebClicks[0]);
    for (let i = 1; i < allRrwebClicks.length; i++) {
      const prevClick = currentGroup[currentGroup.length - 1];
      const currentClick = allRrwebClicks[i];

      uniqueClickGroups.push(currentGroup);
      currentGroup = [currentClick];
    }
    uniqueClickGroups.push(currentGroup);
  }

  console.log(
    `Found ${allRrwebClicks.length} raw rrweb clicks, grouped into ${uniqueClickGroups.length} unique interactions.`
  );

  const idNodeMap = buildRrwebNodeMap(processedRecording);
  if (!idNodeMap) {
    console.warn("Could not build node map for rrweb click log generation.");
    return { matches: [], logEntries: [] };
  }

  // 3. Select a representative click from each group. We'll just take the first one for simplicity.
  const representativeClicks = uniqueClickGroups.map((group, index) => {
    const click = group[0];
    click.eventIndex = index; // Assign an index for filenames
    return click;
  });

  // 4. Create clip data and log entries for each representative click
  const matches = [];
  const logEntries = [];
  const videoStartTimestamp = processedRecording[0].timestamp;
  const videoDurationSeconds =
    (processedRecording[processedRecording.length - 1].timestamp -
      videoStartTimestamp) /
    1000;

  for (let index = 0; index < representativeClicks.length; index++) {
    const click = representativeClicks[index];
    const videoTimeSeconds = (click.timestamp - videoStartTimestamp) / 1000;

    const clipDurationSeconds = CLIP_DURATION_MS / 1000;
    const previousClick = index > 0 ? representativeClicks[index - 1] : null;
    const gapToPreviousMs = previousClick
      ? Math.max(0, click.timestamp - previousClick.timestamp)
      : Number.POSITIVE_INFINITY;
    const preBufferMs = computeInterClickBufferMs(gapToPreviousMs);
    const preBufferSeconds = preBufferMs / 1000;

    let clipStartTimeSeconds = Math.max(0, videoTimeSeconds - preBufferSeconds);

    // FIX: Ensure start time doesn't overlap with the previous clip's end time
    if (matches.length > 0) {
      const previousClip = matches[matches.length - 1];
      if (clipStartTimeSeconds < previousClip.clipEndTimeSeconds) {
        if (DEBUG_CLICK) {
          console.log(
            `[CLICK-CLIP][OVERLAP] Adjusting start time for event #${
              click.eventIndex
            } from ${clipStartTimeSeconds.toFixed(
              3
            )}s to previous clip's end ${previousClip.clipEndTimeSeconds.toFixed(
              3
            )}s`
          );
        }
        clipStartTimeSeconds = previousClip.clipEndTimeSeconds;
      }
    }
    if (DEBUG_CLICK) {
      console.log(
        `[CLICK-CLIP][PLAN] click@${
          click.timestamp
        }ms rel=${videoTimeSeconds.toFixed(
          3
        )}s -> start=${clipStartTimeSeconds.toFixed(
          3
        )}s (preBuffer ${preBufferMs}ms)`
      );
    }

    // Calculate initial end time
    let clipEndTimeSeconds = Math.min(
      videoDurationSeconds,
      clipStartTimeSeconds + clipDurationSeconds
    );
    if (DEBUG_CLICK) {
      console.log(
        `[CLICK-CLIP][PLAN] initial end=${clipEndTimeSeconds.toFixed(
          3
        )}s (dur=${(clipEndTimeSeconds - clipStartTimeSeconds).toFixed(3)}s)`
      );
    }

    // 🎯 SHORTEN CLIP if next click would appear in it
    // Look ahead to see if any subsequent clicks fall within this clip's timeframe
    const clipEndMs = clipEndTimeSeconds * 1000 + videoStartTimestamp;
    const allClicks = processedRecording.filter(
      (ev) =>
        ev.type === 3 && ev.data && ev.data.source === 2 && ev.data.type === 2
    );
    const nextClickInClip = allClicks.find((otherClick) => {
      return (
        otherClick.timestamp > click.timestamp &&
        otherClick.timestamp < clipEndMs
      );
    });

    let finalClipEndTimeSeconds = clipEndTimeSeconds;
    if (nextClickInClip) {
      // The goal is to shorten the current clip so it doesn't show the next click.
      const gapToNextMs = Math.max(
        0,
        nextClickInClip.timestamp - click.timestamp
      );
      const postBufferMs = computeInterClickBufferMs(gapToNextMs);
      const adjustedClipEndMs = nextClickInClip.timestamp - postBufferMs; // Absolute timestamp

      // Convert the absolute end timestamp back to relative seconds
      finalClipEndTimeSeconds =
        (adjustedClipEndMs - videoStartTimestamp) / 1000;

      console.log(
        `  ?? SHORTENING CLIP: Next click at ${
          nextClickInClip.timestamp
        }ms, ending clip at relative time ${finalClipEndTimeSeconds.toFixed(
          3
        )}s (buffer ${postBufferMs}ms)`
      );
      if (DEBUG_CLICK) {
        console.log(
          `[CLICK-CLIP][ADJUST] nextClick=${
            nextClickInClip.timestamp
          }ms -> end=${finalClipEndTimeSeconds.toFixed(
            3
          )}s (buffer ${postBufferMs}ms)`
        );
      }
    }

    const actualClipDurationSeconds =
      finalClipEndTimeSeconds - clipStartTimeSeconds;

    if (actualClipDurationSeconds >= 0) {
      // Create the match object for clip processing
      matches.push({
        rrwebEvent: click,
        // Create a mock postHogEvent for compatibility with the clip analysis pipeline
        postHogEvent: {
          eventIndex: click.eventIndex,
          text: `unlabeled rrweb click`, // All clips_folder will need full analysis
        },
        videoTimeSeconds,
        clipStartTimeSeconds,
        clipEndTimeSeconds: finalClipEndTimeSeconds,
        clipDurationSeconds: actualClipDurationSeconds,
        needsAIAnalysis: true, // Always true for this method
        confidence: 1.0,
        matchReasons: ["rrweb-native-click"],
        elementMatchData: {
          nodeId: click.data.id,
          matchMethod: "rrweb-native",
        },
      });
      if (DEBUG_CLICK) {
        console.log(
          `[CLICK-CLIP][FINAL] idx=${
            click.eventIndex
          } start=${clipStartTimeSeconds.toFixed(
            3
          )}s end=${finalClipEndTimeSeconds.toFixed(
            3
          )}s dur=${actualClipDurationSeconds.toFixed(3)}s`
        );
      }

      // Create the corresponding log entry for the final text summary
      // Use the processed (already adjusted) rrweb timestamp for log timing.
      const adjustedAbsoluteTimestamp = click.timestamp;
      const adjustedRelativeTimeMs =
        adjustedAbsoluteTimestamp - videoStartTimestamp;
      const originalRelativeTimeMs = adjustedRelativeTimeMs;

      // DEBUG: Print timestamp calculations for the first few clicks
      if (click.eventIndex < 3) {
        console.log(`🔍 CLICK TIMESTAMP DEBUG for event #${click.eventIndex}:`);
        console.log(
          `   click.timestamp (adjusted): ${adjustedAbsoluteTimestamp}ms`
        );
        console.log(`   videoStartTimestamp: ${videoStartTimestamp}ms`);
        console.log(
          `   originalRelativeTimeMs (same as adjusted): ${originalRelativeTimeMs}ms`
        );
        console.log(`   adjustedRelativeTimeMs: ${adjustedRelativeTimeMs}ms`);
        console.log(
          `   adjustedSeconds: ${(adjustedRelativeTimeMs / 1000).toFixed(3)}s`
        );
        console.log(
          `   ────────────────────────────────────────────────────────`
        );
      }

      // Don't add log entries here - they will be added after AI analysis
      // Just track the timing info for later use
      // logEntries.push({
      //   adjustedMs: adjustedRelativeTimeMs,
      //   originalMs: originalRelativeTimeMs,
      //   originalAbsoluteMs: originalTimestamp,
      //   text: `unlabeled rrweb click`, // This will trigger full AI analysis
      //   isPreRecording: false,
      //   eventIndex: click.eventIndex, // CRITICAL: Use the same index as the match
      //   originalEvent: click,q
      // });
    }
  }
  return { matches, logEntries };
}

function describeRrwebNode(node, idNodeMap) {
  if (!node) return "an element";

  // Traverse up to find more meaningful context if the clicked node is generic (like a DIV/SPAN)
  let currentNode = node;
  for (let i = 0; i < 3; i++) {
    if (!currentNode) break;
    const tagName = currentNode.tagName?.toLowerCase() || "element";
    const attrs = currentNode.attributes || {};
    let text = currentNode.textContent?.trim() || "";
    if (text.length > 50) text = text.substring(0, 47) + "...";

    // Prioritize descriptive attributes
    if (attrs["aria-label"])
      return `${tagName} with label "${attrs["aria-label"]}"`;
    if (attrs.placeholder)
      return `${tagName} with placeholder "${attrs.placeholder}"`;
    if (text) return `${tagName} with text "${text}"`;
    if (attrs.name) return `${tagName} named "${attrs.name}"`;
    if (attrs.id) return `element with id "#${attrs.id}"`;
    if (attrs.class)
      return `a ${tagName} with class ".${attrs.class.split(" ")[0]}"`;

    // If current node is generic, move to parent
    if (["div", "span"].includes(tagName) && currentNode.parentId) {
      currentNode = idNodeMap.get(currentNode.parentId);
    } else {
      break; // Stop if we hit a non-generic element
    }
  }

  // Fallback to the original node's tag if traversal yields nothing
  return `a ${node.tagName?.toLowerCase() || "element"} element`;
}
async function postProcessClip(inputPath, outputPath, beacons) {
  console.log(
    `[POST-PROCESS] Trimming all pause frames from: ${path.basename(inputPath)}`
  );

  // --- DEBUG LOGGING ---
  console.log(
    `[POST-PROCESS-DEBUG] Received ${
      beacons?.length || 0
    } beacons for processing:`
  );
  console.log(JSON.stringify(beacons, null, 2));
  // --- END DEBUG LOGGING ---

  if (!beacons || beacons.length === 0) {
    console.warn(`[POST-PROCESS] No beacons found, copying original file`);
    await fs.copyFile(inputPath, outputPath);
    return true;
  }

  const ffmpegCommand = ffmpegStatic || "ffmpeg";
  const tempDir = os.tmpdir();
  const tempFrameDir = path.join(
    tempDir,
    `trim_frames_${crypto.randomBytes(6).toString("hex")}`
  );
  const concatListPath = path.join(
    tempDir,
    `trim_list_${crypto.randomBytes(6).toString("hex")}.txt`
  );

  try {
    // 1. Get original frame rate FIRST
    const ffprobeCommand = ffprobeStatic?.path || "ffprobe";
    const probeResult = spawnSync(
      ffprobeCommand,
      [
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        inputPath,
      ],
      { encoding: "utf-8" }
    );

    if (probeResult.status !== 0 || !probeResult.stdout) {
      throw new Error(
        `ffprobe failed to get frame rate. Stderr: ${probeResult.stderr}`
      );
    }
    const frameRate = Math.round(eval(probeResult.stdout));
    console.log(`[POST-PROCESS-DEBUG] Detected frame rate: ${frameRate} FPS`);

    // Extract all frames
    await fs.mkdir(tempFrameDir, { recursive: true });
    const extractArgs = [
      "-i",
      inputPath,
      path.join(tempFrameDir, "frame_%04d.png"),
    ];
    const extractResult = spawnSync(ffmpegCommand, extractArgs);
    if (extractResult.status !== 0) {
      throw new Error(
        `Failed to extract frames for trimming. Stderr: ${extractResult.stderr}`
      );
    }

    // Get all frames and create a list of frames to keep (excluding beacon frames)
    const allFrames = (await fs.readdir(tempFrameDir)).sort();
    const framesToKeep = [];
    const framesToExclude = new Set();

    const greenBeacon = beacons.find((b) => b.color === "green");
    const redBeacon = beacons.find((b) => b.color === "red");

    // --- DEBUG LOGGING ---
    console.log(`[POST-PROCESS-DEBUG] Green beacon found: ${!!greenBeacon}`);
    if (greenBeacon) console.log(JSON.stringify(greenBeacon));
    console.log(`[POST-PROCESS-DEBUG] Red beacon found: ${!!redBeacon}`);
    if (redBeacon) console.log(JSON.stringify(redBeacon));
    // --- END DEBUG LOGGING ---

    // Mark pre-pause (blue) and post-pause (yellow) beacon frames for exclusion
    for (const beacon of beacons) {
      if (beacon.color === "blue" || beacon.color === "yellow") {
        const rawStart = Math.floor(beacon.startTime * frameRate);
        const rawEnd = Math.ceil(beacon.endTime * frameRate);
        const startFrame = Math.max(
          0,
          rawStart - BEACON_EXCLUSION_MARGIN_FRAMES
        );
        const endFrame = rawEnd + BEACON_EXCLUSION_MARGIN_FRAMES;
        for (let i = startFrame; i <= endFrame; i++) {
          framesToExclude.add(i);
        }
        console.log(
          `[POST-PROCESS-DEBUG] Marking yellow/blue frames ${startFrame}-${endFrame} for exclusion (from ${beacon.startTime.toFixed(
            3
          )}s-${beacon.endTime.toFixed(
            3
          )}s, margin=${BEACON_EXCLUSION_MARGIN_FRAMES}f).`
        );
      }
    }

    // Mark the main freeze-frame segment (from green start to red end) for exclusion
    if (greenBeacon && redBeacon) {
      console.log(
        "[POST-PROCESS-DEBUG] Found both green and red beacons. Proceeding with full sequence removal."
      );
      const freezeStartFrame = Math.max(
        0,
        Math.floor(greenBeacon.startTime * frameRate) -
          BEACON_EXCLUSION_MARGIN_FRAMES
      );
      const freezeEndFrame =
        Math.ceil(redBeacon.endTime * frameRate) +
        BEACON_EXCLUSION_MARGIN_FRAMES;
      for (let i = freezeStartFrame; i <= freezeEndFrame; i++) {
        framesToExclude.add(i);
      }
      console.log(
        `[POST-PROCESS-DEBUG] Marking freeze-frame sequence from ${freezeStartFrame} to ${freezeEndFrame} for exclusion (from ${greenBeacon.startTime.toFixed(
          3
        )}s to ${redBeacon.endTime.toFixed(3)}s).`
      );
    } else if (framesToExclude.size > 0) {
      // This case handles when we only have yellow/blue beacons (e.g., a noclick clip).
      console.log(
        `[POST-PROCESS-DEBUG] Only non-click beacons (yellow/blue) found. Proceeding to remove them.`
      );
    } else {
      console.warn(
        `[POST-PROCESS-DEBUG] Could not find any beacons to remove. The clip will NOT be processed to avoid creating a confusing result. The original clip with visible beacons will be used.`
      );
      // If we don't have the full sequence, we should not modify the clip at all.
      // Copy the original to the output path and return.
      await fs.copyFile(inputPath, outputPath);
      return true; // End processing for this clip.
    }

    for (let i = 0; i < allFrames.length; i++) {
      if (!framesToExclude.has(i)) {
        framesToKeep.push(
          `file '${path.join(tempFrameDir, allFrames[i]).replace(/\\/g, "/")}'`
        );
      }
    }

    if (framesToKeep.length === 0) {
      throw new Error("No frames left after removing all beacon sections.");
    }

    // --- DEBUG LOGGING ---
    console.log(
      `[POST-PROCESS-DEBUG] Total frames: ${allFrames.length}. Frames to keep: ${framesToKeep.length}. Frames to exclude: ${framesToExclude.size}.`
    );
    // --- END DEBUG LOGGING ---

    console.log(
      `[POST-PROCESS] Keeping ${framesToKeep.length} frames out of ${allFrames.length} total frames`
    );

    // Write the concat list
    await fs.writeFile(concatListPath, framesToKeep.join("\n"));

    if (probeResult.status !== 0 || !probeResult.stdout) {
      throw new Error(
        `ffprobe failed to get frame rate. Stderr: ${probeResult.stderr}`
      );
    }

    // Rebuild clip from trimmed frames
    const rebuildArgs = [
      "-r",
      frameRate.toString(),
      "-f",
      "concat",
      "-safe",
      "0",
      "-i",
      concatListPath,
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-y", // Overwrite output file
      outputPath,
    ];

    console.log(
      `[POST-PROCESS] Rebuilding clip with frame rate: ${frameRate} FPS`
    );
    const rebuildResult = spawnSync(ffmpegCommand, rebuildArgs);
    if (rebuildResult.status !== 0) {
      throw new Error(
        `Failed to rebuild trimmed video. Stderr: ${rebuildResult.stderr}`
      );
    }

    // Audio is added by the caller, so we don't need to do it here.

    console.log(
      `[POST-PROCESS] Successfully created trimmed clip: ${path.basename(
        outputPath
      )}`
    );
    return true;
  } catch (error) {
    console.error(
      `[POST-PROCESS] Error during clip trimming: ${error.message}`
    );
    return false;
  } finally {
    // Cleanup
    try {
      await fs.rm(tempFrameDir, { recursive: true, force: true });
      await fs.unlink(concatListPath);
    } catch (cleanupError) {
      console.warn(`[POST-PROCESS] Cleanup warning: ${cleanupError.message}`);
    }
  }
}
/**
 * Trim trailing duplicate frames that are exactly identical to the last frame.
 * - Extracts frames to a temp folder
 * - Compares from the end backwards using raw pixel buffers via sharp
 * - Rebuilds the clip excluding the duplicates, keeping only the final frame
 *
 * Returns an object with { removedFrames } on success.
 */

async function trimTrailingDuplicateFrames(inputPath) {
  console.log(
    `\n[TAIL-TRIM] START: Checking for trailing duplicates in: ${path.basename(
      inputPath
    )}`
  );
  const ffmpegCommand = ffmpegStatic || "ffmpeg";
  const ffprobeCommand = ffprobeStatic?.path || "ffprobe";
  const tempDir = os.tmpdir();
  const tempFrameDir = path.join(
    tempDir,
    `tail_frames_${crypto.randomBytes(6).toString("hex")}`
  );
  const concatListPath = path.join(
    tempDir,
    `tail_list_${crypto.randomBytes(6).toString("hex")}.txt`
  );

  try {
    // Probe frame rate
    console.log("[TAIL-TRIM] Probing clip frame rate...");
    const probeResult = spawnSync(
      ffprobeCommand,
      [
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        inputPath,
      ],
      { encoding: "utf-8" }
    );
    if (probeResult.status !== 0 || !probeResult.stdout) {
      console.warn(
        `[TAIL-TRIM] ffprobe failed to get frame rate. Skipping: ${probeResult.stderr}`
      );
      return { removedFrames: 0 };
    }
    const frameRate = Math.round(eval(probeResult.stdout));
    console.log(`[TAIL-TRIM]   -> Detected frame rate: ${frameRate} FPS`);

    // Extract frames
    console.log(
      `[TAIL-TRIM] Extracting all frames to temp directory: ${tempFrameDir}`
    );
    await fs.mkdir(tempFrameDir, { recursive: true });
    const extractArgs = [
      "-i",
      inputPath,
      path.join(tempFrameDir, "frame_%08d.png"),
    ];
    const extractResult = spawnSync(ffmpegCommand, extractArgs);
    if (extractResult.status !== 0) {
      console.warn(
        `[TAIL-TRIM] Failed to extract frames. Skipping. ${extractResult.stderr}`
      );
      return { removedFrames: 0 };
    }

    const frames = (await fs.readdir(tempFrameDir)).sort();
    console.log(`[TAIL-TRIM]   -> Extracted ${frames.length} frames.`);
    if (frames.length < 2) {
      console.log(
        `[TAIL-TRIM] Skipping: only ${frames.length} frame(s) detected.`
      );
      return { removedFrames: 0 };
    }

    // Load last frame as raw pixels
    console.log("[TAIL-TRIM] Loading last frame for comparison...");
    const lastPath = path.join(tempFrameDir, frames[frames.length - 1]);
    const lastImg = sharp(lastPath).ensureAlpha().raw();
    const { data: lastData, info: lastInfo } = await lastImg.toBuffer({
      resolveWithObject: true,
    });
    console.log(
      `[TAIL-TRIM]   -> Last frame loaded (${lastInfo.width}x${lastInfo.height})`
    );

    // Configurable similarity threshold (higher PSNR means more similar)
    const PSNR_THRESHOLD_DB = Number(process.env.TAIL_TRIM_PSNR_DB || 52);
    console.log(
      `[TAIL-TRIM] Using PSNR similarity threshold of ${PSNR_THRESHOLD_DB} dB`
    );

    let duplicates = 0;
    console.log("[TAIL-TRIM] Starting backward frame comparison...");
    for (let idx = frames.length - 2; idx >= 0; idx--) {
      const p = path.join(tempFrameDir, frames[idx]);
      try {
        const img = sharp(p).ensureAlpha().raw();
        const { data, info } = await img.toBuffer({ resolveWithObject: true });
        if (
          info.width !== lastInfo.width ||
          info.height !== lastInfo.height ||
          info.channels !== lastInfo.channels
        ) {
          console.log(
            `[TAIL-TRIM] STOP: Frame #${idx} has different dimensions.`
          );
          break; // Different geometry -> stop
        }
        if (data.length !== lastData.length) {
          console.log(
            `[TAIL-TRIM] STOP: Frame #${idx} has different data length.`
          );
          break;
        }
        // First, quick exact match check
        let exactEqual = true;
        for (let i = 0; i < data.length; i++) {
          if (data[i] !== lastData[i]) {
            exactEqual = false;
            break;
          }
        }
        if (exactEqual) {
          duplicates++;
          console.log(
            `[TAIL-TRIM]      - RESULT: Exact match. (Total duplicates so far: ${duplicates})`
          );
        } else {
          // If not exactly equal, compute PSNR and compare to threshold
          let sse = 0;
          for (let i = 0; i < data.length; i++) {
            const d = data[i] - lastData[i];
            sse += d * d;
          }
          const n = data.length; // per-byte across channels
          if (sse === 0) {
            duplicates++;
            console.log(
              `[TAIL-TRIM]      - RESULT: Exact match (SSE=0). (Total duplicates so far: ${duplicates})`
            );
            continue;
          }
          const mse = sse / n;
          const psnr = 20 * Math.log10(255 / Math.sqrt(mse));
          if (psnr >= PSNR_THRESHOLD_DB) {
            if (duplicates === 0) {
              console.log(
                `[TAIL-TRIM] Treating near-identical frame as duplicate (PSNR=${psnr.toFixed(
                  2
                )} dB, threshold=${PSNR_THRESHOLD_DB} dB).`
              );
            }
            duplicates++;
            /*    console.log(
              `[TAIL-TRIM]      - RESULT: Near-identical match (PSNR=${psnr.toFixed(
                2
              )} dB). (Total duplicates so far: ${duplicates})`
            ); */
          } else {
            // Stop when we reach a frame that is not similar enough
            console.log(
              `[TAIL-TRIM] STOP: Frame #${idx} is different (PSNR=${psnr.toFixed(
                2
              )} dB < ${PSNR_THRESHOLD_DB} dB).`
            );
            break;
          }
        }
      } catch (e) {
        console.warn(
          `[TAIL-TRIM] Error comparing frame #${idx}, stopping scan: ${
            e?.message || e
          }`
        );
        break; // Any read/parse error -> stop
      }
    }

    if (duplicates <= 0) {
      console.log(`[TAIL-TRIM] No trailing duplicates found.`);
      return { removedFrames: 0 };
    }
    console.log(
      `[TAIL-TRIM] Found a total of ${duplicates} trailing duplicate frames.`
    );

    // Cap the amount of trimming to max seconds
    const MAX_SECONDS = Number(process.env.TAIL_TRIM_MAX_SECONDS || 2);
    const maxFramesToRemove = Math.max(0, Math.floor(frameRate * MAX_SECONDS));
    const duplicatesToRemove = Math.min(duplicates, maxFramesToRemove);
    if (duplicatesToRemove < duplicates) {
      const wouldRemoveSec = (duplicates / frameRate).toFixed(3);
      const willRemoveSec = (duplicatesToRemove / frameRate).toFixed(3);
      console.log(
        `[TAIL-TRIM] Capping removal to ${willRemoveSec}s (wanted ${wouldRemoveSec}s, max ${MAX_SECONDS}s). Will remove ${duplicatesToRemove} frames.`
      );
    } else {
      console.log(`[TAIL-TRIM] Will remove ${duplicatesToRemove} frames.`);
    }

    // Build concat list excluding only the last duplicatesToRemove frames before the final frame
    // Keep frames [0 .. N-2-duplicatesToRemove], and always keep frame N-1
    const lines = [];
    const runStartIndex = frames.length - 1 - duplicatesToRemove;
    console.log(
      `[TAIL-TRIM] Building concat list. Keeping frames 0 to ${
        runStartIndex - 1
      }, then the last frame.`
    );
    for (let i = 0; i < frames.length - 1; i++) {
      if (i < runStartIndex) {
        lines.push(
          `file '${path.join(tempFrameDir, frames[i]).replace(/\\\\/g, "/")}'`
        );
      }
    }
    // Always keep the last frame
    lines.push(
      `file '${path
        .join(tempFrameDir, frames[frames.length - 1])
        .replace(/\\\\/g, "/")}'`
    );
    await fs.writeFile(concatListPath, lines.join("\n"));
    console.log(`[TAIL-TRIM]   -> Concat list written to ${concatListPath}`);

    // Rebuild into a temp file then replace input
    const { dir, name, ext } = path.parse(path.resolve(inputPath));
    const tempOut = path.join(dir, `${name}.tailtrim${ext || ".mp4"}`);
    console.log(`[TAIL-TRIM] Rebuilding clip from ${lines.length} frames...`);
    const rebuildArgs = [
      "-r",
      frameRate.toString(),
      "-f",
      "concat",
      "-safe",
      "0",
      "-i",
      concatListPath,
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-y",
      tempOut,
    ];
    const rebuild = spawnSync(ffmpegCommand, rebuildArgs);
    if (rebuild.status !== 0) {
      console.warn(
        `[TAIL-TRIM] Failed to rebuild trimmed video. Skipping. ${rebuild.stderr}`
      );
      return { removedFrames: 0 };
    }
    console.log(`[TAIL-TRIM]   -> Rebuilt clip at ${tempOut}`);

    // Replace original
    await fs.rename(tempOut, inputPath);
    console.log(
      `[TAIL-TRIM] SUCCESS: Removed ${duplicatesToRemove} trailing duplicate frame(s) (${(
        duplicatesToRemove / frameRate
      ).toFixed(3)}s) from ${path.basename(inputPath)}`
    );
    return { removedFrames: duplicatesToRemove };
  } catch (e) {
    console.warn(
      `[TAIL-TRIM] ERROR: Unhandled exception while trimming: ${
        e?.message || e
      }`
    );
    return { removedFrames: 0 };
  } finally {
    console.log(`[TAIL-TRIM] Cleaning up temporary files...`);
    try {
      await fs.rm(tempFrameDir, { recursive: true, force: true });
    } catch (_) {}
    try {
      await fs.unlink(concatListPath);
    } catch (_) {}
    console.log(
      `[TAIL-TRIM] END: Finished processing ${path.basename(inputPath)}\n`
    );
  }
}

async function analyzeClip(
  clipTask,
  refinedLogEntries,
  apiKeys,
  ai,
  processedRecording,
  sessionId
) {
  let {
    clipPath: originalClipPath,
    event,
    rrwebEvent,
    was_second_try,
    coordinateData,
    fileStartSeconds,
    fileEndSeconds,
  } = clipTask;

  // Skip frame-based analysis for non-click coverage clips_folder.
  // These are handled later by the dedicated non-click analysis flow using the raw .mp4 only.
  if (clipTask && clipTask.isNonClickSegment) {
    if (typeof DEBUG_NOCLICK !== "undefined" && DEBUG_NOCLICK) {
      console.log(
        `🔎 [NOCLICK DEBUG] Skipping analyzeClip for non-click segment: ${clipTask.clipPath}`
      );
    }
    return null;
  }

  let clipPath = originalClipPath;
  let frameImagePath;
  let isTempFrame = false;
  // Default: unknown until verified
  let isClickVerified = null;
  let isPointInClaimed = null;
  let verificationCoords = null;

  try {
    const clipsDir = path.dirname(clipPath);
    const { name: clipName } = path.parse(clipPath);
    const savedFramePath = path.join(
      clipsDir,
      `${clipName}_analysis_frame.png`
    );

    if (
      process.env.ONLY_ANALYSIS === "true" ||
      process.env.ONLY_ZOOMED === "true"
    ) {
      // Analysis-only: For non-click clips_folder, do not require _processed.mp4; for click clips_folder prefer processed if available
      frameImagePath = savedFramePath;
      if (clipTask?.isNonClickSegment) {
        if (DEBUG_NOCLICK) {
          console.log(
            `🔎 [NOCLICK DEBUG] ONLY_ANALYSIS: using original noclick clip: ${clipPath}`
          );
        }
      } else {
        const processedCandidate = clipPath.replace(".mp4", "_processed.mp4");
        try {
          await fs.access(processedCandidate);
          clipPath = processedCandidate;
        } catch {}
      }
      await fs.access(clipPath);
      await fs.access(frameImagePath);
    } else {
      frameImagePath = await extractMiddleFrame(clipPath);
      isTempFrame = true;
      await fs.copyFile(frameImagePath, savedFramePath);
      console.log(`     Saved analysis frame: ${savedFramePath}`);

      if (process.env.ONLY_ZOOMED !== "true") {
        // The `_processed.mp4` clip is already generated correctly in the initial `cutVideoBasedOnBeacons` step.
        // The previous logic here re-processed the clip, which was both redundant and buggy (it failed to detect
        // beacons at the edge of the trimmed clip, resulting in an incorrect final duration).
        // The correct approach is to simply use the already-existing processed clip for the AI analysis.
        const processedClipPath = clipPath.replace(".mp4", "_processed.mp4");
        try {
          // Verify the processed clip exists before using it.
          await fs.access(processedClipPath);
          console.log(
            `[analyzeClip] Using pre-existing processed clip: ${path.basename(
              processedClipPath
            )}`
          );
          clipPath = processedClipPath; // Switch clipPath to the processed version for AI analysis
        } catch (err) {
          console.warn(
            `[analyzeClip] Pre-existing processed clip not found at ${processedClipPath}. The AI will receive the unprocessed clip.`
          );
          // If the processed clip is missing, we fall back to the original `clipPath` (the unprocessed clip).
        }
      }
    }

    const isAlreadyLabeled =
      event.text &&
      event.text.includes("Clicked") &&
      !event.text.includes("something (find in the recording)");

    // Try to read verification result saved alongside the clip
    let highlightVisibility = null;
    try {
      const clipsDir = path.dirname(originalClipPath);
      const clipBaseName = path.basename(originalClipPath, ".mp4");
      const match = clipBaseName.match(/^click_(\d+)_/);
      if (!match) {
        throw new Error(
          `Could not parse click index from filename: ${clipBaseName}`
        );
      }
      const clickIndex = match[1];
      const verificationPath = path.join(
        clipsDir,
        `click_${clickIndex}_verification.json`
      );

      const verificationRaw = await fs.readFile(verificationPath, "utf-8");
      const verificationJson = JSON.parse(verificationRaw);
      if (verificationJson.coords) {
        verificationCoords = verificationJson.coords;
      }
      isClickVerified = true;
      isPointInClaimed =
        verificationJson && verificationJson?.pointInClaimed === true
          ? true
          : false;
      // Extract highlight visibility if present
      try {
        highlightVisibility = verificationJson?.highlightVisibility || null;
      } catch (_) {
        highlightVisibility = null;
      }
      console.log(
        `     -> Click verification loaded: match=${isClickVerified}, pointInClaimed=${isPointInClaimed}`
      );
    } catch (e) {
      console.log(
        `     -> Click verification not found or unreadable, proceeding without it (${e.message})`
      );
      isClickVerified = null;
      highlightVisibility = null;
    }

    let clickPositionInfo = null;
    const frameBuffer = await fs.readFile(frameImagePath);
    const frameBase64 = frameBuffer.toString("base64");
    const framePart = {
      inlineData: { mimeType: "image/png", data: frameBase64 },
    };

    // Create zoomed versions if we have click coordinates
    let zoomedPart = null;
    let superZoomedPart = null;
    let zoomedPaths = null;

    if (
      isPointInClaimed === true &&
      rrwebEvent &&
      rrwebEvent.data?.x &&
      rrwebEvent.data?.y
    ) {
      try {
        // 🎯 NEW: Prioritize live coordinates from the browser render
        let transformedClickX;
        let transformedClickY;
        const viewportWidth = 1920;
        const viewportHeight = 1080;

        if (
          verificationCoords &&
          verificationCoords.x &&
          verificationCoords.y
        ) {
          console.log(
            `     -> ✅ Using live coordinates from verification file for zoom.`
          );
          transformedClickX = verificationCoords.x;
          transformedClickY = verificationCoords.y;
        } else if (coordinateData && coordinateData.finalCoordinates) {
          console.log(
            `     -> ✅ Using live coordinates from browser for zoom.`
          );
          transformedClickX = coordinateData.finalCoordinates.x;
          transformedClickY = coordinateData.finalCoordinates.y;
        } else {
          // Fallback to the server-side calculation if live data is missing
          console.warn(
            `     -> ⚠️ Live coordinates not available. Falling back to server-side calculation for zoom.`
          );

          // --- START DEBUG LOGGING ---
          console.log(
            `[ZOOM-DEBUG] Fallback calculation for eventIndex: ${event.eventIndex}`
          );
          console.log(
            `[ZOOM-DEBUG] rrwebEvent timestamp: ${rrwebEvent.timestamp}`
          );
          console.log(
            `[ZOOM-DEBUG] rrwebEvent coords: x=${rrwebEvent.data?.x}, y=${rrwebEvent.data?.y}`
          );
          // --- END DEBUG LOGGING ---

          let originalWidth = 1920;
          let originalHeight = 1080;
          // Find the most recent viewport event before or at the click timestamp
          let clickTimeResolution = null;
          for (let i = 0; i < processedRecording.length; i++) {
            const event = processedRecording[i];
            if (
              (event.type === 4 && // Meta event
                event.data &&
                event.data.width &&
                event.data.height) ||
              (event.type === 3 && // Incremental snapshot, viewport resize
                event.data &&
                event.data.source === 4 &&
                event.data.width &&
                event.data.height)
            ) {
              if (event.timestamp <= rrwebEvent.timestamp) {
                clickTimeResolution = {
                  width: event.data.width,
                  height: event.data.height,
                };
              } else {
                break; // Events are chronological, no need to check further
              }
            }
          }

          if (clickTimeResolution) {
            originalWidth = clickTimeResolution.width;
            originalHeight = clickTimeResolution.height;
          }

          // --- START DEBUG LOGGING ---
          console.log(
            `[ZOOM-DEBUG] Determined content resolution at click time: ${originalWidth}x${originalHeight}`
          );
          // --- END DEBUG LOGGING ---

          const scale = Math.min(
            viewportWidth / originalWidth,
            viewportHeight / originalHeight
          );
          const offsetX = (viewportWidth - originalWidth * scale) / 2;
          const offsetY = (viewportHeight - originalHeight * scale) / 2;

          // --- START DEBUG LOGGING ---
          console.log(`[ZOOM-DEBUG] Calculated scale: ${scale}`);
          console.log(
            `[ZOOM-DEBUG] Calculated offset: x=${offsetX}, y=${offsetY}`
          );
          // --- END DEBUG LOGGING ---

          transformedClickX = offsetX + rrwebEvent.data.x * scale;
          transformedClickY = offsetY + rrwebEvent.data.y * scale;

          // --- START DEBUG LOGGING ---
          console.log(
            `[ZOOM-DEBUG] Final transformed coords: x=${transformedClickX}, y=${transformedClickY}`
          );
          // --- END DEBUG LOGGING ---
        }

        console.log(
          `     -> Using coordinates for zoom: ${transformedClickX.toFixed(
            1
          )}, ${transformedClickY.toFixed(1)}`
        );

        if (
          process.env.ONLY_ANALYSIS === "true" ||
          process.env.ONLY_ZOOMED === "true"
        ) {
          const { dir, name, ext } = path.parse(frameImagePath);
          const middleFrameName = name.replace(
            "_analysis_frame",
            "_middle_frame"
          );
          zoomedPaths = {
            zoomedPath: path.join(dir, `${middleFrameName}_zoomed${ext}`),
            superZoomedPath: path.join(
              dir,
              `${middleFrameName}_super_zoomed${ext}`
            ),
          };
          console.log(`[ONLY ANALYSIS] Using existing zoomed frames.`);
        } else {
          zoomedPaths = await createZoomedFrames(
            frameImagePath,
            transformedClickX,
            transformedClickY,
            viewportWidth,
            viewportHeight
          );
        }

        // Create base64 versions for the AI prompt
        const zoomedBuffer = await fs.readFile(zoomedPaths.zoomedPath);
        const superZoomedBuffer = await fs.readFile(
          zoomedPaths.superZoomedPath
        );

        zoomedPart = {
          inlineData: {
            mimeType: "image/png",
            data: zoomedBuffer.toString("base64"),
          },
        };

        superZoomedPart = {
          inlineData: {
            mimeType: "image/png",
            data: superZoomedBuffer.toString("base64"),
          },
        };

        console.log(`     -> Created zoomed versions for click analysis`);
      } catch (zoomError) {
        console.warn(
          `     -> Failed to create zoomed versions: ${zoomError.message}`
        );
      }
    }

    // Build rrweb event context information
    let rrwebContextInfo = "";
    if (rrwebEvent) {
      // Build node map to get element descriptions
      const idNodeMap = buildRrwebNodeMap(processedRecording);
      const clickedNode = idNodeMap ? idNodeMap.get(rrwebEvent.data?.id) : null;

      // Get human-readable description of the clicked element
      let elementDescription = "unknown element";
      if (clickedNode && idNodeMap) {
        elementDescription = describeRrwebNode(clickedNode, idNodeMap);
      }

      // Get real-time text content at the time of click
      let realTimeText = "";
      if (rrwebEvent.data?.id) {
        const textContent = getTextContentAtTime(
          rrwebEvent.data.id,
          rrwebEvent.timestamp,
          processedRecording
        );
        realTimeText = textContent ? `"${textContent.trim()}"` : "no text";
      }

      rrwebContextInfo = `\n\nAdditional context - The actual rrweb click event data for this interaction:
- Target Node ID: ${rrwebEvent.data?.id || "unknown"}
- X Position: ${rrwebEvent.data?.x || "unknown"}px
- Y Position: ${rrwebEvent.data?.y || "unknown"}px
- Element Description: ${elementDescription}
- Text Content at Click Time: ${realTimeText}`;
    }

    console.log("rrwebContextInforrwebContextInfo ", rrwebContextInfo);

    let promptImages = [framePart];
    let imageDescription = "You are provided with 1 screenshot.";

    if (zoomedPart && superZoomedPart) {
      promptImages = [superZoomedPart, zoomedPart];
      imageDescription =
        "You are provided with 2 screenshots: 1) A super zoomed version, 2) A zoomed version.";
    } else {
      console.log("No zoomed parts found");
    }

    let borderDetection = null;

    console.log("Newkgnwekgwnekgnw ", isPointInClaimed, highlightVisibility);

    if (isPointInClaimed === true) {
      const circleInstruction = `${imageDescription}

Describe the location of the pink dot on the screen. This is clearly an artificial overlay, not part of the original UI.

If you see a red line, ignore it!`;
      const groupReviewCompletion = await generateContentWithRetry(
        ai,
        {
          model: Gemini,
          config: {
            responseMimeType: "text/plain" /* thinkingConfig: {
              thinkingBudget: 32768,  // Max value for deepest reasoning
              // Alternative: thinkingBudget: -1 for dynamic auto-max
            },  */,
            temperature: 0.2,
            top_p: 0.95,
          },
          systemInstruction: circleInstruction,
          contents: createUserContent([...promptImages, circleInstruction]),
        },
        3,
        apiKeys,
        {
          sessionId,
          promptLabel: "click-position",
          promptContext: {
            eventIndex: event?.eventIndex,
            clipPath,
            clipType: "click",
          },
        }
      );
      clickPositionInfo =
        groupReviewCompletion?.text?.trim() || "unknown position";

      console.log(
        "JFbejfwebfjwebfwe",
        clickPositionInfo,
        "\n\n",
        groupReviewCompletion
      );

      console.log(`     -> Extracted click position: "${clickPositionInfo}"`);
    } else if (
      highlightVisibility &&
      highlightVisibility.visibleInViewport === true
    ) {
      // Containment failed: ask model to detect whether a border is visible and what element it borders
      const borderInstruction = `${imageDescription}
What element has the orange border?`;

      const borderReviewCompletion = await generateContentWithRetry(
        ai,
        {
          model: Gemini,
          config: {
            responseMimeType: "text/plain" /* thinkingConfig: {
              thinkingBudget: 32768,  // Max value for deepest reasoning
              // Alternative: thinkingBudget: -1 for dynamic auto-max
            },  */,
            temperature: 0.2,
            top_p: 0.95,
          },
          systemInstruction: borderInstruction,
          contents: createUserContent([...promptImages, borderInstruction]),
        },
        3,
        apiKeys,
        {
          sessionId,
          promptLabel: "click-border-detection",
          promptContext: {
            eventIndex: event?.eventIndex,
            clipPath,
            clipType: "click",
            highlightVisible: highlightVisibility?.visibleInViewport === true,
          },
        }
      );

      // Try to parse strict JSON; fallback to default if parsing fails
      let parsed = null;
      try {
        parsed = borderReviewCompletion?.text.trim();
      } catch (_) {}

      clickPositionInfo = parsed;
    } else {
      // Set placeholder click description when containment failed
      clickPositionInfo = "Pink dot location is unidentified.";
    }

    const analysisFilePath = clipPath.replace(".mp4", ".json");
    const analysisJson = {
      clickDescription: clickPositionInfo,
      coordinateData: coordinateData || null,
      borderDetection,
    };
    await fs.writeFile(analysisFilePath, JSON.stringify(analysisJson, null, 2));

    return {
      eventIndex: event.eventIndex,
      clickDescription: clickPositionInfo,
      clickPositionInfo: clickPositionInfo, // Add this for backward compatibility
      clipPath: clipPath,
      analysisJson: analysisJson,
      isAlreadyLabeled: isAlreadyLabeled,
      event: event,
      was_second_try: was_second_try,
      isClickVerified: isClickVerified,
      isPointInClaimed: isPointInClaimed,
      highlightVisibility: highlightVisibility || null,
      fileStartSeconds,
      fileEndSeconds,
    };
  } catch (error) {
    console.error(
      `  -> ❌ ERROR analyzing clip for event #${event.eventIndex}:`,
      error
    );
    return {
      eventIndex: event.eventIndex,
      clickDescription: "Error during analysis.",
      clickPositionInfo: "Error during analysis.",
      clipPath: originalClipPath,
      analysisJson: null,
      isAlreadyLabeled: false,
      event: event,
      was_second_try: was_second_try,
      error: error.message,
      fileStartSeconds,
      fileEndSeconds,
    };
  } finally {
    if (isTempFrame && frameImagePath) {
      await fs.unlink(frameImagePath).catch(() => {});
    }
  }
}

function parseClockTimestampToMs(ts) {
  const m = ts && ts.match(/^(\d{2}):(\d{2})\.(\d{3})$/);
  if (!m) return NaN;
  const minutes = Number(m[1]);
  const seconds = Number(m[2]);
  const millis = Number(m[3]);
  return minutes * 60_000 + seconds * 1_000 + millis;
}

// Make context range timestamps consecutive by snapping small gaps (<= thresholdMs)
// Adds `entry._adjustedContextTimestamp` for range entries; leaves point entries unchanged.
function makeConsecutiveContextRanges(entries, thresholdMs = 500) {
  let lastEndMs = null;
  return entries.map((entry) => {
    if (entry && typeof entry.contextTimestamp === "string") {
      console.log("Ngewkgnwgnw ", entry.contextTimestamp);
      const parts = entry.contextTimestamp.split(`" end="`);
      console.log("Newnwekfnwef ", parts[0].split(`start="`)[1]);
      console.log("Newnwekfnwef2 ", parts[1].split(`">`)[0]);

      const startMs = parseClockTimestampToMs(
        parts[0].split(`start="`)[1]?.trim()
      );
      const endMs = parseClockTimestampToMs(parts[1].split(`">`)[0]?.trim());
      if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
        return entry;
      }
      if (lastEndMs !== null) {
        const gap = startMs - lastEndMs;
        if (Math.abs(gap) <= thresholdMs) {
          const rawDuration = Math.max(0, endMs - startMs);
          const newStart = Math.max(lastEndMs + 1, 0);
          const newEnd = endMs;
          entry._adjustedContextTimestamp = `<segment>`;
          lastEndMs = newEnd;
          return entry;
        }
      }
      lastEndMs = endMs;
      return entry;
    }
    // Do not alter point events; keep lastEndMs as-is so only ranges drive consecutiveness.
    return entry;
  });
}

function addRrwebContextMenuEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding rrweb contextmenu events to activity log...");

  if (!rrwebEvents || rrwebEvents.length === 0) {
    console.log("   No rrweb events provided - nothing to add");
    return logEntries;
  }

  // Filter for contextmenu events: type 3 (Incremental Snapshot), data.source 2 (MouseInteraction), data.type 3 (ContextMenu)
  const contextMenuEvents = rrwebEvents.filter(
    (event) =>
      event.type === 3 && event.data?.source === 2 && event.data?.type === 3
  );

  if (contextMenuEvents.length === 0) {
    console.log("   No contextmenu events found in rrweb data");
    return logEntries;
  }

  console.log(
    `   Found ${contextMenuEvents.length} contextmenu events to process`
  );

  const adjustedRecordingStart = rrwebEvents[0]?.timestamp;
  if (!adjustedRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  const originalRecordingStart =
    rrwebEvents[0]?.originalTimestamp || adjustedRecordingStart;

  const newLogEntries = [...logEntries];
  const idNodeMap = buildRrwebNodeMap(rrwebEvents);

  for (const contextMenuEvent of contextMenuEvents) {
    const adjustedAbsoluteMs = contextMenuEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - adjustedRecordingStart;

    const originalAbsoluteMs =
      contextMenuEvent.originalTimestamp || contextMenuEvent.timestamp;
    const originalRelativeTimeMs = originalAbsoluteMs - originalRecordingStart;

    let elementDescription = "an element";
    if (idNodeMap) {
      const nodeId = contextMenuEvent.data.id;
      const node = idNodeMap.get(nodeId);
      if (node) {
        elementDescription = describeRrwebNode(node, idNodeMap);
      }
    }

    const contextMenuText = `Right-clicked on something`;

    const isPreRecording = false; // ContextMenu events are from rrweb so they're during recording

    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${contextMenuText}`
      : contextMenuText;

    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: originalAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-contextmenu-${contextMenuEvent.timestamp}`,
      originalEvent: contextMenuEvent,
      source: "rrweb-contextmenu",
    };

    newLogEntries.push(newLogEntry);
  }

  // Sort by time to maintain chronological order
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  console.log(
    `   📊 Added ${contextMenuEvents.length} contextmenu events to the log`
  );

  return newLogEntries;
}

function addRrwebMouseMoveEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding rrweb mousemove events to activity log...");

  if (!rrwebEvents || rrwebEvents.length === 0) {
    console.log("   No rrweb events provided - nothing to add");
    return logEntries;
  }

  // Filter for mousemove events: type 3 (Incremental Snapshot) and data.source 1 (MouseMove)
  const mouseMoveEvents = rrwebEvents.filter(
    (event) => event.type === 3 && event.data?.source === 1
  );

  if (mouseMoveEvents.length === 0) {
    console.log("   No mousemove events found in rrweb data");
    return logEntries;
  }

  console.log(`   Found ${mouseMoveEvents.length} mousemove events to process`);

  // Get the initial recording start timestamp from the ADJUSTED timeline
  const adjustedRecordingStart = rrwebEvents[0]?.timestamp;
  if (!adjustedRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  // Also get the original start time for calculating originalRelativeTimeMs
  const originalRecordingStart =
    rrwebEvents[0]?.originalTimestamp || adjustedRecordingStart;

  const newLogEntries = [...logEntries];

  for (const mouseMoveEvent of mouseMoveEvents) {
    const adjustedAbsoluteMs = mouseMoveEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - adjustedRecordingStart;

    const originalAbsoluteMs =
      mouseMoveEvent.originalTimestamp || mouseMoveEvent.timestamp;
    const originalRelativeTimeMs = originalAbsoluteMs - originalRecordingStart;

    const positions = mouseMoveEvent.data.positions;
    if (!positions || positions.length === 0) {
      continue;
    }

    const firstPos = positions[0];
    const lastPos = positions[positions.length - 1];

    let moveText;
    if (positions.length > 1) {
      moveText = `mouse moved from (${firstPos.x}, ${firstPos.y}) to (${lastPos.x}, ${lastPos.y})`;
    } else {
      moveText = `mouse moved to (${firstPos.x}, ${firstPos.y})`;
    }

    const isPreRecording = false; // Mousemove events are from rrweb so they're during recording

    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${moveText}`
      : moveText;

    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: originalAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-mousemove-${mouseMoveEvent.timestamp}`,
      originalEvent: mouseMoveEvent,
      source: "rrweb-mousemove",
      mouseMoveData: {
        positions: positions,
      },
    };

    newLogEntries.push(newLogEntry);
  }

  // Sort by time to maintain chronological order
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  console.log(
    `   📊 Added ${mouseMoveEvents.length} mousemove events to the log`
  );

  return newLogEntries;
}

function addRrwebMouseDownEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding rrweb mousedown events to activity log...");

  if (!rrwebEvents || rrwebEvents.length === 0) {
    console.log("   No rrweb events provided - nothing to add");
    return logEntries;
  }

  // Filter for mousedown events: type 3 (Incremental Snapshot), data.source 2 (MouseInteraction), data.type 1 (MouseDown)
  const mouseDownEvents = rrwebEvents.filter(
    (event) =>
      event.type === 3 && event.data?.source === 2 && event.data?.type === 1
  );

  if (mouseDownEvents.length === 0) {
    console.log("   No mousedown events found in rrweb data");
    return logEntries;
  }

  console.log(`   Found ${mouseDownEvents.length} mousedown events to process`);

  const adjustedRecordingStart = rrwebEvents[0]?.timestamp;
  if (!adjustedRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  const originalRecordingStart =
    rrwebEvents[0]?.originalTimestamp || adjustedRecordingStart;

  const newLogEntries = [...logEntries];
  const idNodeMap = buildRrwebNodeMap(rrwebEvents);

  for (const mouseDownEvent of mouseDownEvents) {
    const adjustedAbsoluteMs = mouseDownEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - adjustedRecordingStart;

    const originalAbsoluteMs =
      mouseDownEvent.originalTimestamp || mouseDownEvent.timestamp;
    const originalRelativeTimeMs = originalAbsoluteMs - originalRecordingStart;

    let elementDescription = "an element";
    if (idNodeMap) {
      const nodeId = mouseDownEvent.data.id;
      const node = idNodeMap.get(nodeId);
      if (node) {
        elementDescription = describeRrwebNode(node, idNodeMap);
      }
    }

    const mouseDownText = `Mouse down on ${elementDescription}`;

    const isPreRecording = false; // MouseDown events are from rrweb so they're during recording

    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${mouseDownText}`
      : mouseDownText;

    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: originalAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-mousedown-${mouseDownEvent.timestamp}`,
      originalEvent: mouseDownEvent,
      source: "rrweb-mousedown",
    };

    newLogEntries.push(newLogEntry);
  }

  // Sort by time to maintain chronological order
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  console.log(
    `   📊 Added ${mouseDownEvents.length} mousedown events to the log`
  );

  return newLogEntries;
}

function addRrwebFocusEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding rrweb focus events to activity log...");

  if (!rrwebEvents || rrwebEvents.length === 0) {
    console.log("   No rrweb events provided - nothing to add");
    return logEntries;
  }

  // Filter for focus events: type 3 (Incremental Snapshot), data.source 2 (MouseInteraction), data.type 5 (Focus)
  const focusEvents = rrwebEvents.filter(
    (event) =>
      event.type === 3 && event.data?.source === 2 && event.data?.type === 5
  );

  if (focusEvents.length === 0) {
    console.log("   No focus events found in rrweb data");
    return logEntries;
  }

  console.log(`   Found ${focusEvents.length} focus events to process`);

  const adjustedRecordingStart = rrwebEvents[0]?.timestamp;
  if (!adjustedRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  const originalRecordingStart =
    rrwebEvents[0]?.originalTimestamp || adjustedRecordingStart;

  const newLogEntries = [...logEntries];
  const idNodeMap = buildRrwebNodeMap(rrwebEvents);

  for (const focusEvent of focusEvents) {
    const adjustedAbsoluteMs = focusEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - adjustedRecordingStart;

    const originalAbsoluteMs =
      focusEvent.originalTimestamp || focusEvent.timestamp;
    const originalRelativeTimeMs = originalAbsoluteMs - originalRecordingStart;

    let elementDescription = "an element";
    if (idNodeMap) {
      const nodeId = focusEvent.data.id;
      const node = idNodeMap.get(nodeId);
      if (node) {
        elementDescription = describeRrwebNode(node, idNodeMap);
      }
    }

    const focusText = `Focused ${elementDescription}`;

    const isPreRecording = false; // Focus events are from rrweb so they're during recording

    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${focusText}`
      : focusText;

    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: originalAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-focus-${focusEvent.timestamp}`,
      originalEvent: focusEvent,
      source: "rrweb-focus",
    };

    newLogEntries.push(newLogEntry);
  }

  // Sort by time to maintain chronological order
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  console.log(`   📊 Added ${focusEvents.length} focus events to the log`);

  return newLogEntries;
}

function addRrwebMouseUpEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding rrweb mouseup events to activity log...");

  if (!rrwebEvents || rrwebEvents.length === 0) {
    console.log("   No rrweb events provided - nothing to add");
    return logEntries;
  }

  // Filter for mouseup events: type 3 (Incremental Snapshot), data.source 2 (MouseInteraction), data.type 0 (MouseUp is often type 0 in rrweb, same as click)
  const mouseUpEvents = rrwebEvents.filter(
    (event) =>
      event.type === 3 && event.data?.source === 2 && event.data?.type === 2
  );

  if (mouseUpEvents.length === 0) {
    console.log("   No mouseup events found in rrweb data");
    return logEntries;
  }

  console.log(`   Found ${mouseUpEvents.length} mouseup events to process`);

  const adjustedRecordingStart = rrwebEvents[0]?.timestamp;
  if (!adjustedRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  const originalRecordingStart =
    rrwebEvents[0]?.originalTimestamp || adjustedRecordingStart;

  const newLogEntries = [...logEntries];
  const idNodeMap = buildRrwebNodeMap(rrwebEvents);

  for (const mouseUpEvent of mouseUpEvents) {
    const adjustedAbsoluteMs = mouseUpEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - adjustedRecordingStart;

    const originalAbsoluteMs =
      mouseUpEvent.originalTimestamp || mouseUpEvent.timestamp;
    const originalRelativeTimeMs = originalAbsoluteMs - originalRecordingStart;

    let elementDescription = "an element";
    if (idNodeMap) {
      const nodeId = mouseUpEvent.data.id;
      const node = idNodeMap.get(nodeId);
      if (node) {
        elementDescription = describeRrwebNode(node, idNodeMap);
      }
    }

    const mouseUpText = `Mouse up on ${elementDescription}`;

    const isPreRecording = false; // MouseUp events are from rrweb so they're during recording

    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${mouseUpText}`
      : mouseUpText;

    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: originalAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-mouseup-${mouseUpEvent.timestamp}`,
      originalEvent: mouseUpEvent,
      source: "rrweb-mouseup",
    };

    newLogEntries.push(newLogEntry);
  }

  // Sort by time to maintain chronological order
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  console.log(`   📊 Added ${mouseUpEvents.length} mouseup events to the log`);

  return newLogEntries;
}

async function processSession(
  session,
  user,
  posthogProjectID,
  posthogApiKey,
  API_BASE_URL,
  API_BASE_URL2,
  apiKeys,
  options = {}
) {
  const {
    analysisId: existingAnalysisId = null,
    clipBatchResponses = null,
    restoredContext = null,
    contextUri: restoredContextUri = null,
  } = options || {};
  const clipBatchResponseMode =
    GEMINI_BATCH_FOR_CLIP_PROMPTS && Array.isArray(clipBatchResponses);
  let clickPositionResults = [];
  let analysisId = existingAnalysisId || null;
  let contextDescriptions = []; // Collect context descriptions for the final prompt
  let sessionError = null;
  let analysisDeferred = false;
  let finalizationCompleted = false;
  let sessionContextSnapshot = null;

  try {
    if (!session || typeof session.id !== "string") {
      console.warn("Invalid session object found, skipping:", session);
      return;
    }
    initializeSessionCostTracking(session.id);

    console.log(`Processing session: ${session.id} for user ${user.email}`);
    const used = process.memoryUsage();
    let memoryUsageString = "Memory usage: ";
    for (let key in used) {
      memoryUsageString += `${key} ${
        Math.round((used[key] / 1024 / 1024) * 100) / 100
      } MB; `;
    }
    console.log(memoryUsageString);

    if (!analysisId) {
      const insertQuery = `
            INSERT INTO sessionanalysis (posthogrecordingid, userid, status, processedat)
            VALUES ($1, $2, 'PROCESSING', $3)
            RETURNING id;
        `;
      const insertValues = [session.id, user.id, new Date("1980-07-07")];
      const insertResult = await pool.query(insertQuery, insertValues);
      analysisId = insertResult.rows[0].id;
    } else {
      console.log(
        `[BATCH] Reusing existing analysis record ${analysisId} for session ${session.id}.`
      );
    }

    if (
      clipBatchResponseMode &&
      restoredContext &&
      typeof restoredContext === "object"
    ) {
      sessionContextSnapshot = restoredContext;
      const finalizedContextUri =
        restoredContextUri ||
        restoredContext.contextUri ||
        restoredContext?.meta?.contextUri ||
        null;
      console.log(
        `[BATCH] Resuming session ${session.id} from persisted context${finalizedContextUri ? ` (${finalizedContextUri})` : ""
        }.`
      );
      await finalizeSessionFromContext({
        context: sessionContextSnapshot,
        session,
        user,
        analysisId,
        clickPositionResults: clickPositionResults,
        geminiResponses: clipBatchResponses,
        contextUri: finalizedContextUri,

      });
      finalizationCompleted = true;
      return;
    }

    // Prepare variables used by both modes
    let serverStartUrl = null;
    let recording_duration = "N/A";
    let click_count = "N/A";
    let datetime = "N/A";
    let processedAt = null;
    let end_time = "N/A";
    let person_id = "N/A";
    let browser = "N/A";
    let osVar = "N/A";
    let osVersion = "N/A";
    let device_type = "N/A";
    let country = "N/A";
    let referrer = "N/A";
    let idVar = "N/A";
    let user_agent = "";
    let flattenedArray;
    let processedRecording;

    if (USE_OUTPUTFILE) {
      try {
        const raw = await fs.readFile("outputFile.json", "utf-8");
        processedRecording = JSON.parse(raw);
        flattenedArray = processedRecording;
        if (
          Array.isArray(processedRecording) &&
          processedRecording.length > 0
        ) {
          const firstTs = processedRecording[0]?.timestamp;
          const lastTs =
            processedRecording[processedRecording.length - 1]?.timestamp;
          if (typeof firstTs === "number") {
            try {
              datetime = new Date(firstTs).toISOString();
              processedAt =
                typeof datetime === "string" &&
                datetime &&
                datetime !== "N/A" &&
                !Number.isNaN(Date.parse(datetime))
                  ? new Date(datetime).toISOString().slice(0, 10)
                  : null;
            } catch (_) {}
          }
          if (typeof firstTs === "number" && typeof lastTs === "number") {
            recording_duration = String(Math.max(0, (lastTs - firstTs) / 1000));
          }
        }
        serverStartUrl = "LOCAL";
        console.log(
          `[USE_OUTPUTFILE] Loaded ${
            processedRecording?.length || 0
          } rrweb events from outputFile.json.`
        );
      } catch (e) {
        console.error(
          "[USE_OUTPUTFILE] Failed to load or parse outputFile.json:",
          e?.message || e
        );
        throw e;
      }
    } else {
      let replayData;
      try {
        console.log("Nfkwenwnfnwjnf ", API_BASE_URL, API_BASE_URL2);
        replayData = await getEntireSessionReplay(
          posthogProjectID,
          session.id,
          posthogApiKey,
          API_BASE_URL,
          API_BASE_URL2
        );
      } catch (error) {
        console.log("errorerrorerror ", error);
        if (error?.name === "TooManyClicksError") {
          const reason =
            "Session contains more than 100 clicks, skipping session";
          console.log(reason, session.id);
          const failQuery = `
              UPDATE sessionanalysis
              SET status = 'FAILED', analysiscontent = $1, processedat = $3
              WHERE id = $2;
          `;
          await pool.query(failQuery, [
            reason,
            analysisId,
            new Date("1980-07-07"),
          ]);
          console.log(
            `Marked analysis ${analysisId} as FAILED in the database because it exceeded the click limit.`
          );
          return;
        }
        throw error;
      }

      if (replayData === null) {
        const deleteQuery = `DELETE FROM sessionanalysis WHERE id = $1;`;
        await pool.query(deleteQuery, [analysisId]);
        console.log(
          `Deleted placeholder analysis record for session ${session.id} because it is still active.`
        );
        return;
      }

      let [
        replayEvents,
        [
          serverStartUrl0,
          recording_duration0,
          click_count0,
          end_time0,
          datetime0,
          person_id0,
          browser0,
          osVar0,
          osVersion0,
          device_type0,
          country0,
          referrer0,
          idVar0,
          user_agent0,
        ],
      ] = replayData;

      serverStartUrl = serverStartUrl0;
      recording_duration = recording_duration0;
      click_count = click_count0;
      datetime = datetime0;
      processedAt =
        typeof datetime === "string" &&
        datetime &&
        datetime !== "N/A" &&
        !Number.isNaN(Date.parse(datetime))
          ? new Date(datetime).toISOString().slice(0, 10)
          : null;
      end_time = end_time0;
      person_id = person_id0;
      browser = browser0;
      osVar = osVar0;
      osVersion = osVersion0;
      device_type = device_type0;
      country = country0;
      referrer = referrer0;
      idVar = idVar0;
      user_agent = user_agent0;

      const allSessionReplays = { [session.id]: replayEvents };
      console.log("serverStartUrlserverStartUrl ", serverStartUrl);
      /*   if (!serverStartUrl) {
        const reason = "No server start url found, skipping session";
        console.log(reason, session.id);
        const failQuery = `
                UPDATE sessionanalysis
                SET status = 'FAILED', analysiscontent = $1, processedat = $3
                WHERE id = $2;
            `;
        await pool.query(failQuery, [reason, analysisId,new Date("1980-07-07")]);
        console.log(
          `Marked analysis ${analysisId} as FAILED in the database because it lacked a start_url.`
        );
        return;
      } else */

      console.log("111gf ", serverStartUrl)
      if (
        typeof serverStartUrl === "string" &&
        (serverStartUrl.includes("http://localhost") ||
          serverStartUrl.includes("http://127.0.0.1"))
      ) {
        const reason = "No server start url found, skipping session";
        console.log(reason, session.id);

        const failQuery = `
        UPDATE sessionanalysis
        SET status = 'FAILED', analysiscontent = $1, processedat = $3
        WHERE id = $2;
    `;
        await pool.query(failQuery, [
          reason,
          analysisId,
          new Date("1980-07-07"),
        ]);
        console.log(
          `Marked analysis ${analysisId} as FAILED in the database because it lacked a start_url.`
        );
        return;
      }

      flattenedArray = Object.values(allSessionReplays)[0]
        .map((event) => event.data)
        .flat()
        .flat();
      let flattenedArray2 = Object.values(allSessionReplays)[0].flat().flat();
      if (!flattenedArray[0]) {
        flattenedArray = flattenedArray2.filter(
          (event) => typeof event !== "string"
        );
      }
    }

    const numericClickCount = Number(click_count);
      if(!Number.isNaN(numericClickCount) && (process.env.RUN_PARTICULAR_SESSION==="true" && numericClickCount > 150) || (process.env.RUN_PARTICULAR_SESSION!="true" &&numericClickCount > 60)) {
      const reason = `Session contains more than 60 clicks (count=${numericClickCount}), skipping session`;
      console.log(reason, session.id);
      const failQuery = `
      UPDATE sessionanalysis
      SET status = 'FAILED', analysiscontent = $1, processedat = $3
      WHERE id = $2;
  `;
      await pool.query(failQuery, [reason, analysisId, new Date("1980-07-07")]);
      console.log(
        `Marked analysis ${analysisId} as FAILED in the database because it exceeded the click limit.`
      );
      return;
    }

    // Start with the raw events and progressively refine a single source-of-truth variable.
    if (!processedRecording) {
      processedRecording = processPostHogRecording(flattenedArray);
    }
    // 1. Sanitize the DOCTYPE to prevent player errors.
    processedRecording = removeDoctypeFromRrwebEvents(processedRecording);

    // 2. Pre-fetch external images and rewrite events to embed data URIs for reliability.
    processedRecording = await prefetchAllImagesAndRewriteEvents(
      processedRecording
    );



    // We will re-run this later with the correct offset.
    let preprocessingResult = preprocessEventsForSkipWithMessage(
      processedRecording,
      INACTIVITY_THRESHOLD_MS,
      SKIP_MESSAGE_DURATION_MS
    );

    // Calculate and log the session duration after inactivity skipping
    if (preprocessingResult.events && preprocessingResult.events.length > 1) {
      const firstEventTs = preprocessingResult.events[0].timestamp;
      const lastEventTs =
        preprocessingResult.events[preprocessingResult.events.length - 1]
          .timestamp;
      const durationMs = lastEventTs - firstEventTs;

      /* console.log(
        `[TIMELINE] Session duration after inactivity skipping: ${
          durationMs / 1000
        }s (${durationMs}ms)`
      ); */
    /*   if (durationMs / 1000 > 360) {
        const reason = "No server start url found, skipping session";
        console.log(reason, session.id);
        const failQuery = `
        UPDATE sessionanalysis
        SET status = 'FAILED', analysiscontent = $1, processedat = $3
        WHERE id = $2;
    `;
        await pool.query(failQuery, [
          reason,
          analysisId,
          new Date("1980-07-07"),
        ]);
        console.log(
          `Marked analysis ${analysisId} as FAILED in the database because it lacked a start_url.`
        );
        return;
      } */
    }

    console.log("nfenwjfnwejfnwejn ", preprocessingResult)

    // From this point on, use the events from the preprocessing result for the clip timeline.
    processedRecording = preprocessingResult.events;
    const skips = preprocessingResult.skips;
    const overlayInstructionsDebug = preprocessingResult.overlayInstructions;

    // Log all inactivity periods for the current session recording
    try {
      const sessionStartTs = processedRecording?.[0]?.timestamp;
      console.log(
        `[INACTIVITY] Found ${skips.length} inactivity periods (threshold=${INACTIVITY_THRESHOLD_MS}ms).`
      );
      if (sessionStartTs && skips.length > 0) {
        skips.forEach((skip, idx) => {
          const startRel = skip.startTime - sessionStartTs;
          const endRel = skip.endTime - sessionStartTs;
          const originalGapMs = Math.max(0, skip.endTime - skip.startTime);
          const durationRel = skip.timeRemoved; // time actually removed from the timeline
          const keptTailMs = Math.max(0, originalGapMs - durationRel);
          console.log(
            `[INACTIVITY ${idx + 1}/${skips.length}] ${formatMilliseconds(
              startRel
            )} → ${formatMilliseconds(
              endRel
            )} | original gap ${formatMilliseconds(
              originalGapMs
            )}, removed ${formatMilliseconds(
              durationRel
            )}, kept ${formatMilliseconds(keptTailMs)}`
          );
        });
      }

      // Also log overlay messages (which are based on the original gap, not the removed duration)
      if (sessionStartTs && Array.isArray(overlayInstructionsDebug)) {
        console.log(
          `[INACTIVITY-OVERLAYS] ${overlayInstructionsDebug.length} overlay messages:`
        );
        overlayInstructionsDebug.forEach((ov, i) => {
          const showRel = ov.showAt - sessionStartTs;
          const hideRel = ov.hideAt - sessionStartTs;
          console.log(
            `[OVERLAY ${i + 1}/${overlayInstructionsDebug.length}] ${
              ov.message
            } | show ${formatMilliseconds(showRel)} → hide ${formatMilliseconds(
              hideRel
            )}`
          );
        });
      }
    } catch (err) {
      console.warn("[INACTIVITY] Failed to log inactivity periods:", err);
    }

    // All subsequent functions should now use the fully `processedRecording` variable.
    // Save scroll events to file as early as possible
    /*     await saveScrollEventsToFile(processedRecording, session.id, skips);

    // Save navigation events to file as well
    await logPageNavigationEvents(processedRecording, session.id); */

    let allPages = extractAllVisitedUrls(processedRecording);
    let lastPage = allPages.length > 0 ? allPages[allPages.length - 1] : "N/A";
    let startUrl = allPages.length > 0 ? allPages[0] : "N/A";

    const containsStagingDomain =
      Array.isArray(allPages) &&
      allPages.some(
        (url) => typeof url === "string" && url.includes(process.env.STAGING_DOMAIN)
      );
    if (containsStagingDomain) {
      const reason = "Session visited staging environment, skipping session";
      console.log(reason, session.id);
      const failQuery = `
      UPDATE sessionanalysis
      SET status = 'FAILED', analysiscontent = $1, processedat = $3
      WHERE id = $2;
  `;
      await pool.query(failQuery, [reason, analysisId, new Date("1980-07-07")]);
      console.log(
        `Marked analysis ${analysisId} as FAILED due to staging.grow URL.`
      );
      return;
    }

    let sessionActivityLogData = [];
    if (!USE_OUTPUTFILE) {
      sessionActivityLogData = await fetchAndGenerateStandardLog(
        session.id,
        posthogProjectID,
        posthogApiKey,
        API_BASE_URL,
        API_BASE_URL2,
        recording_duration,
        skips,
        datetime,
        processedRecording
      );
    } else {
      console.log(
        "[USE_OUTPUTFILE] Skipping PostHog event fetch; building log from rrweb only."
      );
    }

    if (sessionActivityLogData.length > 0 && processedRecording.length > 0) {
      sessionActivityLogData = await enrichInputChangeEvents(
        sessionActivityLogData,
        processedRecording
      );

      // Add matched rrweb input events as separate entries in the activity log
      sessionActivityLogData = addRrwebInputEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      );

      sessionActivityLogData = addRrwebScrollEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      );

      sessionActivityLogData = addRrwebMouseMoveEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      );

      sessionActivityLogData = addRrwebContextMenuEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      );

      /*    sessionActivityLogData = addRrwebBlurEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      ); */

      /*      sessionActivityLogData = addRrwebFocusEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      ); */

      /*   sessionActivityLogData = addRrwebMouseDownEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      );
 */
      /*        sessionActivityLogData = addRrwebMouseUpEventsToLog(
        sessionActivityLogData,
        processedRecording,
        skips
      );
 */
      // Remove the original PostHog input-change entries; keep only rrweb input entries
      try {
        const beforeCount = sessionActivityLogData.length;
        const filtered = sessionActivityLogData.filter((entry) => {
          const isPhAutocaptureChange =
            entry?.originalEvent?.event === "$autocapture" &&
            entry?.properties?.$event_type === "change";
          return !isPhAutocaptureChange;
        });
        const removed = beforeCount - filtered.length;
        if (removed > 0) {
          console.log(
            `[FILTER][PH-INPUT] removed ${removed} PostHog input-change entries; remaining ${filtered.length}`
          );
        } else {
          console.log(
            "[FILTER][PH-INPUT] no PostHog input-change entries to remove"
          );
        }
        sessionActivityLogData = filtered;
      } catch (e) {
        console.log(
          "[FILTER][PH-INPUT] failed to remove PostHog input-change entries:",
          e?.message || e
        );
      }
    }

    // NEW: Generate rrweb-based clicks (but don't merge log entries yet - they'll be added after AI analysis)
    // 4. Now that we have the processed recording and skips, we can define our click matches.
    const { matches: rrwebMatches, logEntries: rrwebClickLogEntries } =
      createClipsFromRrwebClicks(processedRecording, skips);

    // 5. NOW we can calculate the freeze time using the defined `rrwebMatches`.
    const clickCountForFreeze = rrwebMatches.length;
    const freezeTimePerClickMs = 3900; // This should match the value in recordRrwebEvents
    const totalFreezeTimeMs = clickCountForFreeze * freezeTimePerClickMs;
    console.log(
      `[TIMELINE-SYNC] Pre-calculating total freeze time: ${totalFreezeTimeMs}ms for $ clicks.`
    );

    // 6. Re-run preprocessing for the OVERLAYS ONLY, providing the freeze time as an initial offset.
    // This ensures overlay timestamps are calculated on the final, inflated timeline.
    // We only need the `overlayInstructions` from this result.
    // Use the ORIGINAL recording before the first time shift was applied
    // In USE_OUTPUTFILE mode, processedRecording already contains the rrweb events
    const rawEventsForOverlayBase = USE_OUTPUTFILE
      ? processedRecording
      : processPostHogRecording(flattenedArray);
    let finalPreprocessingResult = preprocessEventsForSkipWithMessage(
      rawEventsForOverlayBase,
      INACTIVITY_THRESHOLD_MS,
      SKIP_MESSAGE_DURATION_MS,
      totalFreezeTimeMs 
    );

    console.log("-------------------------------------------");
    // Separate pre-recording and during-recording events
    const preRecordingEvents = sessionActivityLogData.filter(
      (entry) => entry.isPreRecording && !entry.text?.includes("input changed")
    );
    const duringRecordingEvents = sessionActivityLogData.filter(
      (entry) => !entry.isPreRecording
    );
    console.log(`Found ${preRecordingEvents.length} pre-recording events.`);
    console.log(
      `Found ${duringRecordingEvents.length} during-recording events.`
    );

    // Match PostHog navigation events that happened *during* the recording with rrweb events for accurate timing.
    const matchedNavigationEvents = matchPostHogNavigationWithRrweb(
      duringRecordingEvents, // Pass only during-recording events
      processedRecording,
      skips,
      datetime
    );
    console.log(
      `Matched ${matchedNavigationEvents.length} navigation events between PostHog and rrweb.`
    );

    // Get non-navigation events from the during-recording set.
    const nonNavigationDuringRecordingEvents = duringRecordingEvents.filter(
      (entry) =>
        !entry.text?.includes("Page view:") &&
        !entry.text?.includes("Page leave:")
    );

    // Two-phase inference to avoid during-recording navs affecting pre-recording inference
    // Phase 1: infer leaves within pre-recording events only
    const preRecordingWithLeaves = insertInferredPageLeaves(
      preRecordingEvents,
      null
    );

    // Phase 2: infer leaves within during-recording events, seeded by last pre-recording page view URL
    const duringRecordingCombined = [
      ...nonNavigationDuringRecordingEvents,
      ...matchedNavigationEvents,
    ].sort((a, b) => a.adjustedMs - b.adjustedMs);

    const lastPreRecordingPageView = [...preRecordingEvents]
      .reverse()
      .find((event) => event.text?.includes("Page view:"));
    let seedUrlForDuring = null;
    if (lastPreRecordingPageView) {
      const urlMatch = lastPreRecordingPageView.text.match(/Page view: (.+)/);
      if (urlMatch && urlMatch[1]) {
        seedUrlForDuring = urlMatch[1];
      }
    }

    const duringRecordingWithLeaves = insertInferredPageLeaves(
      duringRecordingCombined,
      seedUrlForDuring
    );

    // Merge phases and sort
    let refinedLogEntries = [
      ...preRecordingWithLeaves,
      ...duringRecordingWithLeaves,
    ];
    refinedLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

    // --- DEBUG: PHASE 3 (Upstream Data) ---
    console.log(`\n\n--- DEBUG: PHASE 3 (Upstream Data) ---`);
    console.log(
      `[processSession] Master 'refinedLogEntries' created with ${refinedLogEntries.length} entries.`
    );
    console.log(`[processSession] First 10 entries in master log:`);
    console.log(
      JSON.stringify(
        refinedLogEntries.slice(0, 10).map((e) => ({
          adjustedMs: e.adjustedMs,
          text: e.text,
          source: e.source,
        })),
        null,
        2
      )
    );
    console.log(`--- END DEBUG ---\n\n`);

    // EARLY: Add inactivity overlay entries before any clip logs are constructed
    if (
      preprocessingResult.overlayInstructions &&
      preprocessingResult.overlayInstructions.length > 0
    ) {
      console.log(
        `Adding ${preprocessingResult.overlayInstructions.length} overlay instructions to the log (early)`
      );

      preprocessingResult.overlayInstructions.forEach((overlay) => {
        // Align overlay times to the adjusted (post-skip) timeline.
        // overlay.showAt and overlay.hideAt are already adjusted absolute timestamps from preprocessing.
        const showAtRelativeMs =
          overlay.showAt - processedRecording[0].timestamp;
        const hideAtRelativeMs =
          overlay.hideAt - processedRecording[0].timestamp;

        // Format the overlay text for logs, this text will be parsed later.
        const showAtFormatted = formatMilliseconds(showAtRelativeMs);
        const hideAtFormatted = formatMilliseconds(hideAtRelativeMs);
        const overlayText = `[${showAtFormatted} to ${hideAtFormatted}]: ${overlay.message}`;

        refinedLogEntries.push({
          adjustedMs: showAtRelativeMs,
          originalMs: showAtRelativeMs, // Original time is not readily available for synthetic overlay events.
          originalAbsoluteMs: overlay.showAt, // This is an adjusted absolute timestamp.
          text: overlayText,
          isPreRecording: false,
          eventIndex: `overlay-${overlay.showAt}`,
          source: "inactivity-overlay",
        });
      });

      // Re-sort after adding overlay instructions
      refinedLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);
    }

    console.log("-------------------------------------------");

    console.log(
      `Created a refined log with ${refinedLogEntries.length} total events.`
    );

    /*    while (true) {} */

    /*    await new Promise((resolve) => setTimeout(resolve, 1000000000)); */

    // Since we're now using rrweb clicks directly, we don't need to filter PostHog events
    console.log(`📊 CLIP ANALYSIS SUMMARY:`);
    console.log(`   Total rrweb clicks for analysis: ${rrwebMatches.length}`);

    // =================================================================================
    // THIS IS THE NEW, CORRECTED BLOCK TO PASTE IN
    // =================================================================================

    const SKIP_MAIN_RECORDING =
      process.env.SKIP_MAIN_RECORDING === "true" ||
      START_FROM_RECORDED_CLIPS ||
      process.env.ONLY_ANALYSIS === "true";
    const videoFileName = `clips_foldervideo${session.id}.mp4`;

    // This variable will hold the result of the recording, including the timing offset.
    // We declare it here so it's available to the subsequent logic blocks.
    let recordingResult;
console.log("kfwnwnfkw ", processedRecording)
    // Ensure we have rrweb events before computing clip plans.
    if (!Array.isArray(processedRecording) || processedRecording.length === 0) {
      throw new Error(
        "[CLIP] Missing processedRecording events; cannot build clip plan."
      );
    }

    // 1. Calculate the total logical duration from the rrweb events.
    const firstEventTs = processedRecording[0].timestamp;
    const lastEventTs =
      processedRecording[processedRecording.length - 1].timestamp;
    const totalLogicalDurationSeconds = (lastEventTs - firstEventTs) / 1000;

    // 2. Define all initial logical clip segments.
    const { allClips: initialClips } = defineLogicalClipSegments(
      rrwebMatches,
      totalLogicalDurationSeconds
    );

    // 3. Merge short pre-click gaps.
    const mergedClips = mergeShortPreClickSegments(initialClips);

    // 4. NEW: Adjust the merged clip boundaries to fully contain any adjacent inactivity overlays.
    // This creates the FINAL MASTER PLAN for all clips_folder.
    const finalClips = adjustClipBoundariesForOverlays_V3C(
      // <-- USE THE NEW FUNCTION HERE
      mergedClips,
      preprocessingResult.overlayInstructions,
      firstEventTs
    );

    // 5. Inject markers based on the FINAL master plan. This ensures beacons in the main clip are correct.
    console.log(
      "Injecting clip markers into event stream based on final plan..."
    );
    const processedRecordingWithMarkers = injectClipMarkersIntoEventStream(
      processedRecording,
      finalClips, // <--- Use the FINAL, adjusted clips_folder for beacon injection
      {
        sessionId: session.id,
        overlayInstructions: preprocessingResult.overlayInstructions,
        overlayToleranceAfterEndMs: 0,
        epsilonMs: 1,
      }
    );

    // Add markers to refinedLogEntries from the marked-up stream
    try {
      const markerEvents = processedRecordingWithMarkers.filter(
        (event) => event.type === 5 && event.data?.tag === "CLIP_MARKER"
      );
      if (markerEvents.length > 0) {
        console.log(
          `[LOG-INJECT] Found ${markerEvents.length} marker events to inject into the log.`
        );
        const markerLogEntries = markerEvents.map((marker) => {
          console.log(
            `\n[MARKER_TO_LOG] Processing marker: "${marker.data.payload.description}"`
          );
          // NOTE: marker.timestamp is already on the ADJUSTED (post-skip) timeline,
          // because we injected markers into the preprocessed event stream.
          // Therefore, DO NOT apply skip-based time shifting again here.
          const adjustedAbsoluteTimestamp = marker.timestamp;
          console.log(
            `[MARKER_TO_LOG]   -> adjustedAbsoluteTimestamp: ${adjustedAbsoluteTimestamp}`
          );
          const adjustedRelativeTimeMs =
            adjustedAbsoluteTimestamp - firstEventTs;
          console.log(
            `[MARKER_TO_LOG]   -> firstEventTs:                ${firstEventTs}`
          );
          console.log(
            `[MARKER_TO_LOG]   -> FINAL adjustedMs:            ${adjustedRelativeTimeMs} (already adjusted)`
          );

          return {
            adjustedMs: adjustedRelativeTimeMs,
            // We don't have a reliable original (pre-skip) time for synthetic markers.
            originalMs: adjustedRelativeTimeMs,
            originalAbsoluteMs: adjustedAbsoluteTimestamp,
            text: `--- ${marker.data.payload.description} ---`,
            isPreRecording: false,
            eventIndex: `marker-${marker.timestamp}`,
            source: "system-marker",
          };
        });
        refinedLogEntries.push(...markerLogEntries);
        // Re-sort to ensure chronological order after injection
        refinedLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);
        console.log(
          `[LOG-INJECT] Successfully injected ${markerLogEntries.length} marker log entries.`
        );
      }
    } catch (e) {
      console.warn(
        `[LOG-INJECT] Failed to inject marker events into log: ${
          e?.message || e
        }`
      );
    }

    // Now decide whether to record video
    if (!SKIP_MAIN_RECORDING) {
      // Use the stream with markers for clip generation
      processedRecording = processedRecordingWithMarkers;

      // Store events before pauses are added, but after markers are injected.
      const eventsBeforePauses = processedRecording;

      // NEW: Add 2s pauses BEFORE each yellow marker in the stream
      processedRecording = addPausesForMarkers(processedRecording, 2000);

      // NEW: Adjust overlay instructions for the new timeline
      const adjustedOverlayInstructions = adjustOverlayInstructionsForPauses(
        preprocessingResult.overlayInstructions,
        eventsBeforePauses,
        2000
      );

      // STEP 4: Record the main session clip (now with markers and pauses) and get the result.
      recordingResult = await recordRrwebEvents(
        processedRecording,
        adjustedOverlayInstructions, // Use the adjusted instructions
        videoFileName,
        session.id,
        apiKeys
      );

      if (!recordingResult || !recordingResult.success) {
        throw new Error("Main clip generation failed.");
      }
    } else {
      console.log(
        `[SKIP_MAIN_RECORDING] Assuming ${videoFileName} exists and skipping main recording.`
      );
      // When skipping, we must create a placeholder result object.
      // Assume 0 offset if we don't have the real data.
      recordingResult = { success: true, timingOffsetMs: 0 };

      // Immediately cut the existing master clip into segments using beacons
      try {
        const shouldCutByBeacons =
          String(
            process.env.CUT_VIDEO_BASED_ON_BEACONS || "true"
          ).toLowerCase() === "true";
        if (shouldCutByBeacons && process.env.ONLY_ANALYSIS !== "true") {
          const sessionDir = path.join(
            process.cwd(),
            "clips_folder",
            `session_${session.id}`
          );
          await fs.mkdir(sessionDir, { recursive: true });
          console.log(
            `[BEACON-CUT][SKIP] Cutting existing master clip into: ${sessionDir}`
          );
          const cutResult = await cutVideoBasedOnBeacons(
            videoFileName,
            sessionDir,
             {
              includeClick: true,
              includeNoClick: true,
              removeBeaconFrames: true,
              addAudio: true,
            }
         
            /*    {
              includeClick: true,
              includeNoClick: NOCLICK_FROM_GAPS
                ? false
                : INCLUDE_NOCLICK_SEGMENTS,
              removeBeaconFrames: true,
              addAudio: true,
            } */
          );
          try {
            const indexPath = path.join(
              sessionDir,
              "beacon_segments",
              "segments_index.json"
            );
            await fs.mkdir(path.dirname(indexPath), { recursive: true });
            await fs.writeFile(
              indexPath,
              JSON.stringify({ segments: cutResult.segments }, null, 2)
            );
            console.log(
              `[BEACON-CUT][SKIP] Wrote segments index: ${indexPath}`
            );
          } catch (_) {}
        } else {
          console.log(
            "[BEACON-CUT][SKIP] Skipped (CUT_VIDEO_BASED_ON_BEACONS != true)"
          );
        }
      } catch (e) {
        console.warn(
          "[BEACON-CUT][SKIP] Cutting step failed (non-fatal):",
          e?.message || e
        );
      }
    }

    // --- End logical clip generation section ---

    const clipsDir = path.join(
      process.cwd(),
      "clips_folder",
      `session_${session.id}`
    );
    await fs.mkdir(clipsDir, { recursive: true });

    const videoStartTimestamp = processedRecording[0].timestamp;

    // =============================================================
    // STEP 1: Generate all clips_folder and collect them for analysis
    // =============================================================
    console.log("\n==== STEP 1: GENERATING ALL VIDEO CLIPS ====");
    const clipsToAnalyze = [];

    // Use a single browser instance with isolated contexts to avoid memory accumulation
    // Browser contexts provide isolation while sharing the browser process
    console.log(
      "🚀 Using shared browser with isolated contexts for memory efficiency"
    );

    let sharedBrowser = null;
    let browserLaunchAttempts = 0;
    const maxBrowserLaunchAttempts = 3;

    try {
      const filesInClipsDir = await fs.readdir(clipsDir);
      const clickWindows = [];

      for (let i = 0; i < rrwebMatches.length; i++) {
        const match = rrwebMatches[i];
        const phEvent = match.postHogEvent;

        // Find the corresponding click clip file on disk.
        // New format: click_0_from_11.625s_to_20.375s.mp4
        const clipFileRegex = new RegExp(
          `^click_${phEvent.eventIndex}_from_.*\\.mp4$`
        );
        // Find the base clip file, not the processed one.
        const clipFileName = filesInClipsDir.find(
          (f) => clipFileRegex.test(f) && !f.includes("_processed")
        );

        if (!clipFileName) {
          console.warn(
            `[ANALYSIS-QUEUE] Clip file for event ${phEvent.eventIndex} not found in ${clipsDir} matching click_${phEvent.eventIndex}_from_....mp4. Skipping analysis.`
          );
          continue;
        }

        const filenameTimeRegex =
          /^click_\d+_from_([0-9]+(?:\.[0-9]+)?)s_to_([0-9]+(?:\.[0-9]+)?)s\.mp4$/;
        const timeMatch = clipFileName.match(filenameTimeRegex);
        let fileStartSeconds = null;
        let fileEndSeconds = null;
        if (timeMatch) {
          fileStartSeconds = parseFloat(timeMatch[1]);
          fileEndSeconds = parseFloat(timeMatch[2]);
        }

        let tempClipPath = path.join(clipsDir, clipFileName);
        let clipResult = {
          success: false,
          was_second_try: false,
          coordinateData: null,
        };

        try {
          await fs.access(tempClipPath);
          clipResult.success = true;
          console.log(
            `📹 Found pre-generated clip, queuing for analysis: ${clipFileName}`
          );
        } catch (e) {
          console.warn(
            `[ANALYSIS-QUEUE] Pre-generated clip not found: ${clipFileName}`
          );
          clipResult.success = false;
        }

        if (clipResult.success) {
          clipsToAnalyze.push({
            clipPath: tempClipPath,
            event: phEvent,
            rrwebEvent: match.rrwebEvent, // Include rrweb event data for AI analysis
            was_second_try: clipResult.was_second_try,
            coordinateData: clipResult.coordinateData,
            preClickDurationMs: clipResult.preClickDurationMs, // Store the freeze start time
            // Surface verification to downstream prompts
            isClickVerified: clipResult.isClickVerified,
            fileStartSeconds,
            fileEndSeconds,
          });
                    if (
            Number.isFinite(fileStartSeconds) &&
            Number.isFinite(fileEndSeconds)
          ) {
            clickWindows.push([fileStartSeconds, fileEndSeconds]);
          }
console.log(`✅ Queued for analysis: ${path.basename(tempClipPath)}`);

          // Immediately upload the raw clip in production
          if (process.env.NODE_ENV === "production") {
            try {
              const destinationFileName = `clips_folder/${
                session.id
              }/clips_folder/${path.basename(
                tempClipPath
              )}`;
              await uploadToGCS(tempClipPath, destinationFileName);
              console.log(
                `  -> [GCS] Uploaded raw clip immediately: ${path.basename(
                  tempClipPath
                )}`
              );
            } catch (uploadError) {
              console.error(
                `[GCS] Failed to upload raw clip immediately:`,
                uploadError
              );
            }
          }
        }
      }

      // Derive noclick segments from gaps between click clips (log-only; no video files produced)
      if (NOCLICK_FROM_GAPS) {
        clickWindows.sort((a, b) => a[0] - b[0]);
        console.log("[DEBUG-GAPS] clickWindows:", JSON.stringify(clickWindows));
        

        const sessionDurationMs =
          processedRecording[processedRecording.length - 1].timestamp -
          processedRecording[0].timestamp;
        const sessionDurationSeconds = sessionDurationMs / 1000;
        let cursor = 0;
        let gapIndex = 1;
        for (const [start, end] of clickWindows) {
          if (start - cursor > 0.05) {
            clipsToAnalyze.push({
              clipPath: path.join(
                clipsDir,
                `noclick_gap_${gapIndex}_from_${cursor.toFixed(
                  3
                )}s_to_${start.toFixed(3)}s.mp4`
              ),
              event: {
                eventIndex: `noclick-${gapIndex}`,
                text: `No-click segment (gap)`,
              },
              rrwebEvent: null,
              was_second_try: false,
              coordinateData: null,
              preClickDurationMs: 0,
              isClickVerified: undefined,
              isNonClickSegment: true,
              isGapBasedNoClick: true,
              fileStartSeconds: cursor,
              fileEndSeconds: start,
            });
            gapIndex++;
          }
          cursor = Math.max(cursor, end);
        }
        if (sessionDurationSeconds - cursor > 0.05) {
          clipsToAnalyze.push({
            clipPath: path.join(
              clipsDir,
              `noclick_gap_${gapIndex}_from_${cursor.toFixed(
                3
              )}s_to_${sessionDurationSeconds.toFixed(3)}s.mp4`
            ),
            event: {
              eventIndex: `noclick-${gapIndex}`,
              text: `No-click segment (gap)`,
            },
            rrwebEvent: null,
            was_second_try: false,
            coordinateData: null,
            preClickDurationMs: 0,
            isClickVerified: undefined,
            isNonClickSegment: true,
            isGapBasedNoClick: true,
            fileStartSeconds: cursor,
            fileEndSeconds: sessionDurationSeconds,
          });
        }
      }

      // Instead of generating non-click clips_folder, load existing ones and queue for analysis
      try {
        const files = await fs.readdir(clipsDir);
        const noclickRegex =
          /^noclick_(\d+)_from_([0-9]+(?:\.[0-9]+)?)s_to_([0-9]+(?:\.[0-9]+)?)s\.mp4$/;
        let noclickFiles = files.filter((f) => noclickRegex.test(f));

        // TESTING: Limit which non-click clips_folder are loaded if TESTING_NOCLICK_DROPDOWN is enabled
        if (process.env.TESTING_NOCLICK_DROPDOWN === "true") {
          // Mirror click behavior: take the last one by default
          noclickFiles = noclickFiles.slice(3, 6);
          console.log(
            `\n🧪 NOCLICK TESTING MODE: Limiting queued existing non-click clips_folder to ${
              noclickFiles.length
            } (out of ${
              files.filter((f) => noclickRegex.test(f)).length
            } total)`
          );
        }

        for (const fileName of noclickFiles) {
          const match = fileName.match(noclickRegex);
          if (!match) continue;
          const idx = parseInt(match[1], 10);
          const fileStartSeconds = parseFloat(match[2]);
          const fileEndSeconds = parseFloat(match[3]);
          const fileDur = fileEndSeconds - fileStartSeconds;
          if (DEBUG_NOCLICK) {
            console.log(
              `[ANALYSIS-QUEUE][NOCLICK] Found existing noclick file: ${fileName} start=${fileStartSeconds.toFixed(
                3
              )}s end=${fileEndSeconds.toFixed(3)}s dur=${fileDur.toFixed(3)}s`
            );
            if (fileDur < 0.25) {
              console.warn(
                `[ANALYSIS-QUEUE][NOCLICK][WARN] Very short noclick file detected (${fileDur.toFixed(
                  3
                )}s): ${fileName}`
              );
            }
          }
          const outputPath = path.join(clipsDir, fileName);
          clipsToAnalyze.push({
            clipPath: outputPath,
            event: {
              eventIndex: `noclick-${idx}`,
              text: `No-click segment`,
            },
            rrwebEvent: null,
            was_second_try: false,
            coordinateData: null,
            preClickDurationMs: 0,
            isClickVerified: undefined,
            isNonClickSegment: true,
          });
          console.log(
            `✅ Queued existing non-click coverage clip for analysis: ${fileName}`
          );
        }
      } catch (e) {
        console.warn(
          `⚠️  Failed to load existing non-click clips_folder from ${clipsDir}:`,
          e
        );
      }
    } catch (error) {
      console.error("Error during clip generation:", error);
      throw error; // Re-throw to maintain existing error handling behavior
    } finally {
      // Clean up shared browser instance
      if (sharedBrowser) {
        try {
          console.log(
            `[MEMORY-OPT] Closing shared browser after processing ${matches.length} clips_folder`
          );
          const browserProcess = sharedBrowser.process();
          await sharedBrowser.close();
          if (browserProcess && !browserProcess.killed) {
            console.log(
              `[MEMORY-OPT] Forcefully killing browser process PID: ${browserProcess.pid}`
            );
            browserProcess.kill("SIGKILL");
          }
          // Force garbage collection after browser cleanup
          if (global.gc) {
            global.gc();
            const postCleanupMemUsage = process.memoryUsage();
            console.log(
              `[MEMORY-OPT] Memory after browser cleanup (GC): RSS=${Math.round(
                postCleanupMemUsage.rss / 1024 / 1024
              )}MB, Heap=${Math.round(
                postCleanupMemUsage.heapUsed / 1024 / 1024
              )}MB`
            );
          }
        } catch (cleanupError) {
          console.error(
            `[MEMORY-OPT] Error during browser cleanup:`,
            cleanupError.message
          );
        }
      }
    }

    // =============================================================
    // STEP 2: Two-Phase Analysis: Extract click positions first, then comprehensive analysis
    // =============================================================

    // TESTING: Limit to first 10 clips_folder if TESTING_DROPDOWN is enabled
    let finalClipsToAnalyze = clipsToAnalyze;
    if (process.env.TESTING_DROPDOWN === "true") {
      /*   finalClipsToAnalyze = clipsToAnalyze.slice(-18, -17).reverse(); */
      console.log(
        `\n🧪 TESTING MODE: Limiting analysis to first ${finalClipsToAnalyze.length} clips_folder (out of ${clipsToAnalyze.length} total)`
      );
    }

    // For Phase 1 (click position extraction), only include actual click clips_folder
    const clickClipsToAnalyze = finalClipsToAnalyze.filter(
      (task) => task && !task.isNonClickSegment
    );
    if (
      clipBatchResponseMode &&
      clipBatchResponses.length !== clickClipsToAnalyze.length
    ) {
      throw new Error(
        `[BATCH] Mismatch between clip batch responses (${clipBatchResponses.length}) and click clips (${clickClipsToAnalyze.length}) for session ${session.id}.`
      );
    }
    let clipBatchResponseIndex = 0;

    console.log(
      `\n==== STEP 2A: PHASE 1 - EXTRACTING CLICK POSITIONS FOR ${clickClipsToAnalyze.length} CLICK CLIPS ====`
    );
    // --- End of debug log ---
    // Phase 1: Extract click positions for all clips_folder (conditional based on INCLUDE_CLICK_POSITION_PROMPT)

    if (INCLUDE_CLICK_POSITION_PROMPT) {
      const clickPositionsFilePath = path.join(
        clipsDir,
        "click_positions.json"
      );

      const analysisResults = await processInParallel(
        clickClipsToAnalyze,
        (clipTask) =>
          analyzeClip(
            clipTask,
            refinedLogEntries,
            apiKeys,
            ai,
            processedRecording,
            session.id
          ),
        1 // Concurrency level
      );

      const successfulResults = analysisResults.filter(
        (result) => result && !result.error
      );
      successfulResults.sort((a, b) => a.eventIndex - b.eventIndex);

      await fs.writeFile(
        clickPositionsFilePath,
        JSON.stringify(successfulResults, null, 2)
      );
      console.log(
        `\n✅ All click positions extracted and saved to ${clickPositionsFilePath}`
      );

      clickPositionResults = successfulResults;
    } else {
      // INCLUDE_CLICK_POSITION_PROMPT is false, load from file
      const clickPositionsFilePath = path.join(
        clipsDir,
        "click_positions.json"
      );
      try {
        const data = await fs.readFile(clickPositionsFilePath, "utf-8");
        clickPositionResults = JSON.parse(data);
        console.log(
          `🚫 Loaded click positions from ${clickPositionsFilePath} (prompts skipped)`
        );
      } catch (err) {
        console.error(
          `❌ Failed to load click positions from ${clickPositionsFilePath}.`,
          err
        );
        clickPositionResults = [];
      }
    }
    if (process.env.NODE_ENV === "production") {
      console.log(
        `[GCS] Uploading entire clips_folder folder to GCS for session ${session.id}...`
      );
      try {
        // Get all files in the clips_folder directory
        const clipFiles = await fs.readdir(clipsDir);
        console.log(`  -> Found ${clipFiles.length} files to upload`);

        for (const fileName of clipFiles) {
          const filePath = path.join(clipsDir, fileName);
          const fileStat = await fs.stat(filePath);

          if (fileStat.isFile()) {
            try {
              const destinationFileName = `clips_folder/${session.id}/clips_folder/${fileName}`;
              await uploadToGCS(filePath, destinationFileName);
              console.log(`  -> Uploaded ${fileName}`);
            } catch (uploadError) {
              console.error(
                `[GCS] Failed to upload file ${fileName}:`,
                uploadError
              );
            }
          }
        }
        console.log(
          `[GCS] Completed uploading clips_folder folder for session ${session.id}`
        );
      } catch (dirError) {
        console.error(
          `[GCS] Failed to read clips_folder directory ${clipsDir}:`,
          dirError
        );
      }
    }

    console.log(
      `\n==== STEP 2B: PHASE 2 - COMPREHENSIVE ANALYSIS WITH CONTEXT ====`
    );
    // Phase 2: Comprehensive analysis with access to next clicks
    // Pre-process clips_folder to avoid event duplication across multiple clips_folder
    // Use the ACTUAL adjusted clip boundaries that were used during clip generation
    console.log("GNewkngwengwg ", clickPositionResults);
    // Get the chronological list of beacon-based click segments

    const clipsWithTiming = clickPositionResults
      .map((positionResult, index) => {
        // Find the original clip task, which contains event metadata
        const clipTask = finalClipsToAnalyze.find(
          (task) => task.event.eventIndex === positionResult.eventIndex
        );

        if (clipTask) {
          // Find the original rrweb match data for this event
          const match = rrwebMatches.find(
            (m) => m.postHogEvent.eventIndex === clipTask.event.eventIndex
          );

          // The index of the click in the analysis results corresponds to the
          // index in our chronologically sorted clickSegments array.
          // const segment = clickSegments[index];

          if (match && match.rrwebEvent) {
            // These are relative to the start of the recording.
            const logicalClipStartMs = clipTask.fileStartSeconds * 1000;
            const logicalClipEndMs = clipTask.fileEndSeconds * 1000;

            return {
              ...positionResult,
              rrwebEvent: match.rrwebEvent,
              // These are now relative ms.
              clipStartMs: logicalClipStartMs,
              clipEndMs: logicalClipEndMs,
            };
          }
        }
        // For non-click segments or if anything fails, return the original data.
        return positionResult;
      })
      .sort((a, b) => (a.clipStartMs || 0) - (b.clipStartMs || 0)); // Sort by clip start time

    try {
      console.log(
        "[CLIP-TIMING][INIT] Built clipsWithTiming (beacon-anchored)"
      );
      clipsWithTiming.forEach((c, i) => {
        if (!c) return;
        const dur =
          typeof c.clipStartMs === "number" && typeof c.clipEndMs === "number"
            ? c.clipEndMs - c.clipStartMs
            : null;
        console.log(
          `[CLIP-TIMING][INIT] #${i} eventIndex=${c.eventIndex} start=${c.clipStartMs} end=${c.clipEndMs} duration=${dur}ms rrwebTs=${c?.rrwebEvent?.timestamp} path=${c?.clipPath}`
        );
        // Probe processed clip duration if possible (for diagnostics only)
        try {
          let processedSeconds = null;
          if (c.clipPath && typeof getVideoDurationSecondsSync === "function") {
            const processedCandidate = /_processed\.mp4$/i.test(c.clipPath)
              ? c.clipPath
              : c.clipPath.replace(/\.mp4$/i, "_processed.mp4");
            try {
              processedSeconds =
                getVideoDurationSecondsSync(processedCandidate);
            } catch (_) {
              try {
                processedSeconds = getVideoDurationSecondsSync(c.clipPath);
              } catch (_) {
                processedSeconds = null;
              }
            }
          }
          if (
            Number.isFinite(processedSeconds) &&
            typeof c.clipStartMs === "number"
          ) {
            const processedMs = Math.round(processedSeconds * 1000);
            const desiredProcessedEnd =
              c.clipStartMs + Math.max(0, processedMs);
            console.log(
              `[CLIP-TIMING][PROC] #${i} processedSeconds=${processedSeconds}s processedMs=${processedMs} desiredProcessedEnd=${desiredProcessedEnd}`
            );
          } else {
            console.log(
              `[CLIP-TIMING][PROC] #${i} processed duration unavailable`
            );
          }
        } catch (_) {}
      });
      for (let i = 0; i < clipsWithTiming.length - 1; i++) {
        const cur = clipsWithTiming[i];
        const nxt = clipsWithTiming[i + 1];
        if (!cur || !nxt) continue;
        if (
          typeof cur.clipEndMs === "number" &&
          typeof nxt.clipStartMs === "number"
        ) {
          const gap = nxt.clipStartMs - cur.clipEndMs;
          console.log(
            `[CLIP-TIMING][GAP-CHECK][INIT] #${i} end -> #${
              i + 1
            } start gap=${gap}ms (cur.end=${cur.clipEndMs}, next.start=${
              nxt.clipStartMs
            })`
          );
        }
        // Also compute gap vs processed-duration-based end (diagnostics)
        try {
          if (
            typeof nxt.clipStartMs === "number" &&
            typeof getVideoDurationSecondsSync === "function"
          ) {
            let processedSeconds = null;
            if (cur.clipPath) {
              const processedCandidate = /_processed\.mp4$/i.test(cur.clipPath)
                ? cur.clipPath
                : cur.clipPath.replace(/\.mp4$/i, "_processed.mp4");
              try {
                processedSeconds =
                  getVideoDurationSecondsSync(processedCandidate);
              } catch (_) {
                try {
                  processedSeconds = getVideoDurationSecondsSync(cur.clipPath);
                } catch (_) {
                  processedSeconds = null;
                }
              }
            }
            if (
              Number.isFinite(processedSeconds) &&
              typeof cur.clipStartMs === "number"
            ) {
              const processedMs = Math.round(processedSeconds * 1000);
              const desiredProcessedEnd =
                cur.clipStartMs + Math.max(0, processedMs);
              const gapVsProcessed = nxt.clipStartMs - desiredProcessedEnd;
              console.log(
                `[CLIP-TIMING][GAP-CHECK][PROC] #${i} desiredProcessedEnd=${desiredProcessedEnd} -> #${
                  i + 1
                } start=${nxt.clipStartMs} gapVsProcessed=${gapVsProcessed}ms`
              );
            }
          }
        } catch (_) {}
      }
    } catch (_) {}

    // Adjust clip boundaries to keep inactivity overlays fully within the first clip they start in.
    // If an overlay starts within clip i and ends after clip i, extend clip i to the overlay end
    // (capped to just before the next click), and trim clip i+1 start by the same amount.
    (function adjustClipBoundariesForInactivityOverlays() {
      try {
        const overlays = preprocessingResult?.overlayInstructions || [];
        if (!Array.isArray(overlays) || overlays.length === 0) return;

        for (let i = 0; i < clipsWithTiming.length; i++) {
          const cur = clipsWithTiming[i];
          const next =
            i < clipsWithTiming.length - 1 ? clipsWithTiming[i + 1] : null;

          if (!cur || cur.clipStartMs == null || cur.clipEndMs == null)
            continue;

          console.log(
            `[OVERLAY-ADJUST][BEFORE] #${cur?.eventIndex ?? i} start=${
              cur.clipStartMs
            } end=${cur.clipEndMs} nextStart=${next?.clipStartMs}`
          );

          const nextClickTs = next?.rrwebEvent?.timestamp ?? null;

          // Find all overlays that START inside this clip and END after this clip currently ends
          const spilling = overlays.filter(
            (ov) =>
              typeof ov?.showAt === "number" &&
              typeof ov?.hideAt === "number" &&
              ov.showAt >= cur.clipStartMs &&
              ov.showAt < cur.clipEndMs &&
              ov.hideAt > cur.clipEndMs
          );

          if (spilling.length === 0) continue;

          // Determine how far we can extend: to the latest overlay.hideAt (plus small epsilon) but never past next click
          let targetEnd = cur.clipEndMs;
          for (const ov of spilling) {
            // Add small epsilon to push past last overlay frame in the recorded main video
            let candidateEnd = ov.hideAt + 50; // ms
            if (nextClickTs != null) {
              // Preserve the rule that we do not show the next click in the current clip
              candidateEnd = Math.min(candidateEnd, nextClickTs - 1);
            }
            if (candidateEnd > targetEnd) targetEnd = candidateEnd;
          }

          if (targetEnd > cur.clipEndMs) {
            const extensionMs = targetEnd - cur.clipEndMs;
            const oldEnd = cur.clipEndMs;
            cur.clipEndMs = targetEnd;

            if (
              next &&
              typeof next.clipStartMs === "number" &&
              typeof next.clipEndMs === "number"
            ) {
              // Start the next clip just after the new end of the current clip to avoid overlay tail
              const desiredNextStart = Math.max(
                next.clipStartMs,
                cur.clipEndMs + 50
              );
              // Ensure the next clip does not become negative/zero duration
              const MIN_MS = 100;
              next.clipStartMs = Math.min(
                next.clipEndMs - MIN_MS,
                desiredNextStart
              );

              console.log(
                `[OVERLAY-ADJUST] Extended clip #${
                  cur?.eventIndex ?? i
                } end from ${oldEnd} to ${
                  cur.clipEndMs
                } due to inactivity overlay; ` +
                  `trimmed next clip start to ${next.clipStartMs}`
              );
            } else {
              console.log(
                `[OVERLAY-ADJUST] Extended last clip #${
                  cur?.eventIndex ?? i
                } end from ${oldEnd} to ${
                  cur.clipEndMs
                } due to inactivity overlay`
              );
            }
          }

          console.log(
            `[OVERLAY-ADJUST][AFTER] #${cur?.eventIndex ?? i} start=${
              cur.clipStartMs
            } end=${cur.clipEndMs} nextStart=${next?.clipStartMs}`
          );
        }
      } catch (e) {
        console.warn(
          "[OVERLAY-ADJUST] Failed to adjust clip boundaries:",
          e?.message || e
        );
      }
    })();

    // Enforce no-overlap in analysis windows as well (mirror generation behavior)
    (function clampAnalysisClipWindowsToNoOverlap() {
      const SMOOTH_TRANSITION_THRESHOLD_MS = 5000;
      let lastEndMs = null;
      for (let i = 0; i < clipsWithTiming.length; i++) {
        const cur = clipsWithTiming[i];
        if (!cur) continue;
        if (i > 0) {
          const prev = clipsWithTiming[i - 1];
          const prevClickTs = prev?.rrwebEvent?.originalTimestamp ?? null;
          const curClickTs = cur?.rrwebEvent?.originalTimestamp ?? null;
          const gapMs =
            prevClickTs != null && curClickTs != null
              ? curClickTs - prevClickTs
              : null;
          const isSmooth =
            gapMs != null && gapMs < SMOOTH_TRANSITION_THRESHOLD_MS;
          if (!isSmooth && lastEndMs != null && cur.clipStartMs < lastEndMs) {
            console.log(
              `[NO-OVERLAP][ADJUST] #${cur?.eventIndex ?? i} start ${
                cur.clipStartMs
              } -> ${lastEndMs} (prev end)`
            );
            cur.clipStartMs = lastEndMs;
          }
        }
        if (cur.clipEndMs <= cur.clipStartMs) {
          console.log(
            `[NO-OVERLAP][MIN-DUR] #${cur?.eventIndex ?? i} end ${
              cur.clipEndMs
            } <= start ${cur.clipStartMs} → bump end by 100ms`
          );
          cur.clipEndMs = cur.clipStartMs + 100; // Ensure minimum duration
        }
        lastEndMs = cur.clipEndMs;
        const next =
          i < clipsWithTiming.length - 1 ? clipsWithTiming[i + 1] : null;
        if (next && typeof next.clipStartMs === "number") {
          const gapNow = next.clipStartMs - cur.clipEndMs;
          console.log(
            `[NO-OVERLAP][STATE] #${i} end=${cur.clipEndMs} → #${i + 1} start=${
              next.clipStartMs
            } gap=${gapNow}ms`
          );
        }
      }
    })();

    // Track which events have been claimed by earlier clips_folder
    const clipLogsFilePath = path.join(clipsDir, "clip_logs.json");

    let analysisResults = [];

    // === GLOBAL FILTER COVERAGE DIAGNOSTICS (init) ===
    // Collect rrweb-aligned filter windows across all clips_folder to verify adjacency and coverage
    const rrwebFilterWindows = [];
    // Build a universe of filterable events (non pre-recording) to audit leftovers across all windows
    const filterableEventsUniverse = Array.isArray(refinedLogEntries)
      ? refinedLogEntries.filter(isEventFilterable)
      : [];
    if (DEBUG_CLICK) {
      try {
        const missingAdjMs = filterableEventsUniverse.filter(
          (e) => typeof e.adjustedMs !== "number"
        ).length;
        console.log(
          `[FILTER-COVERAGE][INIT] filterableEvents=${filterableEventsUniverse.length} missingAdjustedMs=${missingAdjMs}`
        );
      } catch (_) {}
    }

    // NEW: Create sorted lists of end markers for both clip types
    const allClickEndMarkers = refinedLogEntries
      .filter((e) => e.text && e.text.includes("End of click clip"))
      .sort((a, b) => a.adjustedMs - b.adjustedMs);

    if (INCLUDE_CLIP_LOG_PROMPT) {
      const analysisPromises = clipsWithTiming.map(
        async (positionResult, currentIndex) => {
          const {
            eventIndex,
            clickPositionInfo,
            isAlreadyLabeled,
            clipPath,
            event,
            was_second_try,
            clipStartMs,
            clipEndMs,
          } = positionResult;
          try {
            console.log(
              `  -> Comprehensive analysis for event #${eventIndex}: ${path.basename(
                clipPath
              )}`
            );

            // Get the next 2 clicks for context
            const nextClicksInfo = [];
            for (
              let i = currentIndex + 1;
              i < Math.min(currentIndex + 3, clipsWithTiming.length);
              i++
            ) {
              const nextResult = clipsWithTiming[i];
              nextClicksInfo.push(
                `Clicked "${nextResult.clickPositionInfo}"\n`
              );
            }
            const nextClicks =
              nextClicksInfo.length > 0
                ? nextClicksInfo.join("\n")
                : "No more clicks in this session";

            console.log(`     -> Next clicks context: ${nextClicks}`);

            // Prepare clip for analysis
            const videoBuffer = await fs.readFile(clipPath);
            const videoBase64 = videoBuffer.toString("base64");
            // Use dynamic FPS matching the processed clip duration when available
            let effectiveClipDurationSeconds = (clipEndMs - clipStartMs) / 1000;
            try {
              if (
                clipPath &&
                typeof getVideoDurationSecondsSync === "function"
              ) {
                const processedCandidate = /_processed\.mp4$/i.test(clipPath)
                  ? clipPath
                  : clipPath.replace(/\.mp4$/i, "_processed.mp4");
                const secs =
                  getVideoDurationSecondsSync(processedCandidate) ??
                  getVideoDurationSecondsSync(clipPath);
                if (Number.isFinite(secs)) {
                  effectiveClipDurationSeconds = Math.max(0, secs);
                }
              }
            } catch (_) {}
            const { apiFps: clipFpsForApi } = getRecordingSettings(
              effectiveClipDurationSeconds
            );
            const videoPart = {
              inlineData: { mimeType: "video/mp4", data: videoBase64 },
              videoMetadata: {
                fps: clipFpsForApi,
              },
            };

            console.log(
              "clickPositInfoclickPositionInfo ",
              clickPositionInfo,
              "\n-----------------\n",
              nextClicks
            );

            // Diagnostics: show clip timing windows and gaps near analysis time
            try {
              const curIdx = currentIndex;
              const curClip = clipsWithTiming[curIdx];
              const prevClip = curIdx > 0 ? clipsWithTiming[curIdx - 1] : null;
              const nextClip =
                curIdx < clipsWithTiming.length - 1
                  ? clipsWithTiming[curIdx + 1]
                  : null;
              console.log("[ANALYSIS][CLIP] index=", curIdx, {
                clipStartMs: curClip?.clipStartMs,
                clipEndMs: curClip?.clipEndMs,
                rrwebClickTs: positionResult?.rrwebEvent?.timestamp,
                prevEndMs: prevClip?.clipEndMs,
                nextStartMs: nextClip?.clipStartMs,
                gapFromPrev:
                  prevClip &&
                  typeof prevClip.clipEndMs === "number" &&
                  typeof curClip?.clipStartMs === "number"
                    ? curClip.clipStartMs - prevClip.clipEndMs
                    : null,
                gapToNext:
                  nextClip &&
                  typeof nextClip.clipStartMs === "number" &&
                  typeof curClip?.clipEndMs === "number"
                    ? nextClip.clipStartMs - curClip.clipEndMs
                    : null,
              });
            } catch (_) {}

            let reviewedText = null;
            let contextDescription = null;

            console.log(`     -> Event needs full analysis with context`);

            if (currentIndex === 0) {
              /*      refinedLogEntries = refinedLogEntries.map(value => {
                  if(value?.text) {
                    return {
                      ...value,
                      text: value?.text.replace("[PRE-RECORDING (mention in the output)] ","")
                    }
                  } else {
                    return value
                  }
                })


                console.log(
                  "refinedLogEntriesrefinedLogEntries2 ",
                  refinedLogEntries
                ); */
            }
            let clipEventLog =
              "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded";
            let clipEventIds = []; // Track event IDs included in this clip

            console.log(`clipEventLogclipEventLog`, clipEventLog);
            console.log(
              `DEBUG: clipStartMs=${clipStartMs}, clipEndMs=${clipEndMs}, refinedLogEntries.length=${refinedLogEntries.length}`
            );

            // --- DEBUG: PHASE 1 (Filtering Window for First Clip) ---
            if (currentIndex === 0) {
              console.log(
                `\n\n--- DEBUG: PHASE 1 (Filtering Window for First Clip) ---`
              );
              console.log(
                `[analyzeClip][First Clip] Initial clipStartMs: ${clipStartMs}, clipEndMs: ${clipEndMs}`
              );
              console.log(
                `[analyzeClip][First Clip] Refined log has ${refinedLogEntries.length} entries.`
              );
              console.log(
                `[analyzeClip][First Clip] First 5 log entries:`,
                JSON.stringify(
                  refinedLogEntries.slice(0, 5).map((e) => ({
                    adjustedMs: e.adjustedMs,
                    text: e.text,
                    source: e.source,
                  })),
                  null,
                  2
                )
              );
            }
            // --- END DEBUG ---

            let clipStartRelativeMs = null;
            let effectiveClipStartRelativeMs = null;
            let filterEndRelativeMs = null; // New variable for end

            if (
              typeof clipStartMs === "number" &&
              typeof clipEndMs === "number" &&
              refinedLogEntries.length > 0
            ) {
              // --- START: Symmetrical logic for event window calculation ---
              const allMarkers = refinedLogEntries
                .filter((e) => e.text && e.text.startsWith("--- End of"))
                .sort((a, b) => a.adjustedMs - b.adjustedMs);

              // NEW: Use the currentIndex to find the correct end marker
              const thisClickEndMarker = allClickEndMarkers[currentIndex];

              if (thisClickEndMarker) {
                filterEndRelativeMs = thisClickEndMarker.adjustedMs;
                const currentMarkerIndex = allMarkers.findIndex(
                  (m) =>
                    m.adjustedMs === thisClickEndMarker.adjustedMs &&
                    m.text === thisClickEndMarker.text
                );

                if (currentMarkerIndex > 0) {
                  effectiveClipStartRelativeMs =
                    allMarkers[currentMarkerIndex - 1].adjustedMs;
                } else if (currentMarkerIndex === 0) {
                  effectiveClipStartRelativeMs = 0;
                } else {
                  // Fallback for safety if marker not found in allMarkers
                  console.warn(
                    `[CLICK-WARN] Could not find end marker for click #${eventIndex} in allMarkers list. Falling back to clip start time.`
                  );
                  effectiveClipStartRelativeMs = clipStartMs;
                }
              } else {
                console.error(
                  `[CLICK-ERROR] No end marker for click clip #${currentIndex} (eventIndex: ${eventIndex}). There are ${allClickEndMarkers.length} markers for ${clipsWithTiming.length} clips_folder. Using clip times as fallback.`
                );
                effectiveClipStartRelativeMs = clipStartMs;
                filterEndRelativeMs = clipEndMs;
              }
              // --- END: Symmetrical logic ---

              // --- DEBUG: PHASE 1 (cont.) ---
              console.log("Nkewngwkngewng ", currentIndex);
              if (currentIndex === 0) {
                console.log(
                  `[analyzeClip][First Clip] Calculated filtering window: [${effectiveClipStartRelativeMs}, ${filterEndRelativeMs}]`
                );
                console.log(`--- END DEBUG ---\n\n`);
              }
              // --- END DEBUG ---

              console.log("[FILTER-WINDOW]", {
                eventIndex,
                filterStartMs: effectiveClipStartRelativeMs,
                filterEndMs: filterEndRelativeMs,
                rrwebWindowMs:
                  filterEndRelativeMs - effectiveClipStartRelativeMs,
                note: "Using logical clip boundaries on adjusted timeline",
              });

              let clipEvents = refinedLogEntries.filter((entry) => {
                // --- DEBUG: PHASE 2 (Event Filtering Logic for First Clip) ---
                const logAndReturn = (decision, reason) => {
                  if (currentIndex === 0) {
                    console.log(
                      `[Filter][Clip 0] ${decision} (t=${
                        entry.adjustedMs
                      }ms): ${reason} -> "${entry.text
                        ?.substring(0, 70)
                        .replace(/\n/g, " ")}..."`
                    );
                  }
                  return decision === "INCLUDE";
                };
                // --- END DEBUG ---

                // Basic sanity checks: must have text, not be a pre-recording event, and not an internal marker.
                if (
                  !entry.text ||
                  entry.isPreRecording ||
                  entry.isReplayMarker
                ) {
                  return logAndReturn("EXCLUDE", "Basic sanity check failed");
                }

                if (entry.text.includes("-- End of ")) {
                  return logAndReturn("EXCLUDE", "Is an end marker");
                }

                // Exclude raw PostHog click events from the clip's context log
                if (
                  entry.originalEvent &&
                  entry.originalEvent.event === "$autocapture" &&
                  entry.originalEvent.properties?.$event_type === "click"
                ) {
                  return logAndReturn(
                    "EXCLUDE",
                    "Is a raw PostHog autocapture click"
                  );
                }

                // Check if the event's timestamp is within the clip's boundaries.
                const inWindow =
                  entry.adjustedMs >= effectiveClipStartRelativeMs &&
                  entry.adjustedMs <= filterEndRelativeMs;

                // Special handling for inactivity overlays that might span across clip boundaries.
                // We include them if they overlap with the clip's window at all.
                let isSpanningOverlay = false;
                if (
                  entry.source === "inactivity-overlay" &&
                  typeof entry.text === "string"
                ) {
                  const m = entry.text.match(
                    /^\[(\d):(\d)\.(\d) to (\d):(\d)\.(\d)\]: (.+)$/
                  );
                  if (m) {
                    const startMs =
                      Number(m[1]) * 60000 + Number(m[2]) * 1000 + Number(m[3]);
                    const endMs =
                      Number(m[4]) * 60000 + Number(m[5]) * 1000 + Number(m[6]);
                    // Check for any overlap: (StartA < EndB) and (EndA > StartB)
                    if (
                      startMs < filterEndRelativeMs &&
                      endMs > effectiveClipStartRelativeMs
                    ) {
                      isSpanningOverlay = true;
                    }
                  }
                }

                if (entry.text.startsWith("Clicked ")) {
                  return logAndReturn(
                    "EXCLUDE",
                    "Is a 'Clicked ' summary event"
                  );
                }

                if (entry.text.startsWith("Event: ")) {
                  return logAndReturn("EXCLUDE", "Is a generic 'Event:' log");
                }

                // The event should be included if it's strictly within the window or is an overlapping overlay.
                const finalDecision = inWindow || isSpanningOverlay;

                if (!finalDecision) {
                  return logAndReturn(
                    "EXCLUDE",
                    `Timestamp not in window [${effectiveClipStartRelativeMs}, ${filterEndRelativeMs}]`
                  );
                }

                return logAndReturn("INCLUDE", "Passed all checks");
              });

              const clipTask = finalClipsToAnalyze.find(
                (task) => task.event.eventIndex === eventIndex
              );
              if (clipTask) {
                const borderVisibleFromResult =
                  positionResult && positionResult.highlightVisibility
                    ? positionResult.highlightVisibility.visibleInViewport ===
                      true
                    : false;

                console.log(
                  "Ngkewgnwkgewkgnw ",
                  positionResult,
                  borderVisibleFromResult
                );

                const clickDescriptionBody =
                  positionResult.isPointInClaimed === true
                    ? `**Pink dot = 100% accurate pointer release (pointerup) location**\n\n${clickPositionInfo}`
                    : borderVisibleFromResult
                    ? `**Element with orange border = element where the pointerup occured**\n\n${clickPositionInfo}`
                    : `The cursor and pointerup locations are incorrect because the rrweb data is corrupted.\n\n**Instructions:** **Never ever specify a location of cursor/pointerup!** State that the user released the pointer ('pointerup'), but explain that the exact location is unavailable due to data corruption. Additionally, say if something happened as a result of that release!`;

                // Ensure the click event timestamp is relative to the same
                // clip window baseline we use for formatting. In rare cases,
                // different baselines can cause the click to appear far
                // outside the clip (e.g., ~50s). Compute the click offset
                // within this clip from absolute rrweb times, then anchor
                // it to the clip's relative start used for rendering.
                // Compute the click time relative to the processed recording start
                // so we stay on the same adjusted timeline as refinedLogEntries/markers.
                const recordingStartAbsoluteMs =
                  (processedRecording && processedRecording[0]
                    ? processedRecording[0].timestamp
                    : typeof firstEventTs === "number"
                    ? firstEventTs
                    : 0) || 0;

                const rrwebEventRelativeMs = Math.max(
                  0,
                  (positionResult?.rrwebEvent?.timestamp ?? 0) -
                    recordingStartAbsoluteMs
                );

                // Offset of the click within this window, measured from the window's baseline
                // (previous end marker). This ensures consistency with filtering/formatting baselines.
                const clickOffsetWithinClipMs = Math.max(
                  0,
                  rrwebEventRelativeMs - (effectiveClipStartRelativeMs ?? 0)
                );

                // Window baseline + in-window offset equals the correct adjusted timestamp
                // for the click on the refined/processed timeline.
                const clickAdjustedForLogMs =
                  (effectiveClipStartRelativeMs ?? 0) + clickOffsetWithinClipMs;

                const clickEvent = {
                  adjustedMs: clickAdjustedForLogMs,
                  text: `<pointerup>\n${clickDescriptionBody}\n</pointerup>`,
                };
                clipEvents.push(clickEvent);
              }

              clipEvents.sort((a, b) => a.adjustedMs - b.adjustedMs);

              // NEW: Find the page URL active *before* this clip started
              let pageUrlBeforeClip = null;
              for (let j = refinedLogEntries.length - 1; j >= 0; j--) {
                const entry = refinedLogEntries[j];
                if (
                  entry.adjustedMs < clipStartRelativeMs &&
                  entry.text &&
                  entry.text.includes("Page view:")
                ) {
                  const urlMatch = entry.text.match(/Page view: (.+)/);
                  if (urlMatch && urlMatch[1]) {
                    pageUrlBeforeClip = urlMatch[1];
                    break; // Found the most recent one
                  }
                }
              }

              // Insert inferred page leave events, providing the initial context.
              clipEvents = insertInferredPageLeaves(
                clipEvents,
                pageUrlBeforeClip
              );

              clipEventIds = clipEvents.map((entry) => {
                const eventId = `${
                  entry.originalAbsoluteMs || entry.adjustedMs
                }_${entry.text}_${entry.source}`;
                return eventId;
              });
              // Per-clip coverage diagnostic vs universe
              try {
                const includedFromUniverse = filterableEventsUniverse.filter(
                  (e) => clipEventIds.includes(mkEventId(e))
                ).length;
                const firstAdj = clipEvents[0]?.adjustedMs;
                const lastAdj = clipEvents[clipEvents.length - 1]?.adjustedMs;
                console.log(
                  `[FILTER-WINDOW][EVENTS] #${currentIndex} eventIndex=${eventIndex} includedFromUniverse=${includedFromUniverse} totalRendered=${clipEvents.length} firstAdj=${firstAdj} lastAdj=${lastAdj}`
                );
              } catch (_) {}

              clipEvents.sort((a, b) => a.adjustedMs - b.adjustedMs);

              if (clipEvents.length > 0) {
                console.log(
                  `DEBUG: clipEvents sample:`,
                  clipEvents.slice(0, 2)
                );
                let previousRelativeMs = null;
                clipEventLog = clipEvents
                  .map((entry, index) => {
                    let relativeMs = Math.max(
                      0,
                      (entry.adjustedMs ?? 0) -
                        (effectiveClipStartRelativeMs ?? 0)
                    );
                    let formattedLine = null;
                    // Special handling for inactivity overlays to avoid duplicates across adjacent clicks
                    if (
                      entry.source === "inactivity-overlay" &&
                      typeof entry.text === "string"
                    ) {
                      const m = entry.text.match(
                        /^\[(\d{2}):(\d{2})\.(\d{3}) to (\d{2}):(\d{2})\.(\d{3})\]: (.+)$/
                      );
                      if (m) {
                        const startMs =
                          Number(m[1]) * 60000 +
                          Number(m[2]) * 1000 +
                          Number(m[3]);
                        const endMs =
                          Number(m[4]) * 60000 +
                          Number(m[5]) * 1000 +
                          Number(m[6]);

                        // If the inactivity overlay extends past this clip, keep it in this clip
                        // and clamp the displayed end to the clip boundary to avoid spillover.
                        const clampedEndMs = Math.min(
                          endMs,
                          filterEndRelativeMs
                        );
                        const adjStart = Math.max(
                          0,
                          startMs - effectiveClipStartRelativeMs
                        );
                        const adjEnd = Math.max(
                          0,
                          clampedEndMs - effectiveClipStartRelativeMs
                        );
                        relativeMs = adjStart;
                        if (USE_RELATIVE_TIME) {
                          const deltaMs =
                            index === 0 || previousRelativeMs == null
                              ? adjStart
                              : adjStart - previousRelativeMs;
                          const attrName =
                            index === 0 || previousRelativeMs == null
                              ? "delta_from_start"
                              : "delta_from_prev";
                          const durationMs = Math.max(0, adjEnd - adjStart);
                          formattedLine = `<event ${attrName}="${formatSecondsDelta(
                            deltaMs
                          )}" duration="${formatSecondsDelta(durationMs)}">${
                            m[7]
                          }</event>`;
                        } else {
                          formattedLine = `<event start="${formatMilliseconds(
                            adjStart
                          )}" end="${formatMilliseconds(adjEnd)}">${
                            m[7]
                          }</event>`;
                        }
                      }
                      // fallback to generic formatting
                    }
                    // Format timestamp relative to the calculated generous window start
                    if (!formattedLine) {
                      const clipRelativeMs =
                        entry.adjustedMs - effectiveClipStartRelativeMs;
                      relativeMs = Math.max(0, clipRelativeMs);
                      if (USE_RELATIVE_TIME) {
                        const deltaMs =
                          index === 0 || previousRelativeMs == null
                            ? relativeMs
                            : relativeMs - previousRelativeMs;
                        const attrName =
                          index === 0 || previousRelativeMs == null
                            ? "delta_from_start"
                            : "delta_from_prev";
                        formattedLine = `<event ${attrName}="${formatSecondsDelta(
                          deltaMs
                        )}">${entry.text}</event>`;
                      } else {
                        formattedLine = `<event time="${formatMilliseconds(
                          relativeMs
                        )}">${entry.text}</event>`;
                      }
                    }
                    if (formattedLine) {
                      previousRelativeMs = relativeMs;
                    }
                    return formattedLine;
                  })
                  .filter(Boolean)
                  .join(USE_RELATIVE_TIME ? "\n\n" : "\n");
              } else {
                console.log(`DEBUG: No clipEvents found after filtering`);
              }
            }

            const shouldAddCurrentPageInstruction = !(
              clipEventLog &&
              (clipEventLog.includes("Page view:") ||
                clipEventLog.includes("Page leave:"))
            );

            // Resolve current page URL robustly from surrounding log entries if missing
            let currentPageResolved = null;
            // Use the same adjusted and processed-clamped window as for event filtering
            // NOTE: clipStartMs/clipEndMs are derived from processedRecording (already adjusted
            // for inactivity skips). Do NOT subtract skip-based shifts again here, or times will
            // be double-shifted and appear earlier than they should (e.g., ~9s instead of ~59s).
            // Use the same baseline/window as filtering (previous end marker -> this clip's end)
            const lookupClipStartRelativeMs =
              effectiveClipStartRelativeMs !== null
                ? effectiveClipStartRelativeMs
                : clipStartRelativeMs;
            let lookupClipEndRelativeMsUnclamped =
              typeof filterEndRelativeMs !== "undefined"
                ? filterEndRelativeMs
                : clipEndMs - videoStartTimestamp;
            // Clamp to processed duration if available
            try {
              if (
                clipPath &&
                typeof getVideoDurationSecondsSync === "function"
              ) {
                const processedCandidate = /_processed\.mp4$/i.test(clipPath)
                  ? clipPath
                  : clipPath.replace(/\.mp4$/i, "_processed.mp4");
                const secs =
                  getVideoDurationSecondsSync(processedCandidate) ??
                  getVideoDurationSecondsSync(clipPath);
                if (Number.isFinite(secs)) {
                  const processedEndRel =
                    lookupClipStartRelativeMs +
                    Math.max(0, Math.round(secs * 1000));
                  lookupClipEndRelativeMsUnclamped = Math.min(
                    lookupClipEndRelativeMsUnclamped,
                    processedEndRel
                  );
                }
              }
            } catch (_) {}
            const lookupClipEndRelativeMs = lookupClipEndRelativeMsUnclamped;
            console.log("DEBUG currentPageResolved.init", {
              lookupClipStartRelativeMs,
              lookupClipEndRelativeMs,
              refinedLogEntriesLength: Array.isArray(refinedLogEntries)
                ? refinedLogEntries.length
                : null,
              shouldAddCurrentPageInstruction,
            });
            if (!currentPageResolved && Array.isArray(refinedLogEntries)) {
              // 1) Most recent non-pre-recording page view before clip start
              console.log(
                "DEBUG currentPageResolved.step1a.start",
                "Scanning backwards for non-pre-recording 'Page view:' before clip start"
              );
              for (let j = refinedLogEntries.length - 1; j >= 0; j--) {
                const entry = refinedLogEntries[j];
                const text = entry && entry.text;
                if (
                  entry.adjustedMs < lookupClipStartRelativeMs &&
                  text &&
                  text.includes("Page view:") &&
                  !text.includes("[PRE-RECORDING (mention")
                ) {
                  const m = text.match(/Page view: (.+)/);
                  if (m && m[1]) {
                    currentPageResolved = m[1];
                    console.log("DEBUG currentPageResolved.step1a.match", {
                      adjustedMs: entry.adjustedMs,
                      text,
                      resolved: currentPageResolved,
                    });
                    break;
                  }
                }
              }
              // 1b) Fallback: allow pre-recording mentions if none found (regardless of timestamp)
              if (!currentPageResolved) {
                console.log(
                  "DEBUG currentPageResolved.step1b.start",
                  "Scanning backwards for pre-recording 'Page view:' regardless of timestamp"
                );
                for (let j = refinedLogEntries.length - 1; j >= 0; j--) {
                  const entry = refinedLogEntries[j];
                  const text = entry && entry.text;
                  if (
                    text &&
                    text.includes("Page view:") &&
                    text.includes("[PRE-RECORDING (mention")
                  ) {
                    const m = text.match(/Page view: (.+)/);
                    if (m && m[1]) {
                      currentPageResolved = m[1];
                      console.log(
                        "DEBUG currentPageResolved.step1b.match.preRecording",
                        {
                          adjustedMs: entry.adjustedMs,
                          text,
                          resolved: currentPageResolved,
                        }
                      );
                      break;
                    }
                  }
                }
              }
            }
            console.log("DEBUG currentPageResolved.final", currentPageResolved);

            // Log the size of the screen state prompt to debug potential fetch errors
            const fpsForPrompt = videoPart?.videoMetadata?.fps ?? 20;
            const msPerFrameForPrompt = Math.round(1000 / fpsForPrompt);

            const rightClickCount = (
              clipEventLog.match(/Right-clicked on/g) || []
            ).length;

            console.log(
              "nggnwgnewkgnwegnwjn ",
              `<task>Create a detailed text version of the rrweb clip that makes watching it unnecessary</task>

<guidelines>
<item>Limit: 10,000 words</item>
<item>Include all visual details so the clip doesn’t need to be watched</item>
<item>Include timestamps</item>
<item>The clip runs at ${fpsForPrompt} FPS (1 frame per ${msPerFrameForPrompt} ms)</item>
<item>${
                rightClickCount === 0
                  ? "The clip contains 1 pointerup event"
                  : `The clip contains 1 pointerup event</item>
<item>The clip contains ${rightClickCount} right click`
              }</item>${
                positionResult.visibleInViewport === true
                  ? `
<item>Based on the clip, decide whether the pointerup event was a click, tap, misclick, end of text selection or something else</item>`
                  : ""
              }
<item>Events can also occur automatically, without user interaction (e.g., browser autofill, auto-scroll,...)</item>
<item>If you are not 100% sure whether an event was user-initiated or automatic, always use the passive voice (i.e., to describe the outcome without naming an actor)</item>
<item>Note that OS-rendered UI (e.g., native scrollbars, browser window controls, context menus, tooltips) isn’t captured, so if the rrweb cursor appears to interact with elements that aren’t visible in the video, interpret those as interactions with the underlying OS/UI components</item>
<item>If you see a screen viewport resize, ignore it</item>
<item>If you see a red line in the rrweb, interpret it as the pointer’s recent path. Never mention "red line" itself in the output.</item>
${
  positionResult.isPointInClaimed === true
    ? "</guidelines>"
    : `<item>If you see a click animation, ignore it</item>
<item>If you see a cursor, do not include its location in the output</item>
</guidelines>`
} 

<log>
${
  clipEventLog ||
  "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"
}
</log>`
            );

            console.log(
              "emwfnwkfnw ",
              `- The clip runs at ${fpsForPrompt} FPS (1 frame per ${msPerFrameForPrompt} ms).`
            );

            console.log("nwefkwefnwe ", positionResult.isPointInClaimed);
            const systemInstruction = `<task>Create a detailed text version of the rrweb clip that makes watching it unnecessary</task>

<guidelines>
<item>Limit: 10,000 words</item>
<item>Include all visual details so the clip doesn't need to be watched</item>
<item>Include timestamps</item>
<item>The clip runs at ${fpsForPrompt} FPS (1 frame per ${msPerFrameForPrompt} ms)</item>
<item>${
              rightClickCount === 0
                ? "The clip contains 1 pointerup event"
                : `The clip contains 1 pointerup event</item>
<item>The clip contains ${rightClickCount} right click`
            }</item>${
              positionResult.visibleInViewport === true
                ? `
<item>Based on the clip, decide whether the pointerup event was a click, tap, misclick, end of text selection or something else</item>`
                : ""
            }
<item>Events can also occur automatically, without user interaction (e.g., browser autofill, auto-scroll,...)</item>
<item>If you are not 100% sure whether an event was user-initiated or automatic, always use the passive voice (i.e., to describe the outcome without naming an actor)</item>
<item>Note that OS-rendered UI (e.g., native scrollbars, browser window controls, context menus, tooltips) isn't captured, so if the rrweb cursor appears to interact with elements that aren't visible in the video, interpret those as interactions with the underlying OS/UI components</item>
<item>If you see a screen viewport resize, ignore it</item>
<item>If you see a red line in the rrweb, interpret it as the pointer's recent path. Never mention "red line" itself in the output.</item>
${
  positionResult.isPointInClaimed === true
    ? "</guidelines>"
    : `<item>If you see a click animation, ignore it</item>
<item>If you see a cursor, do not include its location in the output</item>
</guidelines>`
}`;
            const promptBody = `<task>Create a detailed text version of the rrweb clip that makes watching it unnecessary</task>

<guidelines>
<item>Limit: 10,000 words</item>
<item>Include all visual details so the clip doesn't need to be watched</item>
<item>Include timestamps</item>
<item>The clip runs at ${fpsForPrompt} FPS (1 frame per ${msPerFrameForPrompt} ms)</item>
<item>${
              rightClickCount === 0
                ? "The clip contains 1 pointerup event"
                : `The clip contains 1 pointerup event</item>
<item>The clip contains ${rightClickCount} right click`
            }</item>${
              positionResult.visibleInViewport === true
                ? `
<item>Based on the clip, decide whether the pointerup event was a click, tap, misclick, end of text selection or something else</item>`
                : ""
            }
<item>Events can also occur automatically, without user interaction (e.g., browser autofill, auto-scroll,...)</item>
<item>If you are not 100% sure whether an event was user-initiated or automatic, always use the passive voice (i.e., to describe the outcome without naming an actor)</item>
<item>Note that OS-rendered UI (e.g., native scrollbars, browser window controls, context menus, tooltips) isn't captured, so if the rrweb cursor appears to interact with elements that aren't visible in the video, interpret those as interactions with the underlying OS/UI components</item>
<item>If you see a screen viewport resize, ignore it</item>
<item>If you see a red line in the rrweb, interpret it as the pointer's recent path. Never mention "red line" itself in the output.</item>
${
  positionResult.isPointInClaimed === true
    ? "</guidelines>"
    : `<item>If you see a click animation, ignore it</item>
<item>If you see a cursor, do not include its location in the output</item>
</guidelines>`
}

<log>
${
  clipEventLog ||
  "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"
}
</log>`;
            const clipPromptRequest = {
              model: Gemini,
              config: {
                responseMimeType: "text/plain" /* thinkingConfig: {
              thinkingBudget: 32768,  // Max value for deepest reasoning
              // Alternative: thinkingBudget: -1 for dynamic auto-max
            },  */,
                temperature: 0.2,
                top_p: 0.95,
              },
              systemInstruction,
              contents: createUserContent([videoPart, promptBody]),
            };
            // Ensure prevClipEndtime is defined before use.
            // Always use the end time of the chronologically previous clip (click or noclick).
            // Strategy: find the clip in finalClipsToAnalyze with the greatest end <= current start.
            let prevClipEndtime = "00:00.000";
            try {
              // Determine current click start (seconds)
              let currentClickStartSeconds = null;
              if (typeof clipStartMs === "number") {
                currentClickStartSeconds = Math.max(0, clipStartMs) / 1000;
              } else if (clipPath) {
                const curBase = path.basename(clipPath);
                const mCur = curBase.match(/_from_([0-9]+(?:\.[0-9]+)?)s_/i);
                if (mCur) currentClickStartSeconds = parseFloat(mCur[1]);
              }

              if (
                Number.isFinite(currentClickStartSeconds) &&
                Array.isArray(finalClipsToAnalyze)
              ) {
                let bestEndSeconds = -Infinity;
                for (const t of finalClipsToAnalyze) {
                  if (!t) continue;
                  let endS = null;
                  // Prefer parsing from filename to ensure consistency across both types
                  if (t.clipPath) {
                    const base = path.basename(t.clipPath);
                    const m = base.match(/_to_([0-9]+(?:\.[0-9]+)?)s\.mp4$/i);
                    if (m) endS = parseFloat(m[1]);
                  }
                  if (!Number.isFinite(endS)) {
                    // Fallback to recorded properties for click clips
                    if (typeof t.fileEndSeconds === "number")
                      endS = t.fileEndSeconds;
                  }
                  if (!Number.isFinite(endS)) continue;
                  if (
                    endS <= currentClickStartSeconds &&
                    endS > bestEndSeconds
                  ) {
                    bestEndSeconds = endS;
                  }
                }
                if (bestEndSeconds !== -Infinity) {
                  prevClipEndtime = formatMilliseconds(
                    Math.round(bestEndSeconds * 1000)
                  );
                }
              }
            } catch (_) {
              // Keep default on any unexpected error
            }

            console.log("prevClipEndtime (click)", prevClipEndtime);

            let timestampShiftMs = 0;
            try {
              timestampShiftMs = parseTimecodeMs(prevClipEndtime);
            } catch (_) {
              timestampShiftMs = 0;
            }

            const clipMetadata = {
              eventIndex,
              clipPath,
              clipStartMs,
              clipEndMs,
              clipEventLog,
              clipEventIds,
              currentIndex,
              rightClickCount,
              currentPageResolved,
              shouldAddCurrentPageInstruction,
              timestampShiftMs,
            };

            const shouldBatchRequest =
              GEMINI_BATCH_FOR_CLIP_PROMPTS && !clipBatchResponseMode;

            if (shouldBatchRequest) {
              return {
                type: "batch-request",
                batchRequest: clipPromptRequest,
                metadata: clipMetadata,
              };
            }

            let contextCompletionText = "";
            if (clipBatchResponseMode) {
              if (clipBatchResponseIndex >= clipBatchResponses.length) {
                throw new Error(
                  `[BATCH] Ran out of clip batch responses while processing event ${eventIndex} for session ${session.id}.`
                );
              }
              contextCompletionText =
                clipBatchResponses[clipBatchResponseIndex++]?.trim() || "";
            } else {
              const contextCompletion = await generateContentWithRetry(
                ai,
                clipPromptRequest,
                3,
                apiKeys,
                {
                  sessionId: session?.id,
                  promptLabel: "click-clip-analysis",
                  promptContext: {
                    eventIndex,
                    clipPath,
                    clipStartMs,
                    clipEndMs,
                    currentIndex,
                    rightClickCount,
                    isPointInClaimed: positionResult?.isPointInClaimed ?? null,
                  },
                }
              );

              contextCompletionText = contextCompletion?.text?.trim() || "";
            }

            if (contextCompletionText) {
              return materializeClickAnalysisResult({
                metadata: clipMetadata,
                responseText: contextCompletionText,
              });
            }

            return {
              eventIndex,
              reviewedText: null,
            };
          } catch (error) {
            console.error(
              `  -> ❌ Failed comprehensive analysis for event #${eventIndex}:`,
              error
            );
            throw new Error(
              "Failed comprehensive analysis for event #" + eventIndex,
              " Error: ",
              error
            );
            return {
              eventIndex,
              reviewedText: null,
            };
          }
        }
      );

      const pendingBatchRequests = [];
      analysisResults = [];

      (await Promise.all(analysisPromises)).forEach((result) => {
        if (!result) {
          return;
        }
        if (result.type === "batch-request") {
          pendingBatchRequests.push(result);
        } else {
          analysisResults.push(result);
        }
      });
      const resolvedClickResults = analysisResults.slice();

      // Additionally analyze non-click coverage clips_folder with a tailored prompt
      let nonClickAnalysisResults = [];
      const nonClickClips = finalClipsToAnalyze.filter(
        (task) => task && task.isNonClickSegment
      );
      // NEW: Sort non-click clips_folder by their numeric index to ensure correct order
      nonClickClips.sort((a, b) => {
        const indexA = parseInt(a.event.eventIndex.split("-")[1], 10);
        const indexB = parseInt(b.event.eventIndex.split("-")[1], 10);
        return indexA - indexB;
      });
      if (nonClickClips.length > 0) {
        // NEW: Get all noclick end markers once and sort them
        const allNoclickEndMarkers = refinedLogEntries
          .filter((e) => e.text && e.text.includes("End of noclick clip"))
          .sort((a, b) => a.adjustedMs - b.adjustedMs);

        console.log(
          `[NOCLICK-MARKERS] Found ${allNoclickEndMarkers.length} 'End of noclick clip' markers.`
        );
        allNoclickEndMarkers.forEach((m, i) =>
          console.log(`  -> Marker ${i}: [${m.adjustedMs}ms] "${m.text}"`)
        );
        const nonClickResults = await Promise.all(
          nonClickClips.map(async (clipTask, clipIndex) => {
            try {
              const { clipPath, event } = clipTask;
              const fileName = clipPath ? path.basename(clipPath) : "";
              // Parse start/end seconds from filename pattern: noclick_{j}_from_{start}s_to_{end}s.mp4
              let startSeconds = clipTask?.fileStartSeconds ?? null;
              let endSeconds = clipTask?.fileEndSeconds ?? null;
              const matchSecs =
                fileName &&
                fileName.match(
                  /noclick_\d+_from_([0-9]+(?:\.[0-9]+)?)s_to_([0-9]+(?:\.[0-9]+)?)s\.mp4/
                );
              if (matchSecs) {
                startSeconds = parseFloat(matchSecs[1]);
                endSeconds = parseFloat(matchSecs[2]);
              }
              if (startSeconds == null || endSeconds == null) {
                console.warn(
                  `[NOCLICK] Could not parse start/end seconds from filename: ${fileName}`
                );
                return null;
              }

              // Timestamps from `computeNonClickSegmentsFromMatches` are already on the adjusted timeline and relative to its start.
              // Inclusive start for first non-click segment; for subsequent segments, shift start by +1ms to be exclusive
              const isNotFirstNonClick =
                typeof event.eventIndex === "string" &&
                /^noclick-(\d+)$/.test(event.eventIndex) &&
                parseInt(event.eventIndex.match(/^noclick-(\d+)$/)[1], 10) > 0;
              const clipStartRelativeMs = Math.round(startSeconds * 1000);
              const clipEndRelativeMs = Math.round(endSeconds * 1000);
              if (DEBUG_NOCLICK) {
                console.log(
                  `🔎 [NOCLICK DEBUG] Parsed clip window for ${fileName}: rel[${clipStartRelativeMs}-${clipEndRelativeMs}] duration=${
                    clipEndRelativeMs - clipStartRelativeMs
                  }ms`
                );
              }

              console.log(`[NOCLICK-DEBUG] 1. PARSED FILENAME:`);
              console.log(
                `   -> Parsed Start: ${startSeconds}s (${clipStartRelativeMs}ms)`
              );
              console.log(
                `   -> Parsed End:   ${endSeconds}s (${clipEndRelativeMs}ms)`
              );
              console.log(
                `   -> Type of clipStartRelativeMs: ${typeof clipStartRelativeMs}`
              );

              // Track non-click window for global coverage diagnostics
              try {
                rrwebFilterWindows.push({
                  kind: "noclick",
                  index: event.eventIndex,
                  eventIndex: event.eventIndex,
                  start: Math.max(0, clipStartRelativeMs),
                  end: Math.max(0, clipEndRelativeMs),
                  visualStart: Math.max(0, clipStartRelativeMs),
                  visualEnd: Math.max(0, clipEndRelativeMs),
                  absStartMs: null,
                  absEndMs: null,
                });
              } catch (_) {}

              // Prepare clip content (skip video when gap-based noclick mode)
              let videoPart = null;
              if (!clipTask?.isGapBasedNoClick) {
                const videoBuffer = await fs.readFile(clipPath);
                const videoBase64 = videoBuffer.toString("base64");
                // Use dynamic FPS for non-click clips_folder based on their duration
                const noclickDurationSeconds = endSeconds - startSeconds;
                let { apiFps: noclickFpsForApi } = getRecordingSettings(
                  noclickDurationSeconds
                );

                videoPart = {
                  inlineData: { mimeType: "video/mp4", data: videoBase64 },
                  videoMetadata: { fps: noclickFpsForApi },
                };
              }

              // Build clipEventLog for this window (mirror click-clip logic: dedupe + event classes)
              let excludedReasonCounts = {};
              let candidateCount = 0;

              const allMarkers = refinedLogEntries
                .filter((e) => e.text && e.text.startsWith("--- End of"))
                .sort((a, b) => a.adjustedMs - b.adjustedMs);

              console.log(`[NOCLICK-DEBUG] 2. MARKER DATA:`);
              console.log(
                `   -> Total markers found in session: ${allMarkers.length}`
              );
              if (allMarkers.length > 0) {
                console.log(
                  `   -> First marker: [${allMarkers[0].adjustedMs}ms] "${allMarkers[0].text}"`
                );
                console.log(
                  `   -> Last marker:  [${
                    allMarkers[allMarkers.length - 1].adjustedMs
                  }ms] "${allMarkers[allMarkers.length - 1].text}"`
                );
              }

              // --- START: Corrected Logic ---
              const endMarker = allNoclickEndMarkers[clipIndex];
              let filterEndMs;
              let filterStartMs;

              if (endMarker) {
                filterEndMs = endMarker.adjustedMs;
                const currentMarkerIndex = allMarkers.findIndex(
                  (m) =>
                    m.adjustedMs === endMarker.adjustedMs &&
                    m.text === endMarker.text
                );

                if (currentMarkerIndex > 0) {
                  filterStartMs = allMarkers[currentMarkerIndex - 1].adjustedMs;
                } else if (currentMarkerIndex === 0) {
                  filterStartMs = 0;
                } else {
                  // Fallback for safety
                  const prevMarker = allMarkers
                    .slice()
                    .reverse()
                    .find((m) => m.adjustedMs < clipStartRelativeMs);
                  filterStartMs = prevMarker ? prevMarker.adjustedMs : 0;
                }
              } else if (clipIndex === nonClickClips.length - 1) {
                const sessionVideoEndTimestamp =
                  processedRecording[processedRecording.length - 1].timestamp;
                const sessionVideoStartTimestamp =
                  processedRecording[0].timestamp;
                filterEndMs =
                  sessionVideoEndTimestamp - sessionVideoStartTimestamp;
                // For the last segment, use the explicit clip start time
                    // For the last segment, start from the last marker
                filterStartMs =
                  allMarkers.length > 0
                    ? allMarkers[allMarkers.length - 1].adjustedMs
                    : 0;
              } else {
                console.error(
                  `[NOCLICK-ERROR] No end marker for non-click clip #${clipIndex} (and it's not the last one). There are ${allNoclickEndMarkers.length} markers for ${nonClickClips.length} clips_folder. Skipping.`
                );
                return null; // Skip this clip
              }
              // --- END: Corrected Logic ---

              // If there was no previous marker, bound the start at the clip start.
              const effectiveFilterStartMs = filterStartMs;
              console.log(`[NOCLICK-DEBUG] 6. EVENT FILTERING WINDOW:`);
              console.log(`   -> Start: ${effectiveFilterStartMs}ms`);
              console.log(`   -> End:   ${filterEndMs}ms`);

              let clipEvents = refinedLogEntries.filter((entry) => {
                if (
                  !entry.text ||
                  entry.isPreRecording ||
                  entry.isReplayMarker
                ) {
                  return false;
                }

                if (entry.text.includes("-- End of ")) {
                  return false;
                }

                const inWindow =
                  entry.adjustedMs >= effectiveFilterStartMs &&
                  entry.adjustedMs <= filterEndMs;

                let isSpanningOverlay = false;
                if (
                  entry.source === "inactivity-overlay" &&
                  typeof entry.text === "string"
                ) {
                  const m = entry.text.match(
                    /^\[(\d{2}):(\d{2})\.(\d{3}) to (\d{2}):(\d{2})\.(\d{3})\]: (.+)$/
                  );
                  if (m) {
                    const startMs =
                      Number(m[1]) * 60000 + Number(m[2]) * 1000 + Number(m[3]);
                    const endMs =
                      Number(m[4]) * 60000 + Number(m[5]) * 1000 + Number(m[6]);
                    if (startMs < filterEndMs && endMs > filterStartMs) {
                      isSpanningOverlay = true;
                    }
                  }
                }

                if (entry.text.includes("/segment>")) {
                  return false;
                }

                // For noclick clips_folder, filter out anything that looks like a click.
                if (entry.text.startsWith("Clicked ")) {
                  return false;
                }

                if (entry.text.startsWith("Event: ")) {
                  return false;
                }

                return inWindow || isSpanningOverlay;
              });

              clipEvents.sort((a, b) => a.adjustedMs - b.adjustedMs);

              if (DEBUG_NOCLICK) {
                const srcCounts = {};
                for (const e of clipEvents) {
                  srcCounts[e.source || "unknown"] =
                    (srcCounts[e.source || "unknown"] || 0) + 1;
                }
                console.log(
                  `🔎 [NOCLICK DEBUG] Selected ${clipEvents.length} events (candidates=${candidateCount}) for ${fileName}. Source breakdown:`,
                  srcCounts
                );
                console.log(
                  `🔎 [NOCLICK DEBUG] Excluded counts by reason:`,
                  excludedReasonCounts
                );
                if (clipEvents.length > 0) {
                  console.log(
                    `🔎 [NOCLICK DEBUG] First 3 included:`,
                    clipEvents.slice(0, 3).map((e) => ({
                      t: e.adjustedMs,
                      text: e.text,
                      src: e.source,
                    }))
                  );
                } else {
                  // Show a few in-window entries even if excluded (to debug conditions)
                  const inWindow = refinedLogEntries
                    .filter(
                      (e) =>
                        e.adjustedMs >= clipStartRelativeMs &&
                        e.adjustedMs <= clipEndRelativeMs
                    )
                    .slice(0, 5)
                    .map((e) => ({
                      t: e.adjustedMs,
                      text: e.text,
                      src: e.source,
                      pre: e.isPreRecording,
                    }));
                  console.log(
                    `🔎 [NOCLICK DEBUG] In-window sample (first 5):`,
                    inWindow
                  );
                }
              }

              // Insert inferred page leaves based on the most recent page view before the clip start (parity with click clips_folder)
              let pageUrlBeforeClip = null;
              for (let j = refinedLogEntries.length - 1; j >= 0; j--) {
                const entry = refinedLogEntries[j];
                if (
                  entry.adjustedMs < clipStartRelativeMs &&
                  entry.text &&
                  entry.text.includes("Page view:")
                ) {
                  const urlMatch = entry.text.match(/Page view: (.+)/);
                  if (urlMatch && urlMatch[1]) {
                    pageUrlBeforeClip = urlMatch[1];
                    break;
                  }
                }
              }
              clipEvents = insertInferredPageLeaves(
                clipEvents,
                pageUrlBeforeClip
              );

              const clipEventIds = clipEvents.map((entry) => {
                const eventId = `${
                  entry.originalAbsoluteMs || entry.adjustedMs
                }_${entry.text}_${entry.source}`;
                return eventId;
              });

              let clipEventLog;
              if (clipEvents.length > 0) {
                let previousRelativeMs = null;
                clipEventLog = clipEvents
                  .map((entry, index) => {
                    let relativeMs = Math.max(
                      0,
                      (entry.adjustedMs ?? 0) - (effectiveFilterStartMs ?? 0)
                    );
                    let formattedLine = null;
                    // Special handling for inactivity overlays to avoid double timestamps
                    if (
                      entry.source === "inactivity-overlay" &&
                      typeof entry.text === "string"
                    ) {
                      const m = entry.text.match(
                        /^\[(\d{2}):(\d{2})\.(\d{3}) to (\d{2}):(\d{2})\.(\d{3})\]: (.+)$/
                      );
                      if (m) {
                        const startMs =
                          Number(m[1]) * 60000 +
                          Number(m[2]) * 1000 +
                          Number(m[3]);
                        const endMs =
                          Number(m[4]) * 60000 +
                          Number(m[5]) * 1000 +
                          Number(m[6]);

                        // If the inactivity overlay extends past this clip, keep it here
                        // and clamp the displayed end to the clip boundary.
                        const clampedEndMs = Math.min(endMs, filterEndMs);
                        const adjStart = Math.max(
                          0,
                          startMs - effectiveFilterStartMs
                        );
                        const adjEnd = Math.max(
                          0,
                          clampedEndMs - effectiveFilterStartMs
                        );
                        relativeMs = adjStart;
                        if (USE_RELATIVE_TIME) {
                          const deltaMs =
                            index === 0 || previousRelativeMs == null
                              ? adjStart
                              : adjStart - previousRelativeMs;
                          const attrName =
                            index === 0 || previousRelativeMs == null
                              ? "delta_from_start"
                              : "delta_from_prev";
                          const durationMs = Math.max(0, adjEnd - adjStart);
                          formattedLine = `<event ${attrName}="${formatSecondsDelta(
                            deltaMs
                          )}" duration="${formatSecondsDelta(durationMs)}">${
                            m[7]
                          }</event>`;
                        } else {
                          formattedLine = `<event start="${formatMilliseconds(
                            adjStart
                          )}" end="${formatMilliseconds(adjEnd)}">${
                            m[7]
                          }</event>`;
                        }
                      }
                      // fallback
                    }
                    if (!formattedLine) {
                      const rel = entry.adjustedMs - effectiveFilterStartMs;
                      relativeMs = Math.max(0, rel);
                      if (USE_RELATIVE_TIME) {
                        const deltaMs =
                          index === 0 || previousRelativeMs == null
                            ? relativeMs
                            : relativeMs - previousRelativeMs;
                        const attrName =
                          index === 0 || previousRelativeMs == null
                            ? "delta_from_start"
                            : "delta_from_prev";
                        formattedLine = `<event ${attrName}="${formatSecondsDelta(
                          deltaMs
                        )}">${entry.text}</event>`;
                      } else {
                        formattedLine = `<event time="${formatMilliseconds(
                          relativeMs
                        )}">${entry.text}</event>`;
                      }
                    }
                    if (formattedLine) {
                      previousRelativeMs = relativeMs;
                    }
                    return formattedLine;
                  })
                  .filter(Boolean)
                  .join(USE_RELATIVE_TIME ? "\n\n" : "\n");
              } else {
                clipEventLog =
                  "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded";
              }

              console.log("clipEventLogclipEventLog ", clipEventLog);

              // Run AI analysis for non-click clip
              const fpsForPrompt = videoPart?.videoMetadata?.fps ?? 20;
              const msPerFrameForPrompt = Math.round(1000 / fpsForPrompt);

              const shouldAddCurrentPageInstruction = !(
                clipEventLog &&
                (clipEventLog.includes("Page view:") ||
                  clipEventLog.includes("Page leave:"))
              );

              // Resolve current page URL robustly from surrounding log entries if missing
              let currentPageResolved = null;
              const lookupClipStartRelativeMs = clipStartRelativeMs;
              const lookupClipEndRelativeMs = clipEndRelativeMs;
              console.log("DEBUG currentPageResolved.init", {
                lookupClipStartRelativeMs,
                lookupClipEndRelativeMs,
                refinedLogEntriesLength: Array.isArray(refinedLogEntries)
                  ? refinedLogEntries.length
                  : null,
                shouldAddCurrentPageInstruction,
              });
              if (!currentPageResolved && Array.isArray(refinedLogEntries)) {
                // 1) Most recent non-pre-recording page view before clip start
                console.log(
                  "DEBUG currentPageResolved.step1a.start",
                  "Scanning backwards for non-pre-recording 'Page view:' before clip start"
                );
                for (let j = refinedLogEntries.length - 1; j >= 0; j--) {
                  const entry = refinedLogEntries[j];
                  const text = entry && entry.text;
                  if (
                    entry.adjustedMs < lookupClipStartRelativeMs &&
                    text &&
                    text.includes("Page view:") &&
                    !text.includes("[PRE-RECORDING (mention")
                  ) {
                    const m = text.match(/Page view: (.+)/);
                    if (m && m[1]) {
                      currentPageResolved = m[1];
                      console.log("DEBUG currentPageResolved.step1a.match", {
                        adjustedMs: entry.adjustedMs,
                        text,
                        resolved: currentPageResolved,
                      });
                      break;
                    }
                  }
                }
                // 1b) Fallback: allow pre-recording mentions if none found (regardless of timestamp)
                if (!currentPageResolved) {
                  console.log(
                    "DEBUG currentPageResolved.step1b.start",
                    "Scanning backwards for pre-recording 'Page view:' regardless of timestamp"
                  );
                  for (let j = refinedLogEntries.length - 1; j >= 0; j--) {
                    const entry = refinedLogEntries[j];
                    const text = entry && entry.text;
                    if (
                      text &&
                      text.includes("Page view:") &&
                      text.includes("[PRE-RECORDING (mention")
                    ) {
                      const m = text.match(/Page view: (.+)/);
                      if (m && m[1]) {
                        currentPageResolved = m[1];
                        console.log(
                          "DEBUG currentPageResolved.step1b.match.preRecording",
                          {
                            adjustedMs: entry.adjustedMs,
                            text,
                            resolved: currentPageResolved,
                          }
                        );
                        break;
                      }
                    }
                  }
                }
              }
              console.log(
                "DEBUG currentPageResolved.final",
                currentPageResolved
              );

              const rightClickCount = (
                clipEventLog.match(/Right-clicked on/g) || []
              ).length;
              console.log(
                "Fwfwfwefwfw ",
                clipEventLog,
                clipEventLog.includes("Right-clicked on"),
                "count:",
                rightClickCount
              );
              // If INCLUDE_NOCLICK_CLIPS is false, skip AI and return the raw clip logs inside </segment>
              if (!INCLUDE_NOCLICK_CLIPS) { 
                let normalizedClipEventLog = clipEventLog;
                const shiftOffsetMs =
                  typeof clipStartRelativeMs === "number" &&
                  Number.isFinite(clipStartRelativeMs) &&
                  clipStartRelativeMs > 0
                    ? clipStartRelativeMs
                    : typeof effectiveFilterStartMs === "number" &&
                      Number.isFinite(effectiveFilterStartMs) &&
                      effectiveFilterStartMs > 0
                    ? effectiveFilterStartMs
                    : 0;
                if (
                  normalizedClipEventLog &&
                  typeof normalizedClipEventLog === "string" &&
                  shiftOffsetMs > 0
                ) {
                  normalizedClipEventLog = shiftTimestampsInText(
                    normalizedClipEventLog,
                    shiftOffsetMs
                  );
                }

                const header = currentPageResolved
                  ? shouldAddCurrentPageInstruction
                    ? `<current_page>${currentPageResolved}</current_page>\n\n`
                    : `<initial_page>${currentPageResolved}</initial_page>\n\n`
                  : "";
                const reviewedText = `\n${header}${
                  normalizedClipEventLog ||
                  "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"
                }\n</segment>`;
                return {
                  eventIndex: event?.eventIndex,
                  reviewedText,
                  contextDescription: null,
                  clipStartMs: clipStartRelativeMs,
                  clipEndMs: clipEndRelativeMs,
                  clipEventLog: normalizedClipEventLog,
                  clipEventIds,
                };
              }

              console.log(
                "nggnwgnewkgnwegnwjn ",
                `<task>Create a detailed text version of the rrweb clip that makes watching it unnecessary</task>

<guidelines>
<item>Limit: 10,000 words</item>
<item>Include all visual details so the clip doesn’t need to be watched</item>
<item>Include timestamps</item>
<item>The clip runs at ${fpsForPrompt} FPS (1 frame per ${msPerFrameForPrompt} ms)</item>
<item>${
                  rightClickCount === 0
                    ? "The clip contains 0 left/right clicks/taps."
                    : `The clip contains ${rightClickCount} right click`
                }</item>
<item>Do not invent clicks or interactions that aren't present in the log.</item>
<item>Events can also occur automatically, without user interaction (e.g., browser autofill, auto-scroll,...)</item>
<item>If you are not 100% sure whether an event was user-initiated or automatic, always use the passive voice (i.e., to describe the outcome without naming an actor)</item>
<item>Note that OS-rendered UI (e.g., native scrollbars, browser window controls, context menus, tooltips) isn’t captured, so if the rrweb cursor appears to interact with elements that aren’t visible in the video, interpret those as interactions with the underlying OS/UI components</item>
<item>If you see a screen viewport resize, ignore it</item>
<item>If you see a red line in the rrweb, interpret it as the pointer’s recent path. Never mention "red line" itself in the output.</item>
</guidelines>

<log>
${
  clipEventLog ||
  "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"
}
</log>`
              );

              const contextCompletion = await generateContentWithRetry(
                ai,
                {
                  model: Gemini,
                  config: {
                    responseMimeType: "text/plain" /* thinkingConfig: {
              thinkingBudget: 32768,  // Max value for deepest reasoning
              // Alternative: thinkingBudget: -1 for dynamic auto-max
            },  */,
                    temperature: 0.2,
                    top_p: 0.95,
                  },
                  systemInstruction: `<task>Create a detailed text version of the rrweb clip that makes watching it unnecessary</task>

<guidelines>
<item>Limit: 10,000 words</item>
<item>Include all visual details so the clip doesn’t need to be watched</item>
<item>Include timestamps</item>
<item>The clip runs at ${fpsForPrompt} FPS (1 frame per ${msPerFrameForPrompt} ms)</item>
<item>${
                    rightClickCount === 0
                      ? "The clip contains 0 left/right clicks/taps."
                      : `The clip contains ${rightClickCount} right click`
                  }</item>
<item>Do not invent clicks or interactions that aren't present in the log.</item>
<item>Events can also occur automatically, without user interaction (e.g., browser autofill, auto-scroll,...)</item>
<item>If you are not 100% sure whether an event was user-initiated or automatic, always use the passive voice (i.e., to describe the outcome without naming an actor)</item>
<item>Note that OS-rendered UI (e.g., native scrollbars, browser window controls, context menus, tooltips) isn’t captured, so if the rrweb cursor appears to interact with elements that aren’t visible in the video, interpret those as interactions with the underlying OS/UI components</item>
<item>If you see a screen viewport resize, ignore it</item>
<item>If you see a red line in the rrweb, interpret it as the pointer’s recent path. Never mention "red line" itself in the output.</item>
</guidelines>

<log>
${
  clipEventLog ||
  "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"
}
</log>`,
                  contents: createUserContent(
                    [
                    videoPart,
                    `<task>Create a detailed text version of the rrweb clip that makes watching it unnecessary</task>

<guidelines>
<item>Limit: 10,000 words</item>
<item>Include all visual details so the clip doesn’t need to be watched</item>
<item>Include timestamps</item>
<item>The clip runs at ${fpsForPrompt} FPS (1 frame per ${msPerFrameForPrompt} ms)</item>
<item>${
                      rightClickCount === 0
                        ? "The clip contains 0 left/right clicks/taps."
                        : `The clip contains ${rightClickCount} right click`
                    }</item>
<item>Do not invent clicks or interactions that aren't present in the log.</item>
<item>Events can also occur automatically, without user interaction (e.g., browser autofill, auto-scroll,...)</item>
<item>If you are not 100% sure whether an event was user-initiated or automatic, always use the passive voice (i.e., to describe the outcome without naming an actor)</item>
<item>Note that OS-rendered UI (e.g., native scrollbars, browser window controls, context menus, tooltips) isn’t captured, so if the rrweb cursor appears to interact with elements that aren’t visible in the video, interpret those as interactions with the underlying OS/UI components</item>
<item>If you see a screen viewport resize, ignore it</item>
<item>If you see a red line in the rrweb, interpret it as the pointer’s recent path. Never mention "red line" itself in the output.</item>
</guidelines>

<log>
${
  clipEventLog ||
  "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"
}
</log>`,
                    ].filter(Boolean)
                  ),
                },
                3,
                apiKeys,
                {
                  sessionId: session?.id,
                  promptLabel: "non-click-clip-analysis",
                  promptContext: {
                    eventIndex: event?.eventIndex,
                    clipPath: clipTask?.clipPath,
                    clipType: "noclick",
                    clipStartMs: clipStartRelativeMs,
                    clipEndMs: clipEndRelativeMs,
                    rightClickCount,
                  },
                }
              );

              let contextDescription = contextCompletion?.text?.trim() || "";

              // Ensure prevClipEndtime is defined before use.
              // Always use the end time of the chronologically previous clip (click or noclick).
              // Strategy: from finalClipsToAnalyze, find latest task whose end <= this noclick's start.
              let prevClipEndtime = "00:00.000";
              try {
                const thisNoclickStartSeconds = Number.isFinite(startSeconds)
                  ? startSeconds
                  : typeof clipStartRelativeMs === "number"
                  ? clipStartRelativeMs / 1000
                  : null;
                if (
                  Number.isFinite(thisNoclickStartSeconds) &&
                  Array.isArray(finalClipsToAnalyze)
                ) {
                  let bestEndSeconds = -Infinity;
                  for (const t of finalClipsToAnalyze) {
                    if (!t) continue;
                    let endS = null;
                    if (t.clipPath) {
                      const base = path.basename(t.clipPath);
                      const m = base.match(/_to_([0-9]+(?:\.[0-9]+)?)s\.mp4$/i);
                      if (m) endS = parseFloat(m[1]);
                    }
                    if (!Number.isFinite(endS)) {
                      if (typeof t.fileEndSeconds === "number")
                        endS = t.fileEndSeconds;
                    }
                    if (!Number.isFinite(endS)) continue;
                    if (
                      endS <= thisNoclickStartSeconds &&
                      endS > bestEndSeconds
                    ) {
                      bestEndSeconds = endS;
                    }
                  }
                  if (bestEndSeconds !== -Infinity) {
                    prevClipEndtime = formatMilliseconds(
                      Math.round(bestEndSeconds * 1000)
                    );
                  }
                }
              } catch (_) {
                // Keep default on any unexpected error
              }

              if (prevClipEndtime === "00:00.000") {
                contextDescription = contextCompletion?.text?.trim() || "";
              } else {
                console.log("prevClipEndtimeprevClipEndtime ", prevClipEndtime);
                try {
                  const offsetMs = parseTimecodeMs(prevClipEndtime);
                  contextDescription = shiftTimestampsInText(
                    contextDescription || "",
                    offsetMs
                  );
                } catch (_) {
                  // Fallback: keep original text if normalization fails
                }
              }

              const reviewedText = `\n${
                currentPageResolved
                  ? shouldAddCurrentPageInstruction
                    ? `<current_page>${currentPageResolved}</current_page>\n\n`
                    : `<initial_page>${currentPageResolved}</initial_page>\n\n`
                  : ""
              }${contextDescription}\n</segment>`;
              if (DEBUG_NOCLICK) {
                console.log(
                  `🔎 [NOCLICK DEBUG] AI context length=${contextDescription.length} for ${fileName}`
                );
              }

              return {
                eventIndex: event.eventIndex,
                reviewedText,
                contextDescription: contextDescription || null,
                clipStartMs: clipStartRelativeMs,
                clipEndMs: clipEndRelativeMs,
                clipEventLog,
                clipEventIds,
              };
            } catch (e) {
              console.error(
                "[NOCLICK] Analysis failed for ",
                clipTask?.clipPath,
                e
              );
              return null;
            }
          })
        );

        // Merge non-null non-click results
        nonClickAnalysisResults = nonClickResults.filter(Boolean);
        analysisResults = analysisResults.concat(nonClickAnalysisResults);
      }

      const clipPositionResults = Array.isArray(clickPositionResults)
        ? clickPositionResults
        : [];
      const pendingClickRequestMetadata = pendingBatchRequests
        .filter((request) => request?.batchRequest)
        .map((request) => request.metadata)
        .filter(Boolean);

      sessionContextSnapshot = {
        version: BATCH_CONTEXT_VERSION,
        createdAt: new Date().toISOString(),
        sessionId: session.id,
        analysisId,
        meta: {
          startUrl,
          serverStartUrl,
          recording_duration,
          click_count,
          datetime,
          processedAt,
          end_time,
          person_id,
          browser,
          osVar,
          osVersion,
          device_type,
          country,
          referrer,
          idVar,
          user_agent,
          allPages,
          lastPage,
        },
        logs: {
          refinedLogEntries,
          preRecordingEvents,
        },
        clips: {
          pendingClickRequests: pendingClickRequestMetadata,
          clickResults: resolvedClickResults,
          nonClickResults: nonClickAnalysisResults,
          finalClipsToAnalyze,
          clickClipsToAnalyze,
        },
        rrwebMatches,
        rrwebFilterWindows,
        processedRecording,
        clipPositionResults,
        clipLogsFilePath,
        options: {
          INCLUDE_CLIP_LOG_PROMPT,
          INCLUDE_ENTIRE_LOG_PROMPT,
        },
      };

      if (clipBatchResponseMode) {
        if (pendingBatchRequests.length > 0) {
          throw new Error(
            `[BATCH] Replay mode detected but still found ${pendingBatchRequests.length} batch-request placeholders for session ${session.id}.`
          );
        }
        if (clipBatchResponseIndex !== clipBatchResponses.length) {
          console.warn(
            `[BATCH] Consumed ${clipBatchResponseIndex} of ${clipBatchResponses.length} clip batch responses for session ${session.id}.`
          );
        }
      } else if (GEMINI_BATCH_FOR_CLIP_PROMPTS) {
        const clipBatchRequests = pendingBatchRequests
          .map((result) => result?.batchRequest)
          .filter((request) => request && request.contents)
          .map((request) => ({
            model: request.model || Gemini,
            contents: request.contents,
            config: request.config,
            systemInstruction: request.systemInstruction,
          }));

        if (clipBatchRequests.length > 0) {
          const { batchJob } = await createGeminiBatchJob({
            inlinedRequests: clipBatchRequests,
            sessionId: session?.id,
            analysisId,
            displayNameHint: "clip-analysis",
          });
          const batchJobName = batchJob?.name;
          if (!batchJobName) {
            throw new Error(
              "Gemini batch job did not return a job name to persist in the database."
            );
          }

          let analysisContentValue = batchJobName;
          let contextUri = null;

          if (
            Array.isArray(sessionContextSnapshot?.clips?.pendingClickRequests) &&
            sessionContextSnapshot.clips.pendingClickRequests.length > 0
          ) {
            try {
              const payloadToPersist = {
                ...sessionContextSnapshot,
                mode: "GEMINI_BATCH",
                batchJobName,
              };
              contextUri = await saveBatchContext({
                storage,
                bucketName,
                sessionId: session.id,
                analysisId,
                payload: payloadToPersist,
              });
              analysisContentValue = JSON.stringify({
                mode: "GEMINI_BATCH",
                version: BATCH_CONTEXT_VERSION,
                batchJob: batchJobName,
                contextUri,
                createdAt: new Date().toISOString(),
              });
              console.log(
                `[BATCH] Saved context for session ${session.id} at ${contextUri}.`
              );
            } catch (contextError) {
              console.error(
                `[BATCH] Failed to persist context for session ${session.id}:`,
                contextError?.message || contextError
              );
              contextUri = null;
              analysisContentValue = batchJobName;
            }
          }

          await pool.query(
            `
              UPDATE sessionanalysis
              SET analysiscontent = $1
              WHERE id = $2;
            `,
            [analysisContentValue, analysisId]
          );

          console.log(
            `Queued Gemini batch job ${batchJobName} for clip analysis in session ${session.id}; analysis remains in PROCESSING until job completion.`
          );

          analysisDeferred = true;

          return;
        } else if (pendingBatchRequests.length > 0) {
          console.warn(
            "[BATCH] Enabled GEMINI_BATCH_FOR_CLIP_PROMPTS but no requests were generated; continuing with synchronous flow."
          );
        }
      }

    } else {
      const clipPositionResults = Array.isArray(clickPositionResults)
        ? clickPositionResults
        : [];

      sessionContextSnapshot = {
        version: BATCH_CONTEXT_VERSION,
        createdAt: new Date().toISOString(),
        sessionId: session.id,
        analysisId,
        meta: {
          startUrl,
          serverStartUrl,
          recording_duration,
          click_count,
          datetime,
          processedAt,
          end_time,
          person_id,
          browser,
          osVar,
          osVersion,
          device_type,
          country,
          referrer,
          idVar,
          user_agent,
          allPages,
          lastPage,
        },
        logs: {
          refinedLogEntries,
          preRecordingEvents,
        },
        clips: {
          pendingClickRequests: [],
          clickResults: [],
          nonClickResults: [],
          finalClipsToAnalyze,
          clickClipsToAnalyze,
        },
        rrwebMatches,
        rrwebFilterWindows,
        processedRecording,
        clipPositionResults,
        clipLogsFilePath,
        options: {
          INCLUDE_CLIP_LOG_PROMPT,
          INCLUDE_ENTIRE_LOG_PROMPT,
        },
      };
    }

      if (!sessionContextSnapshot) {
        throw new Error(
          `[FINALIZE] Missing session context snapshot for session ${session.id}.`
        );
      }

      await finalizeSessionFromContext({
        context: sessionContextSnapshot,
        session,
        user,
        analysisId,
        clickPositionResults: clickPositionResults
      });
      finalizationCompleted = true;

  } catch (error) {
    sessionError = error;
    console.error(
      `An error occurred while processing session ${session.id} for user ${user.email}:`,
      error
    );
    // . If an error occurred, update the status to FAILED ---
    if (analysisId) {
      try {
        const failQuery = `
        UPDATE sessionanalysis
        SET status = 'FAILED', analysiscontent = $1, processedat = $3
        WHERE id = $2;
    `;
        await pool.query(failQuery, [
          error.message,
          analysisId,
          new Date("1980-07-07"),
        ]);
        console.log(`Marked analysis ${analysisId} as FAILED in the database.`);
      } catch (dbError) {
        console.error(
          `Failed to update analysis status to FAILED for ID ${analysisId}:`,
          dbError
        );
      }
    }
    throw error;
  } finally {
    try {
      await writeSessionCostReport(session.id, user?.id, {
        status: sessionError ? "FAILED" : "COMPLETED",
        errorMessage: sessionError
          ? sessionError?.stack || sessionError?.message || String(sessionError)
          : null,
      });
    } catch (costReportError) {
      console.error(
        `[COST-TRACKING] Failed to write cost report for session ${session?.id}:`,
        costReportError
      );
    }
  }
}

/**
 * Removes DocumentType (type 1) nodes from rrweb full snapshot events to prevent
 * "Only one doctype on document allowed" errors, while preserving the main Document node.
 * This version includes extensive logging for debugging.
 * @param {Array} events - An array of rrweb events.
 * @returns {Array} A new array of rrweb events with doctype nodes removed.
 */
async function finalizeSessionFromContext({
  context,
  session,
  user,
  analysisId,
  clickPositionResults,
  geminiResponses = null,
  contextUri = null,
} = {}) {
  if (!context || typeof context !== "object") {
    throw new Error("[FINALIZE] Missing context payload for session finalization.");
  }
  if (!session || !session.id) {
    throw new Error("[FINALIZE] Missing session metadata for finalization.");
  }

  const options = context.options || {};
  const includeClipPrompt =
    typeof options.INCLUDE_CLIP_LOG_PROMPT === "boolean"
      ? options.INCLUDE_CLIP_LOG_PROMPT
      : INCLUDE_CLIP_LOG_PROMPT;
  const includeEntireLogPrompt =
    typeof options.INCLUDE_ENTIRE_LOG_PROMPT === "boolean"
      ? options.INCLUDE_ENTIRE_LOG_PROMPT
      : INCLUDE_ENTIRE_LOG_PROMPT;

  const meta = context.meta || {};
  const logs = context.logs || {};
  const clipState = context.clips || {};

  let refinedLogEntries = Array.isArray(logs.refinedLogEntries)
    ? logs.refinedLogEntries
    : [];
  const preRecordingEvents = Array.isArray(logs.preRecordingEvents)
    ? logs.preRecordingEvents
    : [];

  const rrwebFilterWindows = Array.isArray(context.rrwebFilterWindows)
    ? context.rrwebFilterWindows
    : [];
  const rrwebMatches = Array.isArray(context.rrwebMatches)
    ? context.rrwebMatches
    : [];
  const processedRecording = Array.isArray(context.processedRecording)
    ? context.processedRecording
    : [];
  const clipPositionResults = Array.isArray(context.clipPositionResults)
    ? context.clipPositionResults
    : [];
  const clipLogsFilePath = context.clipLogsFilePath;

  const finalClipsToAnalyze = Array.isArray(clipState.finalClipsToAnalyze)
    ? clipState.finalClipsToAnalyze
    : [];
  const clickClipsToAnalyze = Array.isArray(clipState.clickClipsToAnalyze)
    ? clipState.clickClipsToAnalyze
    : [];
  const pendingClickRequests = Array.isArray(
    clipState.pendingClickRequests
  )
    ? clipState.pendingClickRequests.filter(Boolean)
    : [];
  const storedClickResults = Array.isArray(clipState.clickResults)
    ? clipState.clickResults.filter(Boolean)
    : [];
  const storedNonClickResults = Array.isArray(clipState.nonClickResults)
    ? clipState.nonClickResults.filter(Boolean)
    : [];

  const clipsDir = inferClipsDir(finalClipsToAnalyze, clipLogsFilePath);
  const filterableEventsUniverse = refinedLogEntries.filter(isEventFilterable);

  let analysisResults = [];
  let clipEventLogs = "";

  const {
    startUrl = "N/A",
    serverStartUrl = null,
    recording_duration = "N/A",
    click_count = "N/A",
    datetime = "N/A",
    processedAt = null,
    end_time = "N/A",
    person_id = "N/A",
    browser = "N/A",
    osVar = "N/A",
    osVersion = "N/A",
    device_type = "N/A",
    country = "N/A",
    referrer = "N/A",
    idVar = "N/A",
    user_agent = "",
    allPages = [],
    lastPage = "N/A",
  } = meta;

  if (includeClipPrompt) {
    let clickResults = storedClickResults;

    if (Array.isArray(geminiResponses) && geminiResponses.length > 0) {
      if (pendingClickRequests.length !== geminiResponses.length) {
        const matchedCount = Math.min(
          pendingClickRequests.length,
          geminiResponses.length
        );
        console.warn(
          `[FINALIZE] Mismatch between pending click requests (${pendingClickRequests.length}) and Gemini responses (${geminiResponses.length}) for session ${session.id}. Proceeding with ${matchedCount} matched items and skipping the rest.`
        );
        clickResults = pendingClickRequests.slice(0, matchedCount).map((metadata, index) =>
          materializeClickAnalysisResult({
            metadata,
            responseText: geminiResponses[index],
          })
        );
      } else {
        clickResults = pendingClickRequests.map((metadata, index) =>
          materializeClickAnalysisResult({
            metadata,
            responseText: geminiResponses[index],
          })
        );
      }
    } else if (!clickResults.length && pendingClickRequests.length > 0) {
      console.warn(
        `[FINALIZE] No Gemini responses provided for ${pendingClickRequests.length} pending click analyses in session ${session.id}. Continuing without these click analyses.`
      );
    }

    analysisResults = [...clickResults, ...storedNonClickResults].filter(Boolean);

    if (clipLogsFilePath) {
      try {
        // Ensure the directory exists before writing, across platforms
        const clipLogsDir = path.dirname(clipLogsFilePath);
        await fs.mkdir(clipLogsDir, { recursive: true });

        await fs.writeFile(
          clipLogsFilePath,
          JSON.stringify(analysisResults, null, 2),
          "utf-8"
        );
        console.log(`?'? Saved clip logs to ${clipLogsFilePath}`);
      } catch (err) {
        console.warn(
          `??????  Failed to save clip logs to ${clipLogsFilePath}:`,
          err
        );
      }
    }
  } else if (clipLogsFilePath) {
    try {
      const data = await fs.readFile(clipLogsFilePath, "utf-8");
      analysisResults = JSON.parse(data);
      console.log(
        `?? Loaded clip logs from ${clipLogsFilePath} (prompts skipped)`
      );
    } catch (err) {
      console.error(
        `??? Failed to load clip logs from ${clipLogsFilePath}`,
        err
      );
      analysisResults = [];
    }
  } else {
    analysisResults = [];
  }

  const allClipEventIds = new Set();
      // Collect all event IDs that were included in any clip log
      analysisResults.forEach((result) => {
        if (result && result.clipEventIds) {
          result.clipEventIds.forEach((id) => allClipEventIds.add(id));
        }
      });
  
      console.log(
        `\n🔍 DEBUG: Total events included in clip logs: ${allClipEventIds.size}`
      );
  
      // Global coverage diagnostics: verify adjacency, overlaps, and leftovers
      try {
        // 1) Adjacency/overlap of rrweb-aligned windows (click-only)
        const clickWindows = rrwebFilterWindows
          .filter((w) => w.kind === "click")
          .sort((a, b) => a.start - b.start);
        let clickOverlaps = 0;
        let clickGaps = 0;
        let clickGapTotalMs = 0;
        for (let i = 0; i < clickWindows.length - 1; i++) {
          const cur = clickWindows[i];
          const nxt = clickWindows[i + 1];
          const delta = nxt.start - cur.end;
          if (delta < 0) {
            clickOverlaps += 1;
            console.log(
              `[FILTER-COVERAGE][OVERLAP][CLICK] #${i}->#${
                i + 1
              } overlapMs=${-delta} cur[end]=${cur.end} next[start]=${nxt.start}`
            );
          } else if (delta > 0) {
            clickGaps += 1;
            clickGapTotalMs += delta;
            console.log(
              `[FILTER-COVERAGE][GAP][CLICK] #${i}->#${
                i + 1
              } gapMs=${delta} cur[end]=${cur.end} next[start]=${nxt.start}`
            );
          } else {
            console.log(
              `[FILTER-COVERAGE][ADJ][CLICK] #${i}->#${
                i + 1
              } contiguous (delta=0)`
            );
          }
        }
  
        // 2) Adjacency/overlap across ALL windows (click + noclick)
        const allWindows = rrwebFilterWindows
          .slice()
          .sort((a, b) => a.start - b.start);
        let allOverlaps = 0;
        let allGaps = 0;
        let allGapTotalMs = 0;
        for (let i = 0; i < allWindows.length - 1; i++) {
          const cur = allWindows[i];
          const nxt = allWindows[i + 1];
          const delta = nxt.start - cur.end;
          if (delta < 0) {
            allOverlaps += 1;
            console.log(
              `[FILTER-COVERAGE][OVERLAP][ALL] ${cur.kind}#${cur.index} -> ${
                nxt.kind
              }#${nxt.index} overlapMs=${-delta}`
            );
          } else if (delta > 0) {
            allGaps += 1;
            allGapTotalMs += delta;
            console.log(
              `[FILTER-COVERAGE][GAP][ALL] ${cur.kind}#${cur.index} -> ${nxt.kind}#${nxt.index} gapMs=${delta}`
            );
          }
        }
  
        // 3) Event coverage: which filterable events (from refinedLogEntries) were left unassigned
        const claimedIds = new Set();
        analysisResults.forEach((r) => {
          if (r && Array.isArray(r.clipEventIds)) {
            r.clipEventIds.forEach((id) => claimedIds.add(id));
          }
        });
        const isOverlayRange = (text) =>
          typeof text === "string" &&
          /^\[(\d{2}):(\d{2})\.(\d{3}) to (\d{2}):(\d{2})\.(\d{3})\]:/.test(text);
        const eventCoveredByWindows = (entry) => {
          if (isOverlayRange(entry.text)) {
            const m = entry.text.match(
              /^\[(\d{2}):(\d{2})\.(\d{3}) to (\d{2}):(\d{2})\.(\d{3})\]:/
            );
            if (!m) return false;
            const startMs =
              Number(m[1]) * 60000 + Number(m[2]) * 1000 + Number(m[3]);
            const endMs =
              Number(m[4]) * 60000 + Number(m[5]) * 1000 + Number(m[6]);
            return rrwebFilterWindows.some(
              (w) => startMs < w.end && endMs > w.start
            );
          }
          const t = entry.adjustedMs;
          if (typeof t !== "number") return false;
          return rrwebFilterWindows.some((w) => t >= w.start && t <= w.end);
        };
        const leftover = filterableEventsUniverse.filter(
          (e) => !claimedIds.has(mkEventId(e)) && !eventCoveredByWindows(e)
        );
        const multiAssignedSample = filterableEventsUniverse
          .map((e) => {
            const t = e.adjustedMs;
            const idxs = rrwebFilterWindows
              .map((w, i) => (t >= w.start && t <= w.end ? i : -1))
              .filter((i) => i >= 0);
            return { e, idxs };
          })
          .filter((x) => x.idxs.length > 1)
          .slice(0, 5);
        console.log("[FILTER-COVERAGE][SUMMARY]", {
          windowsClick: clickWindows.length,
          windowsAll: allWindows.length,
          clickOverlaps,
          clickGaps,
          clickGapTotalMs,
          allOverlaps,
          allGaps,
          allGapTotalMs,
          filterableEventsUniverse: filterableEventsUniverse.length,
          claimedIds: claimedIds.size,
          leftoverEvents: leftover.length,
        });
        if (leftover.length > 0) {
          console.log(
            "[FILTER-COVERAGE][LEFTOVER][SAMPLE]",
            leftover.slice(0, 10).map((e) => ({
              t: e.adjustedMs,
              text: e.text,
              src: e.source,
            }))
          );
        }
        if (multiAssignedSample.length > 0) {
          console.log(
            "[FILTER-COVERAGE][DUPLICATE-COVERAGE][SAMPLE]",
            multiAssignedSample.map((x) => ({
              t: x.e.adjustedMs,
              text: x.e.text,
              src: x.e.source,
              windows: x.idxs,
            }))
          );
        }
      } catch (e) {
        console.warn("[FILTER-COVERAGE] Diagnostics failed:", e?.message || e);
      }
  
      console.log("[DEBUG-SORT] Before sort:");
      analysisResults.forEach((r, i) => {
        console.log(`  [${i}] eventIndex=${r.eventIndex} clipStartMs=${r.clipStartMs} type=${typeof r.clipStartMs}`);
      });
      analysisResults.sort((a, b) => (a.clipStartMs || 0) - (b.clipStartMs || 0));
  
      // Collect all clipEventLogs for the final transcription
      clipEventLogs = analysisResults
        .filter((result) => result && result.clipEventLog)
        .map((result) => result.clipEventLog)
        .join("\n\n--- NEXT CLIP ---\n\n");
  
      if (DEBUG_NOCLICK) {
        const nonClickResultsCount = analysisResults.filter(
          (r) =>
            r &&
            typeof r.eventIndex === "string" &&
            r.eventIndex.startsWith("noclick-")
        ).length;
        console.log(
          `🔎 [NOCLICK DEBUG] Final non-click result count=${nonClickResultsCount}`
        );
      }
  
      // =============================================================
      // STEP 3: Add rrweb clicks with AI analysis to the activity log
      // =============================================================
      console.log(
        "\n==== STEP 3: ADDING RRWEB CLICKS WITH AI ANALYSIS TO LOG ===="
      );
  
      // Get the true original start time for relative calculations
      const initialVideoStartTimestamp = processedRecording[0]?.timestamp;
  
      // Create new summary log entries for each successfully analyzed clip
      const clipSummaryEntries = [];
      let previousClipEndMs = null; // Track the previous clip's end time to prevent overlaps
  
      for (const [index, result] of analysisResults.entries()) {
        if (result && result.reviewedText) {
          // Find the original clip task, which contains the definitive file path
          const originalClipTask = finalClipsToAnalyze.find(
            (task) => task.event.eventIndex === result.eventIndex
          );
  
          if (!originalClipTask) {
            console.warn(
              `[CLIP-LOG] Could not find original clip task for eventIndex ${result.eventIndex}. Skipping log entry.`
            );
            continue;
          }
  
          const clipFilePath = originalClipTask.clipPath;
          const clipFileName = path.basename(clipFilePath || "");
          // Build public GCS URL matching our upload layout
          const gcsClipUrl = getGcsClipUrl(session.id, clipFileName);

          // If the local file is missing, proceed using the GCS URL (when available)
          if (!fs2.existsSync(clipFilePath || "")) {
            if (gcsClipUrl) {
              console.warn(
                `[CLIP-LOG-FILTER] Local clip missing for event ${result.eventIndex} at path: ${clipFilePath}. Using GCS URL instead.`
              );
            } else {
              console.warn(
                `[CLIP-LOG-FILTER] No local clip and no GCS URL available for event ${result.eventIndex}. Skipping log entry.`
              );
              continue;
            }
          }
  
          // Find the corresponding match to get the rrweb event for log generation
          const match = rrwebMatches.find(
            (m) => m.postHogEvent.eventIndex === result.eventIndex
          );
  
          // Ensure we have a valid match and all necessary timestamps before proceeding
          if (!initialVideoStartTimestamp) {
            console.warn(
              `[CLIP-LOG] Missing initialVideoStartTimestamp. Skipping log entry for eventIndex ${result.eventIndex}.`
            );
            continue;
          }
  
          if (
            !match ||
            !match.rrwebEvent ||
            !match.rrwebEvent.originalTimestamp
          ) {
            // Non-click coverage clips_folder won't have an rrweb match; include them using their own clip timing
            if (originalClipTask && originalClipTask.isNonClickSegment) {
              try {
                // Non-click clip times are already relative to the adjusted timeline.
                const startRel = result.clipStartMs;
                const endRel = result.clipEndMs;
                const contextTimestamp = `<segment id="${index}">`;
  
                // For consistency with other entries, set adjusted/original relative ms from the relative start
                const adjustedRelativeTimeMs = startRel;
                const originalRelativeTimeMs = startRel;
  
                const clipSummaryEntry = {
                  adjustedMs: adjustedRelativeTimeMs,
                  originalMs: originalRelativeTimeMs,
                  originalAbsoluteMs: null,
                  text: result.reviewedText,
                  isPreRecording: false,
                  eventIndex: result.eventIndex,
                  originalEvent: null,
                  clipStartMs: result.clipStartMs,
                  clipEndMs: result.clipEndMs,
                  contextDescription: result.contextDescription,
                  contextTimestamp: contextTimestamp,
                  clipUrl: gcsClipUrl || null,
                };
                clipSummaryEntries.push(clipSummaryEntry);
  
                // Persist analysis JSON similar to click clips_folder
                if (
                  originalClipTask.clipPath &&
                  !originalClipTask.isGapBasedNoClick
                ) {
                  const jsonFilePath = originalClipTask.clipPath.replace(
                    ".mp4",
                    ".json"
                  );
                  try {
                    const analysisData = {
                      eventIndex: result.eventIndex,
                      clickPositionInfo: null,
                      analysis: result.reviewedText,
                      log: result.clipEventLog,
                      clipUrl: gcsClipUrl || null,
                    };
                    await fs.writeFile(
                      jsonFilePath,
                      JSON.stringify(analysisData, null, 2)
                    );
                  } catch (e) {
                    console.error(
                      `[JSON-SAVE-ERROR] Failed to save analysis JSON for non-click ${result.eventIndex}: ${e.message}`
                    );
                  }
                }
  
                continue;
              } catch (ncErr) {
                console.warn(
                  `[CLIP-LOG] Failed to include non-click segment for eventIndex ${result.eventIndex}:`,
                  ncErr
                );
                continue;
              }
            }
  
            console.warn(
              `[CLIP-LOG] Could not find a valid match or timing info for eventIndex ${result.eventIndex}. Skipping log entry.`
            );
            continue;
          }
  
          // Now that we've verified the clip exists, proceed with log generation
          // Use adjusted rrweb timestamp directly; do not subtract skip shift again
          const adjustedAbsoluteTimestamp = match.rrwebEvent.timestamp;
          const adjustedRelativeTimeMs =
            adjustedAbsoluteTimestamp - initialVideoStartTimestamp;
          const originalRelativeTimeMs = adjustedRelativeTimeMs;
  
          // ... (the rest of the loop for contextTimestamp calculation and entry creation remains the same)
          // Calculate timestamp for contextDescription based on clip events
          let contextTimestamp = adjustedRelativeTimeMs; // Default to click timestamp
  
          console.log("ewfnwejfnewnf ", result);
          if (
            /*    result.clipEventLog &&
            result.clipEventLog !==
              "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded" */
  
            true
          ) {
            // ... (rest of the if block is unchanged)
            const eventLines = result.clipEventLog
              .split("\n")
              .filter((line) => line.trim().startsWith("<event"));
            const eventCount = eventLines.length;
  
            if (true) {
              // For click clips_folder, the contextTimestamp should be the range from the filename.
              const clipStartAdjustedRelativeMs = result.clipStartMs;
              const clipEndAdjustedMs = result.clipEndMs;
  
              console.log(
                "Jgewjwnfwjn ",
                clipStartAdjustedRelativeMs,
                clipEndAdjustedMs
              );
              if (
                typeof clipStartAdjustedRelativeMs === "number" &&
                typeof clipEndAdjustedMs === "number"
              ) {
                contextTimestamp = `<segment id="${index}">`;
                previousClipEndMs = clipEndAdjustedMs;
              }
              // If start/end are not numbers, it will fallback to the default single timestamp
            }
          }
  
          try {
            console.log("contextTimestampcontextTimestamp ", contextTimestamp);
            console.log(
              "nfewfwenfwejnfwejnfw ",
              parseClockTimestampToMs(
                contextTimestamp.split(`" end="`)[0].split(`start="`)[1]
              )
            );
          } catch (errrr) {
            console.log("errrr ", errrr);
          }
  
          const clipSummaryEntry = {
            adjustedMs: result.clipStartMs,
            originalMs: originalRelativeTimeMs,
            originalAbsoluteMs: adjustedAbsoluteTimestamp,
            text: result.reviewedText,
            isPreRecording: false,
            eventIndex: result.eventIndex,
            originalEvent: match.rrwebEvent,
            clipStartMs: result.clipStartMs,
            clipEndMs: result.clipEndMs,
            contextDescription: result.contextDescription,
            contextTimestamp: contextTimestamp,
            clipUrl: gcsClipUrl || null,
          };
          clipSummaryEntries.push(clipSummaryEntry);
  
          const analysisData = {
            eventIndex: result.eventIndex,
            clickPositionInfo:
              clickPositionResults.find((p) => p.eventIndex === result.eventIndex)
                ?.positionInfo || null,
            analysis: result.reviewedText,
            log: result.clipEventLog,
            clipUrl: gcsClipUrl || null,
          };
  
          const jsonFilePath = originalClipTask.clipPath.replace(".mp4", ".json");
          try {
            await fs.writeFile(
              jsonFilePath,
              JSON.stringify(analysisData, null, 2)
            );
          } catch (e) {
            console.error(
              `[JSON-SAVE-ERROR] Failed to save analysis JSON for event ${result.eventIndex}: ${e.message}`
            );
          }
        }
      }
      // Reconstruct the refinedLogEntries by combining the events from outside the clips_folder with the new summary entries.
      const clipTimeRanges = analysisResults
        .filter((result) => result && result.reviewedText)
        .map((result) => ({
          start: result.clipStartMs,
          end: result.clipEndMs,
        }));
      const logEntriesOutsideClips = refinedLogEntries.filter((entry) => {
        const entryTimestamp = entry.originalAbsoluteMs;
        if (entryTimestamp === undefined) {
          return true;
        }
  
        const isInClip = clipTimeRanges.some(
          (range) =>
            range.start &&
            range.end &&
            entryTimestamp >= range.start &&
            entryTimestamp <= range.end
        );
        return !isInClip;
      });
      refinedLogEntries = [...logEntriesOutsideClips, ...clipSummaryEntries];
  
      /*    console.log("--- Refined Log Entries ---");
        console.log(JSON.stringify(refinedLogEntries, null, 2));
        console.log("---------------------------"); */
  
      finalClipsToAnalyze.forEach((clipTask) => {
        const { name: clipName } = path.parse(clipTask.clipPath);
        const frameImagePath = path.join(
          clipsDir,
          `${clipName}_analysis_frame.png`
        );
        console.log(`  -> Clip: ${path.basename(clipTask.clipPath)}`);
        console.log(`  -> Frame: ${path.basename(frameImagePath)}`);
      });
  
      let sessionVideoStartTimestamp = null;
      let sessionVideoEndTimestamp = null;
  
      if (Array.isArray(processedRecording) && processedRecording.length > 0) {
        for (const event of processedRecording) {
          const ts = event?.timestamp;
          if (typeof ts !== "number" || Number.isNaN(ts)) continue;
          if (
            sessionVideoStartTimestamp === null ||
            ts < sessionVideoStartTimestamp
          ) {
            sessionVideoStartTimestamp = ts;
          }
          if (
            sessionVideoEndTimestamp === null ||
            ts > sessionVideoEndTimestamp
          ) {
            sessionVideoEndTimestamp = ts;
          }
        }
      }
  
      const rawSessionDurationMs =
        sessionVideoStartTimestamp !== null && sessionVideoEndTimestamp !== null
          ? sessionVideoEndTimestamp - sessionVideoStartTimestamp
          : 0;
  
      let sessionDurationMs = rawSessionDurationMs;
      let longestClipEndMs = null;
      if (Array.isArray(finalClipsToAnalyze)) {
        for (const clipTask of finalClipsToAnalyze) {
          if (!clipTask) continue;
          let candidateEndMs = null;
  
          if (
            typeof clipTask.fileEndSeconds === "number" &&
            Number.isFinite(clipTask.fileEndSeconds)
          ) {
            candidateEndMs = Math.round(clipTask.fileEndSeconds * 1000);
          } else if (clipTask.clipPath) {
            const baseName = path.basename(clipTask.clipPath);
            const match = baseName.match(/_to_([0-9]+(?:\.[0-9]+)?)s/i);
            if (match) {
              const endSeconds = parseFloat(match[1]);
              if (Number.isFinite(endSeconds)) {
                candidateEndMs = Math.round(endSeconds * 1000);
              }
            }
          }
  
          if (
            typeof candidateEndMs === "number" &&
            Number.isFinite(candidateEndMs) &&
            (longestClipEndMs === null || candidateEndMs > longestClipEndMs)
          ) {
            longestClipEndMs = candidateEndMs;
          }
        }
      }
  
      if (
        typeof longestClipEndMs === "number" &&
        Number.isFinite(longestClipEndMs) &&
        longestClipEndMs > sessionDurationMs
      ) {
        sessionDurationMs = longestClipEndMs;
        if (sessionVideoStartTimestamp !== null) {
          sessionVideoEndTimestamp =
            sessionVideoStartTimestamp + sessionDurationMs;
        } else if (sessionVideoEndTimestamp === null) {
          sessionVideoEndTimestamp = sessionDurationMs;
        }
      }
  
      const sessionDurationAdjustedMs =
        typeof longestClipEndMs === "number" && Number.isFinite(longestClipEndMs)
          ? longestClipEndMs
          : sessionDurationMs;
  
      const sessionEndMs =
        (sessionDurationAdjustedMs ?? sessionDurationMs ?? 0) + 1; // ensure the session end marker follows the final user-visible event
  
      // Add "Session ended" event at the end of the clip timeline
      // Compute adjusted (skip-aware) end time
      // processedRecording timestamps are already adjusted; no shift here
  
      refinedLogEntries.push({
        adjustedMs: sessionEndMs,
        originalMs: sessionDurationMs,
        originalAbsoluteMs: sessionVideoEndTimestamp,
        text: "Session ended",
        isPreRecording: false,
        eventIndex: "session-end",
        source: "system-generated",
      });
  
      const sessionVideoDurationAdjustedMs = sessionDurationAdjustedMs;
  
      /*  refinedLogEntries.push({
        adjustedMs: sessionVideoDurationAdjustedMs,
        originalMs: sessionDurationMs,
        originalAbsoluteMs: sessionVideoEndTimestamp,
        text: "Session ended",
        isPreRecording: false,
        eventIndex: "session-end",
        source: "system-generated",
      }); */
  
      // Re-sort the log after adding all events including session end
      refinedLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);
  
      // Create a Set of unique "fingerprints" for each pre-recording event.
      // The fingerprint is a combination of the timestamp and the core event text.
      const preRecordingEventFingerprints = new Set(
        preRecordingEvents.map((entry) => {
          // Normalize the text by removing the pre-recording flag for accurate comparison
          const cleanText = entry.text.replace(
            "[PRE-RECORDING (mention in the output)] ",
            ""
          );
          return `${entry.adjustedMs}_${cleanText}`;
        })
      );
      // Ensure strict chronological ordering for the final prompt output.
      // Some entries use a string range in `contextTimestamp` like "00:10.000 to 00:20.000".
      // We sort by the numeric start of that range when present; otherwise by a numeric timestamp.
      function getEntryStartMs(entry) {
        if (entry && typeof entry.contextTimestamp === "string") {
          // Extract left side of the range and parse as mm:ss.mmm
          const left = entry.contextTimestamp
            .split(`" end="`)[0]
            .split(`start="`)[1]
            ?.trim();
  
          console.log("Newkfnweqwnqwndqwjdnqw ", left);
          // Expect format HH:MM.SSS or MM:SS.mmm → normalize and parse
          const m = left.match(/^(\d{2}):(\d{2})\.(\d{3})$/);
          if (m) {
            const minutes = Number(m[1]);
            const seconds = Number(m[2]);
            const millis = Number(m[3]);
            return minutes * 60_000 + seconds * 1_000 + millis;
          }
        }
        if (entry && typeof entry.contextTimestamp === "number") {
          return entry.contextTimestamp;
        }
        return entry?.adjustedMs ?? Number.POSITIVE_INFINITY;
      }
  
      // Prepare entries for final formatting
      const entriesToFormat = refinedLogEntries.filter((entry) => {
        // Only include non-pre-recording entries that have text
        if (!entry.text || entry.isPreRecording) {
          return false;
        }
  
        // --- NEW DEDUPLICATION CHECK ---
        // Create a fingerprint for the current event and check if it exists in the pre-recording set.
        const eventFingerprint = `${entry.adjustedMs}_${entry.text}`;
        if (preRecordingEventFingerprints.has(eventFingerprint)) {
          console.log(
            `[DEDUPLICATION] Filtering out duplicate event: "${entry.text}" at ${entry.adjustedMs}ms`
          );
          return false; // Exclude this duplicate event
        }
  
        // EXCLUDE events that were already included in any clip log
        const eventId = `${entry.originalAbsoluteMs || entry.adjustedMs}_${
          entry.text
        }_${entry.source}`;
        if (allClipEventIds && allClipEventIds.has(eventId)) {
          return false;
        }
  
        // EXCLUDE PostHog events - only keep rrweb events, matched navigation events, and system events
        const isPostHogEvent = entry.originalEvent && entry.originalEvent.event; // PostHog events have an 'event' property
        const isRrwebEvent =
          entry.source === "rrweb-input" ||
          entry.source === "rrweb-navigation" ||
          entry.source === "rrweb-scroll" ||
          entry.eventIndex?.toString().startsWith("rrweb-");
        const isMatchedNavigationEvent = entry.source === "matched-navigation"; // These use rrweb timing
        const isSystemEvent =
          entry.text.includes("Session ended") ||
          entry.text.includes("Page view:") ||
          entry.text.includes("Page leave:");
        const isAIAnalyzedClick = entry.text.includes("/segment>"); // AI-analyzed clicks from rrweb events
        const isInactivityOverlay = entry.source === "inactivity-overlay";
  
        const isSessionEnded = entry.text.includes("Session ended");
        const shouldInclude = isAIAnalyzedClick || isSessionEnded;
  
        // Only include AI-analyzed activity descriptions (from click and no-click clips_folder),
        // inactivity overlays, and the terminal "Session ended" marker.
        return shouldInclude;
      });
      /* .sort((a, b) => getEntryStartMs(a) - getEntryStartMs(b)); */
  
      const entriesForFormatting = entriesToFormat;
  
      const finalEventLogForPrompt = entriesForFormatting
        .map((entry) => {
          // Handle inactivity overlays first, as they have a custom format
          if (entry.source === "inactivity-overlay") {
            // The text is already pre-formatted with the desired timestamp range.
            // We just need to wrap it in markdown bold for consistency.
            return `**${entry.text}**`;
          }
  
          // Check if this entry contains contextDescription (wrapped in quotes)
          const isContextDescription = entry.text?.includes("/segment>");
  
          if (isContextDescription) {
            // For contextDescription entries, use the calculated contextTimestamp
            if (entry.contextTimestamp) {
              if (typeof entry.contextTimestamp === "string") {
                // Range format for multiple events
                // Prefer adjusted range if available to prevent overlaps
                const rangeToUse =
                  entry._adjustedContextTimestamp || entry.contextTimestamp;
                return `${rangeToUse}${entry.text}`;
              } else {
                // Single timestamp for single event
                return `<event time="${formatMilliseconds(
                  entry.contextTimestamp
                )}">${entry.text}</event>`;
              }
            }
            // Fallback: exclude timestamp if no contextTimestamp available
            return entry.text;
          } else {
            if (entry.text === "Session ended") {
              return `<event time="${formatMilliseconds(entry.adjustedMs)}">${
                entry.text
              }</event>`;
            }
            // For other entries, include the timestamp
            return `<event time="${formatMilliseconds(entry.adjustedMs)}">${
              entry.text
            }</event>`;
          }
        })
        .join("\n");
  
      const preRecordingEventLogForPrompt = refinedLogEntries
        .filter((entry) => entry.text !== null && entry.isPreRecording)
        .reduce((acc, entry) => {
          const cleanText = entry.text.replace(
            "[PRE-RECORDING (mention in the output)] ",
            ""
          );
  
          console.log("nwekgnwegnw", cleanText);
          if (!cleanText.startsWith("Event: ")) {
            acc.push(
              `<event time="${formatMilliseconds(
                entry.adjustedMs
              )}">${cleanText}</event>`
            );
          }
          return acc;
        }, [])
        .join("\n");
  
      console.log(
        "--- preRecordingEventLogForPrompt ---",
        preRecordingEventLogForPrompt
      );
      console.log("---------------------------------");
      /*  const noClicksInLog = !finalEventLogForPrompt
        .toLowerCase()
        .includes("segment");
  
      const oneClickInLog =
        finalEventLogForPrompt.split("segment").length - 1 === 2; */
  
      // Log the size of the prompt to debug potential fetch errors
      /*   const promptText = `Write the user journey based on the activity log and provided video. Include everything (actions, UI, etc.)! (Limit: 100,000 words)
  
  Guidelines:
  - The session replay clip runs at 4 FPS (1 frame per 250 ms).
  - Do not invent or assume any mouse click unless it is explicitly shown in the activity log.
  - Just so you know, scroll events are not present in the activity log.
  - If you see no cursor, rrweb isn’t currently replaying a mouse position (e.g., the pointer left the viewport, there are no MouseMove events in that segment, or it’s a touch-only moment). Early frames or long idle spans can show no cursor by design.
  - If you see a pointer moving above the replay iframe with an SVG arrow and a small dot, that’s the cursor overlay (.replayer-mouse).
  - If you see a ripple radiate from the cursor when clicking, that’s the click pulse (.replayer-mouse.active::after using @keyframes click).
  - If you see a blue ring, that’s the touch indicator marking the finger’s last contact point. It follows touchmove, pulses on taps, and remains after touchend until the next pointer event or segment change (.replayer-mouse.touch-device).
  - If you see a red line following the cursor, that’s the mouse trail (canvas overlay .replayer-mouse-tail), which by default lasts about 500 ms.
  - If you see a skipped inactivity overlay, always mention it in the output.
  - Include all the timestamps from the log in the output.
  ${
    noClicksInLog
      ? `- 0 clicks happened in this session! Never mention clicks in the output!`
      : ""
  }
  
  ${
    preRecordingEventLogForPrompt
      ? `
  Pre-recording events (occurred before clip started, not visible in the recording. Mention them):
  """""
  ${preRecordingEventLogForPrompt}
  """""
  `
      : ""
  } 
  Session activity log :
  """""""""""""""
  ${finalEventLogForPrompt || "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"}
  """""""""""""""`;*/
  
      console.log(
        "jfwnfjwfewjfnewjfewnfjwnejfwnjfwnfw\n\n\n\n ",
        preRecordingEventLogForPrompt,
        "\n\nAAAAA\n\n",
        finalEventLogForPrompt
      );
  
      /*  console.log("noClicksInLognoClicksInLog ", noClicksInLog);
      console.log("oneClickInLogoneClickInLog ", oneClickInLog); */
  
      let completion;
      if (INCLUDE_ENTIRE_LOG_PROMPT) {
        const includeClicksInstruction =
          clickClipsToAnalyze.length > 1
            ? `<guidelines>
  <item>Limit: 10,000 words</item>
  <item>Include all visual details so the clip doesn't need to be watched</item>
  <item>Include timestamps</item>
  <item>Never ever omit a click event from any segment! Always mention all click events present in the session!</item>
  </guidelines>`
            : `<guidelines>
  <item>Limit: 10,000 words</item>
  <item>Include all visual details so the clip doesn't need to be watched</item>
  <item>Include timestamps</item>
  </guidelines>`;
  
        completion = await generateContentWithRetry(
          ai,
          {
            model: Gemini2,
            config: {
              temperature: 0.2,
              top_p: 0.95,
              responseMimeType: "text/plain" /* thinkingConfig: {
                thinkingBudget: 32768,  // Max value for deepest reasoning
                // Alternative: thinkingBudget: -1 for dynamic auto-max
              },  */,
            },
            systemInstruction: `<task>Transcribe this session into a plain-text narrative</task>
  
  ${includeClicksInstruction}`,
            contents: `<task>Transcribe this session into a plain-text narrative</task>
  
  ${includeClicksInstruction}
  
  ${
    preRecordingEventLogForPrompt
      ? `<pre_recording_events (occurred before clip started, not visible in the recording. Mention them)>
  ${preRecordingEventLogForPrompt}
  </pre_recording_events>`
      : ""
  }
  
  <session>
  ${
    finalEventLogForPrompt ||
    "No user interactions (e.g., scrolling, navigation, or mouse movement) or system events were recorded"
  }
  </session>`,
          },
          3,
          apiKeys,
          {
            sessionId: session?.id,
            promptLabel: "session-summary",
            promptContext: {
              promptType: "final-session-summary",
              clipCount: Array.isArray(finalClipsToAnalyze)
                ? finalClipsToAnalyze.length
                : 0,
              includesPreRecording: !!preRecordingEventLogForPrompt,
              includesClipLogs: !!clipEventLogs,
              refinedLogEntryCount: Array.isArray(refinedLogEntries)
                ? refinedLogEntries.length
                : 0,
            },
          }
        );
      } else {
        completion = { text: "(Skipped due to INCLUDE_ENTIRE_LOG_PROMPT=false)" };
      }
  
      // Calculate and log the cost for this main analysis prompt
      if (completion.usageMetadata) {
        console.log(
          "totalcompletion.usageMetadatacompletion.usageMetadata ",
          completion.usageMetadata
        );
        console.log(`\n--- Cost Analysis for Main Analysis Prompt ---`);
        const costBreakdown = calculateApiCallCost(
          completion.usageMetadata,
          pricingInfo
        );
        console.log(
          `Total cost for this analysis: $${costBreakdown.totalCost.toFixed(4)}`
        );
        console.log("--------------------------------------\n");
      } else {
        console.warn("No usage data available for this analysis prompt.");
      }
  
      const finalAnalysisText = completion.text;
  
      const transcription =
        JSON.stringify([
          startUrl,
          recording_duration,
          click_count,
          end_time,
          datetime,
          person_id,
          browser,
          osVar,
          osVersion,
          device_type,
          country,
          referrer,
          idVar,
          user_agent,
          allPages?.length || 1,
          lastPage,
          JSON.stringify(allPages || []), // Add the full list of visited URLs
        ]) +
        "FIRST_PART" +
        finalAnalysisText +
        "SECOND_PART" +
        "fwfewfwfwe\n\n\n\n\n\n\n\n\n\n" +
        JSON.stringify(clickPositionResults) +
        "aaaawwwwbbbb\n\n\n\n\n\n\n\n\n\n" +
        finalEventLogForPrompt +
        "\n\n\n\n\n\nAAAAQQQQQ\n\n\n\n\n\n\n\n" +
    /*     preRecordingEventLogForPrompt + */
        "\n\n\n\n\n\n" +
        "clips_foldervideoclips_foldervideoaaaaaa\n\n\n\n\n\n" +
        clipEventLogs;
  
      const updateQuery = `
              UPDATE sessionanalysis
              SET analysiscontent = $1, status = 'COMPLETED', processedat = COALESCE($3::date, NOW()::date)
              WHERE id = $2;
          `;
      const updateValues = [transcription, analysisId, null];
      await pool.query(updateQuery, updateValues);
  
      console.log(
        `Successfully processed and stored analysis for session ${session.id}`
      );
  
      // Save analysis locally
      try {
        const analysesDir = path.join(process.cwd(), "analyses6");
        await fs.mkdir(analysesDir, { recursive: true });
        const analysisFilePath = path.join(
          analysesDir,
          `session_${session.id}_analysis.txt`
        );
        await fs.writeFile(analysisFilePath, transcription, "utf-8");
        console.log(`Successfully saved analysis locally to ${analysisFilePath}`);
      } catch (localSaveError) {
        console.error(
          `Failed to save analysis locally for session ${session.id}:`,
          localSaveError
        );
      }
  
      // Clean up the main session clip file in production
      if (process.env.NODE_ENV === "production") {
        try {
          const videoFileName = `clips_foldervideo${session.id}.mp4`;
          await fs.unlink(videoFileName);
          console.log(`[CLEANUP] Deleted main session video: ${videoFileName}`);
        } catch (e) {
          console.warn(
            `[CLEANUP] Could not delete main session clip file: ${e.message}`
          );
        }
      }
  if (contextUri) {
    await deleteBatchContext({ storage, contextUri });
  }

  return { transcription };
}

function removeDoctypeFromRrwebEvents(events) {
  console.log(
    "[DOCTYPEREMOVER] Starting DOCTYPE sanitization (v3 - precise)..."
  );

  function recursivelyRemoveDoctype(node, depth = 0) {
    if (!node || typeof node !== "object") {
      return node;
    }

    // The type for a DocumentType node in rrweb's serialization is 1.
    if (node.type === 1) {
      console.log(
        `[DOCTYPEREMOVER] ✅ SUCCESS: Found and marked DOCTYPE node (type 1) for removal at depth ${depth}.`
      );
      return null; // Mark this node for removal by returning null.
    }

    // For all other nodes, including the Document node (type 0), we keep them but process their children.
    if (Array.isArray(node.childNodes)) {
      node.childNodes = node.childNodes
        .map((child) => recursivelyRemoveDoctype(child, depth + 1))
        .filter(Boolean); // filter(Boolean) removes any null items (our removed doctype).
    }

    return node;
  }

  let changesMade = false;
  const newEvents = events.map((event) => {
    if (event.type === 2 && event.data && event.data.node) {
      console.log(
        `[DOCTYPEREMOVER] Found FullSnapshot at timestamp ${event.timestamp}. Processing...`
      );

      const originalNodeString = JSON.stringify(event.data.node);
      const nodeToSanitize = JSON.parse(originalNodeString);

      // This will process this tree starting from the Document node (type 0)
      // and will return the modified Document node with the DOCTYPE child removed.
      const sanitizedNode = recursivelyRemoveDoctype(nodeToSanitize);

      if (JSON.stringify(sanitizedNode) !== originalNodeString) {
        changesMade = true;
        console.log(
          `[DOCTYPEREMOVER] Sanitization resulted in changes for FullSnapshot at ${event.timestamp}.`
        );
      }

      return {
        ...event,
        data: {
          ...event.data,
          node: sanitizedNode,
        },
      };
    }
    return event;
  });

  if (changesMade) {
    console.log(
      "[DOCTYPEREMOVER] ✅ Sanitization complete. At least one DOCTYPE node was found and removed."
    );
  } else {
    console.warn(
      "[DOCTYPEREMOVER] ⚠️ Sanitization complete, but NO DOCTYPE nodes (type 1) were found."
    );
  }

  const firstFullSnapshot = newEvents.find((e) => e.type === 2);
  if (firstFullSnapshot && firstFullSnapshot.data.node) {
    const rootNode = firstFullSnapshot.data.node;
    console.log(
      `[DEBUG-SNAPSHOT] After sanitization, the root node of the snapshot is: type=${
        rootNode.type
      }, tagName=${rootNode.tagName || "N/A"}`
    );
    const rootChildren = rootNode.childNodes || [];
    console.log(
      `[DEBUG-SNAPSHOT] Root node now has ${rootChildren.length} children.`
    );
    rootChildren.slice(0, 3).forEach((child, index) => {
      console.log(
        `[DEBUG-SNAPSHOT]   - Child ${index}: type=${child.type}, tagName=${
          child.tagName || "N/A"
        }`
      );
    });
  }

  return newEvents;
}
/**
 * Computes a stable shard index for a given user identifier.
 * Works for both numeric and string IDs and ensures a non-negative modulo result.
 */
function computeStableShardIndexForUserId(userId, totalWorkers) {
  const idAsString = String(userId ?? "");
  let hash = 0;
  for (let i = 0; i < idAsString.length; i++) {
    hash = (hash * 31 + idAsString.charCodeAt(i)) | 0; // 31-based rolling hash
  }
  if (hash < 0) hash = -hash;
  if (!Number.isFinite(totalWorkers) || totalWorkers <= 0) return 0;
  return hash % totalWorkers;
}

function addRrwebBlurEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding rrweb blur events to activity log...");

  if (!rrwebEvents || rrwebEvents.length === 0) {
    console.log("   No rrweb events provided - nothing to add");
    return logEntries;
  }

  // Filter for blur events: type 3 (Incremental Snapshot), data.source 2 (MouseInteraction), data.type 6 (Blur)
  const blurEvents = rrwebEvents.filter(
    (event) =>
      event.type === 3 && event.data?.source === 2 && event.data?.type === 6
  );

  if (blurEvents.length === 0) {
    console.log("   No blur events found in rrweb data");
    return logEntries;
  }

  console.log(`   Found ${blurEvents.length} blur events to process`);

  const adjustedRecordingStart = rrwebEvents[0]?.timestamp;
  if (!adjustedRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  const originalRecordingStart =
    rrwebEvents[0]?.originalTimestamp || adjustedRecordingStart;

  const newLogEntries = [...logEntries];
  const idNodeMap = buildRrwebNodeMap(rrwebEvents);

  for (const blurEvent of blurEvents) {
    const adjustedAbsoluteMs = blurEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - adjustedRecordingStart;

    const originalAbsoluteMs =
      blurEvent.originalTimestamp || blurEvent.timestamp;
    const originalRelativeTimeMs = originalAbsoluteMs - originalRecordingStart;

    let elementDescription = "an element";
    if (idNodeMap) {
      const nodeId = blurEvent.data.id;
      const node = idNodeMap.get(nodeId);
      if (node) {
        elementDescription = describeRrwebNode(node, idNodeMap);
      }
    }

    const blurText = `Blurred ${elementDescription}`;

    const isPreRecording = false; // Blur events are from rrweb so they're during recording

    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${blurText}`
      : blurText;

    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: originalAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-blur-${blurEvent.timestamp}`,
      originalEvent: blurEvent,
      source: "rrweb-blur",
    };

    newLogEntries.push(newLogEntry);
  }

  // Sort by time to maintain chronological order
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  console.log(`   📊 Added ${blurEvents.length} blur events to the log`);

  return newLogEntries;
}

async function main() {
  console.log("Starting script...");


/*   try {

    const response = await ai.models.list()

    console.log("Available Models:", response);
    for (const model of response.models) {
      // Filter for models that support content generation
      if (model.supportedGenerationMethods.includes("generateContent")) {
        console.log(`- ${model.name} (${model.displayName})`);
      }
    }
  } catch(errrr) {
    console.log("errrrerrrr ", errrr)
  }
return; */


 // --- Sharding Logic ---
const workerIdStr =
process.env.WORKER_ID ?? process.env.CLOUD_RUN_TASK_INDEX; // prefer WORKER_ID, fallback for old env name
const totalWorkersStr = process.env.TOTAL_WORKERS || "3"; // Default to 3

const workerId = parseInt(workerIdStr, 10);
const totalWorkers = parseInt(totalWorkersStr, 10);

if (isNaN(workerId)) {
console.error(
  "WORKER_ID environment variable must be set as an integer (e.g., 0, 1, 2)."
);
process.exit(1);
}

if (isNaN(totalWorkers) || totalWorkers <= 0) {
console.error(
  "TOTAL_WORKERS environment variable must be a positive integer."
);
process.exit(1);
}

console.log(`🚀 Starting worker ${workerId} of ${totalWorkers}.`);

  let allUsers;
  try {
    console.log("Fetching all users from the database...");
    // Fetch the last fetched timestamp along with other user details
    const usersResult = await pool.query(
      "SELECT id, email, posthoginstanceurl, posthogapikey, lastfetchedtimestamp FROM user2"
    );
    allUsers = usersResult.rows;
    console.log(`Found ${allUsers.length} total users to process.`);
  } catch (dbError) {
    console.error("Failed to fetch users from the database:", dbError);
    throw dbError;
  }

  // --- Filter users for this worker ---
  let users = allUsers.filter((user) => {
    const numericId = Number(user.id);
    if (Number.isFinite(numericId)) {
      return numericId % totalWorkers === workerId;
    }
    const shardIndex = computeStableShardIndexForUserId(user.id, totalWorkers);
    return shardIndex === workerId;
  });

  if (workerId === 3) {
    users = allUsers;
  }

  /*   // for development
  users = allUsers; */

  console.log(
    `Worker ${workerId} assigned ${users.length} of ${allUsers.length} users.`
  );

  let processedGlobal = 0;
  const GLOBAL_LIMIT = 1;

  for (const user of users) {
    let API_BASE_URL = "https://us.posthog.com/api"; // Or your PostHog instance URL
    let API_BASE_URL2 = "https://eu.posthog.com/api"; // Or your PostHog instance URL

    console.log(`Processing user: ${user.email} (ID: ${user.id})`);

    if (!user.posthogapikey || !user.posthoginstanceurl) {
      console.warn(
        `User ${user.email} is missing PostHog API key or instance URL. Skipping.`
      );
      continue;
    }


    // Store the timestamp from which we need to fetch recordings.
    const fetchFromTimestamp = user.lastfetchedtimestamp;
    console.log(user)

    try {
      const posthogApiKey = decrypt(user.posthogapikey);
      console.log("nfewkfwnefewjnfw ", posthogApiKey);
      const posthogProjectID = user.posthoginstanceurl;

      let batchWatchStatus = {
        completedSessionIds: new Set(),
        hasRunningJobs: false,
        hadPendingJobs: false,
      };
      if (GEMINI_BATCH_FOR_CLIP_PROMPTS_2) {
        batchWatchStatus = await processPendingClipBatchJobsForUser({
          user,
          posthogProjectID,
          posthogApiKey,
          API_BASE_URL,
          API_BASE_URL2,
          apiKeys,
          pool,
          processSessionFn: processSession,
        });
      
        if (batchWatchStatus.hasRunningJobs) {
          console.log(
            `[BATCH-WATCH] Pending batch jobs still running for user ${user.email}; skipping further processing until they complete.`
          );
          continue;
        }
        if (true) {
          console.log(
            `[BATCH-WATCH] Processed ${batchWatchStatus.completedSessionIds.size} pending batch sessions for user ${user.email}. Skipping new session fetch this run.`
          );
          continue;
        }
      }

      console.log("wefwefmwekfmwe ", posthogProjectID, process.env.NODE_ENV); // Daily per-user limit prefetch check based on sessionanalysis.processedat
      let existingCountForDay = 0;
      const runTodayEnv = process.env.RUN_TODAY;
      const runToday =
        typeof runTodayEnv === "string" &&
        runTodayEnv.trim().toLowerCase() === "true";
      const runParticularSession =
        typeof process.env.RUN_PARTICULAR_SESSION === "string" &&
        process.env.RUN_PARTICULAR_SESSION.trim().toLowerCase() === "true";
      const particularSessionIds =
        typeof process.env.PARTICULAR_SESSION_ID === "string"
          ? process.env.PARTICULAR_SESSION_ID.split(",")
              .map((s) => s.trim())
              .filter(Boolean)
          : [];
      const targetDay = runToday
        ? new Date().toISOString().slice(0, 10)
        : process.env.PARTICULAR_DAY; // expected YYYY-MM-DD when not running today
      const dailyAmountEnv = process.env.DAILY_AMOUNT;
      const dailyAmount = dailyAmountEnv ? Number(dailyAmountEnv) : null;
      console.log("Nfewkfewnfw ", targetDay)
      if (
        !runParticularSession &&
        targetDay &&
        dailyAmount &&
        Number.isFinite(dailyAmount) &&
        dailyAmount > 0
      ) {
        try {
          const countRes = await pool.query(
            `SELECT COUNT(*)::int AS cnt
             FROM sessionanalysis
             WHERE userid = $1
               AND (
                 processedat::date = $2::date
                 OR (
                 status = 'PROCESSING'
                 AND processedat::date = DATE '1980-07-07'
                 AND analysiscontent IS NOT NULL
                 )
                 OR (
                 status = 'PROCESSING'
                 AND processedat::date = DATE '1980-07-07'
                 AND analysiscontent IS NULL
                 AND createdat::date = $2::date
                 )
               )`,
            [user.id, targetDay]
          );
          console.log("gmewgwekgw ",user?.id,countRes)
          existingCountForDay = countRes?.rows?.[0]?.cnt ?? 0;
          console.log("fnewjfnwefjwenf ", existingCountForDay, dailyAmount)
        /*   return; */
          if (existingCountForDay >= dailyAmount) {
            console.log(
              `[DAILY-LIMIT] User ${user.id} already has ${existingCountForDay}/${dailyAmount} analyses on ${targetDay}. Skipping fetch.`
            );
            continue; // Next user
          }
        } catch (e) {
          console.warn(
            "[DAILY-LIMIT] Failed to run daily limit prefetch check; proceeding.",
            e?.message || e
          );
        }
      } else if (runParticularSession) {
        console.log(
          `[RUN_PARTICULAR_SESSION] Skipping daily limit and date checks; forcing specific session(s) ${
            particularSessionIds.length > 0
              ? particularSessionIds.join(", ")
              : "(unspecified)"
          }.`
        );
      }
      let sessions;
      if (runParticularSession && particularSessionIds.length > 0) {
        sessions = particularSessionIds.map((id) => ({
          id,
          start_time: new Date().toISOString(),
        }));
        console.log(
          `[RUN_PARTICULAR_SESSION] Enabled; will process only session(s): ${particularSessionIds.join(
            ", "
          )} and ignore date-based filtering.`
        );
      } else if (USE_OUTPUTFILE) {
        console.log(
          "[USE_OUTPUTFILE] Skipping PostHog session fetch; using local outputFile.json."
        );
        sessions = [
          {
            id: process.env.LOCAL_SESSION_ID || "outputfile",
            start_time: new Date().toISOString(),
          },
        ];
      } else {
        sessions = await fetchPostHogSessionRecordings(
          posthogProjectID,
          posthogApiKey,
          API_BASE_URL,
          API_BASE_URL2,
          fetchFromTimestamp
        );
      }
      if (runParticularSession && particularSessionIds.length === 0) {
        console.warn(
          "[RUN_PARTICULAR_SESSION] Enabled but PARTICULAR_SESSION_ID is not set; continuing with fetched sessions."
        );
      }
      console.log("beforesorting ", sessions);

    
      // At the end of your cron script

      // Example Output in Render Logs:
      // Memory usage: rss 55.45 MB; heapTotal 34.47 MB; heapUsed 27.81 MB; external 1.05 MB;

      if (!runParticularSession) {

      }
      // Restrict to particular day and remaining daily allowance (if configured)
      if (
        !runParticularSession &&
        targetDay &&
        dailyAmount &&
        Number.isFinite(dailyAmount) &&
        dailyAmount > 0
      ) {
        sessions = sessions.filter((s) => {
          const d = s?.start_time;
          if (!d) return false;
          try {
            return new Date(d).toISOString().slice(0, 10) === targetDay;
          } catch (_) {
            return false;
          }
        });
        if (sessions.length === 0) {
          console.log(
            `[DAILY-LIMIT] No sessions on ${targetDay} for user ${user.id}; existing ${existingCountForDay}/${dailyAmount}. Skipping user.`
          );
          continue; // Next user
        }
        const remainingDaily = Math.max(0, dailyAmount - existingCountForDay);
        if (sessions.length > remainingDaily) {
          sessions = sessions.slice(0, remainingDaily);
        }
      } // Sort by start_time (oldest first)
      sessions.sort((a, b) => new Date(a.start_time) - new Date(b.start_time));

      console.log("beforesorting ", sessions);
      const remainingCapacity = Math.max(0, GLOBAL_LIMIT - processedGlobal);
      if (remainingCapacity === 0) {
        console.log(
          `Global limit ${GLOBAL_LIMIT} reached before processing user ${user.id}.`
        );
        break;
      }
      let sessionsToProcess = runParticularSession
        ? sessions // allow trying all requested sessions sequentially within this run
        : sessions.slice(0, remainingCapacity);
      console.log("aaaaaaaaaa", user.email, process.env.NODE_ENV);

    

      // sessions = sessions.filter((_, index) => index === 0)
      console.log("aftersorting ", sessionsToProcess);

      if (sessionsToProcess.length === 0) {
        console.log(
          `No new session recordings found for user ${user.email} since ${
            fetchFromTimestamp || "the beginning of time"
          }.`
        );
        continue; // Move to the next user
      }
      console.log(
        `Found ${sessions.length} new sessions; processing ${sessionsToProcess.length} oldest for user ${user.email}.`
      );

      /*   console.log("Fnewkfnewjfnwf ", sessions); */

      console.log("aaaaAPI_BASE_URL ", API_BASE_URL);
      console.log("aaaaaAPI_BASE_URL2 ", API_BASE_URL2);

      // return;
      const CONCURRENCY_LIMIT = 1;
      const sessionQueue = [...sessionsToProcess];
      let lastProcessedStartTime = null; // track last successfully processed session start_time for this user
      let duplicateFound = false;

      // Eagerly advance user's lastfetchedtimestamp to newest of selected sessions (+1s to avoid refetching the same session)
      if (!runParticularSession && sessionsToProcess.length > 0) {
        const eagerAdvanceTo =
          sessionsToProcess[sessionsToProcess.length - 1].start_time;
        const eagerAdvanceToDate = new Date(eagerAdvanceTo);
        const eagerAdvanceToPlusOneSecond = new Date(
          eagerAdvanceToDate.getTime() + 1000
        );
        console.log(
          `Eagerly updating lastfetchedtimestamp for user ${
            user.id
          } to ${eagerAdvanceToPlusOneSecond.toISOString()} (start_time + 1s) before processing`
        );

        if (process.env.NODE_ENV !== "development") {
          await pool.query(
            "UPDATE user2 SET lastfetchedtimestamp = $1 WHERE id = $2",
            [eagerAdvanceToPlusOneSecond, user.id]
          );
        }
      }

      const worker = async () => {
        while (
          sessionQueue.length > 0 &&
          processedGlobal < GLOBAL_LIMIT &&
          !duplicateFound
        ) {
          const session = sessionQueue.shift();
          if (session) {
            try {
              const existingSession = await pool.query(
                "SELECT 1 FROM sessionanalysis WHERE posthogrecordingid = $1 LIMIT 1",
                [session.id]
              );
              if (existingSession.rowCount > 0 && process.env.NODE_ENV !== "development") {
                console.warn(
                  `[DUPLICATE] Session ${session.id} already exists in sessionanalysis; terminating run to avoid double-processing.`
                );
                if (!runParticularSession) {
                  duplicateFound = true;
                  throw new Error("Duplicate session detected; aborting run.");
                }
                console.warn(
                  "[RUN_PARTICULAR_SESSION] Duplicate detected; moving to next requested session."
                );
                continue;
              }

              await processSession(
                session,
                user,
                posthogProjectID,
                posthogApiKey,
                API_BASE_URL,
                API_BASE_URL2,
                apiKeys
              );
              lastProcessedStartTime = session.start_time;
              processedGlobal += 1;
              if (processedGlobal >= GLOBAL_LIMIT) {
                break;
              }
            } catch (err) {
              console.error(
                `Error processing session ${session.id} for user ${user.email}:`,
                err
              );
              if (!runParticularSession) {
                throw err; // Re-throw to abort further processing
              }
              // When forcing particular sessions, continue to the next ID in the list.
              console.warn(
                `[RUN_PARTICULAR_SESSION] Continuing to next session after failure on ${session.id}.`
              );
            }
          }
        }
      };

      const workers = [];
      for (let i = 0; i < CONCURRENCY_LIMIT; i++) {
        workers.push(worker());
      }

      await Promise.all(workers);

      // Stop globally after processing 3 sessions total
      if (processedGlobal >= GLOBAL_LIMIT) {
        console.log(
          `Reached global processing limit of ${GLOBAL_LIMIT}. Stopping.`
        );
        break;
      }
    } catch (error) {
      console.error(
        `A critical error occurred while processing user ${user.email}:`,
        error
      );
      // We don't revert the timestamp, as we want to avoid reprocessing what might have succeeded.
      // The next run will correctly start from the new timestamp.
      throw error; // Stop the script after a critical user-level failure
    }
  }

  console.log("Script has finished processing all users.");
}

// For immediate execution when the script starts
main().catch((error) => {
  console.error(error);
  process.exit(1);
});

export { processSession, finalizeSessionFromContext };

/**
 * Parses PostHog's custom element chain format into structured element data
 * @param {string} elementChainString - PostHog element chain string
 * @returns {Array} Array of parsed element objects
 */
function parsePostHogElementChain(elementChainString) {
  if (!elementChainString || typeof elementChainString !== "string") {
    return [];
  }

  try {
    // Split by semicolons to get individual elements
    const elementParts = elementChainString
      .split(";")
      .filter((part) => part.trim());
    const elements = [];

    for (const part of elementParts) {
      const element = parsePostHogElement(part.trim());
      if (element) {
        elements.push(element);
      }
    }

    return elements;
  } catch (error) {
    console.warn("Error parsing PostHog element chain:", error.message);
    return [];
  }
}

/**
 * Extracts element data directly from PostHog properties
 * @param {Object} properties - PostHog event properties
 * @returns {Object|null} Extracted element object or null
 */
function extractElementDataFromPostHogProperties(properties) {
  if (!properties || properties.$event_type !== "click") return null;

  // Try direct property extraction first
  let tagName = null;
  let textContent = properties.$el_text || null;
  let attributes = {};

  // Extract tag name from various possible properties
  if (properties.$el_tag_name) {
    tagName = properties.$el_tag_name.toLowerCase();
  }

  // Extract attributes
  if (properties.$el_id) attributes.id = properties.$el_id;
  if (properties.$el_class) attributes.class = properties.$el_class;
  if (properties.$el_href) attributes.href = properties.$el_href;
  if (properties.$el_name) attributes.name = properties.$el_name;
  if (properties.$el_type) attributes.type = properties.$el_type;
  if (properties.$el_value) attributes.value = properties.$el_value;

  // If we have enough data, return the element
  if (tagName || textContent) {
    return {
      tag_name: tagName || "unknown",
      textContent: textContent || "",
      attributes: attributes,
    };
  }

  return null;
}

/**
 * Parses a single PostHog element string into an element object
 * @param {string} elementString - Single element string from PostHog
 * @returns {Object|null} Parsed element object or null
 */
// ... existing code ...
function parsePostHogElement(elementString) {
  if (!elementString || typeof elementString !== "string") {
    return null;
  }

  try {
    // Regex to capture the main parts:
    // 1. Tag name (e.g., "div")
    // 2. Optional classes (e.g., ".class1.class2")
    // 3. The rest of the string containing attributes.
    const mainPartsRegex = /^([a-zA-Z0-9_-]+)((?:\.[a-zA-Z0-9_.-]+)*):?(.*)$/;
    const mainMatch = elementString.match(mainPartsRegex);

    if (!mainMatch) {
      // This will filter out malformed entries like "width" or "margin"
      // that don't look like a valid HTML tag/class structure.
      if (!elementString.includes(":") && !elementString.includes(".")) {
        return null;
      }
      // Fallback for simple tags without attributes
      return { tag_name: elementString, attributes: {}, textContent: "" };
    }

    const [, tagName, classPart = "", attrPart = ""] = mainMatch;

    const attributes = {};
    if (classPart) {
      attributes.class = classPart.replace(/\./g, " ").trim();
    }

    let textContent = "";

    // This regex is designed to find all valid attributes and text content,
    // ignoring something else in the string.
    const attrRegex =
      /(?:attr__([a-zA-Z0-9_-]+)="([^"]*)"|text="([^"]*)"|nth-child="([^"]*)"|nth-of-type="([^"]*)")/g;
    let match;
    while ((match = attrRegex.exec(attrPart)) !== null) {
      const [, attrName, attrValue, textValue, nthChildValue, nthOfTypeValue] =
        match;

      if (attrName) {
        attributes[attrName] = attrValue;
      } else if (textValue) {
        textContent = textValue;
      } else if (nthChildValue) {
        attributes["nth-child"] = nthChildValue;
      } else if (nthOfTypeValue) {
        attributes["nth-of-type"] = nthOfTypeValue;
      }
    }

    return {
      tag_name: tagName.toLowerCase(),
      attributes: attributes,
      textContent: textContent,
    };
  } catch (error) {
    console.warn(
      `Error parsing PostHog element string: "${elementString}"`,
      error.message
    );
    return null;
  }
}

/**
 * Fetches PostHog events for a session and generates a comprehensive activity log
 * including user interactions and system events.
 * @param {string} sessionReplayId The session replay ID.
 * @param {string} projectId The PostHog project ID (number as string).
 * @param {string} apiKey The PostHog API key.
 * @param {string} [baseUrl] Optional base URL (default: https://eu.posthog.com/api)
 * @returns {Promise<string[]>} Array of log entries with user actions and system events.
 */
// ... existing code ...
export async function fetchAndGenerateStandardLog(
  sessionReplayId,
  projectId,
  apiKey,
  API_BASE_URL,
  API_BASE_URL2,
  recordingDuration,
  skips,
  recordingStartTime = null,
  rrwebEvents = null
) {
  // Build the HogQL query to fetch events for the specific session
  // Use SELECT * to get all data including element properties
  const queryBody = {
    query: {
      kind: "HogQLQuery",
      query: `
        SELECT * FROM events 
        WHERE properties.$session_id = '${sessionReplayId}'
        ORDER BY timestamp ASC
      `,
    },
  };

  const url = `${API_BASE_URL}/projects/${projectId}/query/`;
  const url2 = `${API_BASE_URL2}/projects/${projectId}/query/`;

  console.log("PostHog query API URL:", url);
  console.log("HogQL Query:", queryBody.query.query);
  let response;
  try {
    response = await fetchWithRetry(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      data: queryBody,
    });
  } catch (err) {
    if (err.message.includes("Session terminated")) {
      throw err;
    }
    console.log("Error with primary PostHog query endpoint:", err?.message);
    try {
      response = await fetchWithRetry(url2, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        data: queryBody,
      });
      API_BASE_URL = API_BASE_URL2;
      console.log("Successfully used fallback PostHog query endpoint");
    } catch (err2) {
      console.log(
        "Failed to fetch from both PostHog query endpoints",
        err2?.message
      );
      return [];
    }
  }

  // The /query endpoint returns data in a different format than /events
  // We need to transform it to match what generateStandardLog expects
  const columns = response.data.columns || [];
  const results = response.data.results || [];

  // Element data (CSS selectors, element text, etc.) is available in properties
  // and can be used for enhanced element-based matching

  // 🔍 DEBUG: Log raw query response structure
  console.log("🔍 RAW POSTHOG QUERY RESPONSE:");
  console.log(`Columns returned: [${columns.join(", ")}]`);
  if (results.length > 0) {
    console.log(
      `Sample first result: [${JSON.stringify(results[0]).substring(
        0,
        200
      )}...]`
    );
  }
  console.log(`Total results: ${results.length}`);

  const transformedResults = results.map((row, rowIndex) => {
    const eventObject = {};
    columns.forEach((col, index) => {
      eventObject[col] = row[index];
    });

    // 🔍 DEBUG: Log properties for first few events
    if (rowIndex < 3) {
      console.log(
        `🔍 Event ${rowIndex} raw properties: ${typeof eventObject.properties} = ${JSON.stringify(
          eventObject.properties
        ).substring(0, 200)}...`
      );
    }

    // Properties are returned as a JSON string
    if (typeof eventObject.properties === "string") {
      try {
        eventObject.properties = JSON.parse(eventObject.properties);

        // 🔍 DEBUG: Log parsed properties for first few events
        if (rowIndex < 3) {
          console.log(
            `🔍 Event ${rowIndex} parsed properties keys: [${Object.keys(
              eventObject.properties || {}
            ).join(", ")}]`
          );
          console.log(
            `🔍 Event ${rowIndex} element data: $el_text="${
              eventObject.properties?.$el_text || "none"
            }", $el_css_selector="${
              eventObject.properties?.$el_css_selector || "none"
            }"`
          );
        }
      } catch (e) {
        console.warn(
          "Could not parse event properties JSON:",
          eventObject.properties
        );
        eventObject.properties = {};
      }
    }

    // Enhanced element reconstruction with CSS selector support
    eventObject.elements = [];

    // 🔍 DEBUG: Log element reconstruction for first few events and failing events
    if (rowIndex < 3 || eventObject.elements_chain) {
      console.log(
        `🔍 Event ${rowIndex} elements_chain: ${typeof eventObject.elements_chain} = ${JSON.stringify(
          eventObject.elements_chain
        ).substring(0, 100)}...`
      );
      console.log(
        `🔍 Event ${rowIndex} elements_chain_elements: ${JSON.stringify(
          eventObject.elements_chain_elements
        )}`
      );
    }

    // Parse elements_chain if available (PostHog custom format)
    if (eventObject.elements_chain) {
      try {
        if (typeof eventObject.elements_chain === "string") {
          // Handle PostHog's custom element chain format
          const parsedElements = parsePostHogElementChain(
            eventObject.elements_chain
          );
          if (parsedElements.length > 0) {
            eventObject.elements = parsedElements;
            if (rowIndex < 3) {
              console.log(
                `🔍 Event ${rowIndex} parsed PostHog elements_chain: ${eventObject.elements.length} elements`
              );
              console.log(
                `🔍 Event ${rowIndex} first element: ${JSON.stringify(
                  eventObject.elements[0]
                )}`
              );
            }
          }
        } else if (Array.isArray(eventObject.elements_chain)) {
          // Handle standard JSON array format
          eventObject.elements = eventObject.elements_chain;
          if (rowIndex < 3) {
            console.log(
              `🔍 Event ${rowIndex} parsed JSON elements_chain: ${eventObject.elements.length} elements`
            );
          }
        }
      } catch (e) {
        if (rowIndex < 3) {
          console.warn(
            `Could not parse elements_chain for event ${rowIndex}:`,
            e.message
          );
        }
      }
    }

    // Fallback to elements_chain_elements if elements_chain is not available
    if (
      eventObject.elements.length === 0 &&
      eventObject.elements_chain_elements
    ) {
      eventObject.elements_chain_elements.forEach((tagName, index) => {
        eventObject.elements.push({
          tag_name: tagName,
          order: index,
        });
      });

      if (rowIndex < 3) {
        console.log(
          `🔍 Event ${rowIndex} used elements_chain_elements fallback: ${eventObject.elements.length} elements`
        );
      }
    }

    // Final fallback: Try to reconstruct element data for $autocapture events
    if (
      eventObject.elements.length === 0 &&
      eventObject.event === "$autocapture"
    ) {
      const extractedElement = extractElementDataFromPostHogProperties(
        eventObject.properties
      );
      if (extractedElement) {
        eventObject.elements.push(extractedElement);
        if (rowIndex < 10) {
          console.log(
            `🔍 Event ${rowIndex} extracted element from PostHog properties: ${JSON.stringify(
              extractedElement
            )}`
          );
        }
      }
    }

    // 🔍 DEBUG: Log final element count for first few events
    if (rowIndex < 3) {
      console.log(
        `🔍 Event ${rowIndex} final elements count: ${eventObject.elements.length}`
      );
      if (eventObject.elements.length > 0) {
        console.log(
          `🔍 Event ${rowIndex} first element: ${JSON.stringify(
            eventObject.elements[0]
          )}`
        );
      }
    }

    // Element data is now directly available in properties since we use SELECT *
    // No need to add it again - properties.$el_text, $el_css_selector, etc. are already there

    // Element data enhancement completed
    console.log("eventObjecteventObject ", eventObject);
    return eventObject;
  });

  const transformedData = {
    results: transformedResults,
  };

  return generateStandardLog(
    transformedData,
    recordingDuration,
    skips,
    recordingStartTime,
    rrwebEvents
  );
}

function formatMilliseconds(ms) {
  if (ms == null || isNaN(ms)) {
    return "00:00.000";
  }
  const roundedMs = Math.round(ms);
  if (roundedMs < 0) {
    return "00:00.000";
  }
  const totalSeconds = Math.floor(roundedMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = roundedMs % 1000;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
    2,
    "0"
  )}.${String(milliseconds).padStart(3, "0")}`;
}

// Parse a timecode in the format MM:SS.mmm into milliseconds.
function parseTimecodeMs(tc) {
  if (tc == null) return 0;
  const str = String(tc).trim();
  const m = str.match(/^(\d{2}):(\d{2})\.(\d{3})$/);
  if (!m) return 0;
  const minutes = Number(m[1]);
  const seconds = Number(m[2]);
  const millis = Number(m[3]);
  if ([minutes, seconds, millis].some((n) => !Number.isFinite(n))) return 0;
  return minutes * 60_000 + seconds * 1_000 + millis;
}

// Shift all MM:SS.mmm timestamps found in text by a positive offset (ms).
function shiftTimestampsInText(text, offsetMs) {
  try {
    const off = Math.max(0, Math.round(Number(offsetMs) || 0));
    if (!off) return text;
    return String(text).replace(
      /\b(\d{2}):(\d{2})\.(\d{3})\b/g,
      (full, mm, ss, mmm) => {
        const baseMs = Number(mm) * 60_000 + Number(ss) * 1_000 + Number(mmm);
        const newMs = baseMs + off;
        return formatMilliseconds(newMs);
      }
    );
  } catch (_) {
    return text;
  }
}

function formatSecondsDelta(ms) {
  if (ms == null || isNaN(ms)) {
    return "0.000s";
  }
  const normalizedMs = Math.max(0, Math.round(Number(ms)));
  return `${(normalizedMs / 1000).toFixed(3)}s`;
}

function calculateTimeShiftForTimestamp(timestamp, skips) {
  // --- Start of Instrumented Function ---
  console.log(`\n[SHIFT_CALC] Calculating shift for timestamp: ${timestamp}`);

  let shift = 0;
  if (!skips || !Array.isArray(skips) || skips.length === 0) {
    console.log(`[SHIFT_CALC] No skips provided. Returning shift = 0.`);
    return 0;
  }

  const sortedSkips = [...skips].sort((a, b) => a.startTime - b.startTime);

  console.log(
    `[SHIFT_CALC] Processing ${sortedSkips.length} sorted skip periods.`
  );

  for (const [index, skip] of sortedSkips.entries()) {
    if (
      typeof skip.startTime !== "number" ||
      typeof skip.endTime !== "number" ||
      typeof skip.timeRemoved !== "number"
    ) {
      console.warn(
        `[SHIFT_CALC] Skipping invalid skip object at index ${index}:`,
        skip
      );
      continue;
    }

    console.log(
      `[SHIFT_CALC] -> Skip #${index}: start=${skip.startTime}, end=${skip.endTime}, removed=${skip.timeRemoved}`
    );

    if (timestamp >= skip.endTime) {
      shift += skip.timeRemoved;
      console.log(
        `[SHIFT_CALC]    - Timestamp is AFTER this skip. Adding ${skip.timeRemoved}. New total shift: ${shift}`
      );
    } else if (timestamp > skip.startTime && timestamp < skip.endTime) {
      const shiftWithinGap = timestamp - skip.startTime;
      shift += shiftWithinGap;
      console.log(
        `[SHIFT_CALC]    - Timestamp is INSIDE this skip. Adding partial shift of ${shiftWithinGap}. Final shift: ${shift}. Breaking loop.`
      );
      break;
    } else {
      console.log(
        `[SHIFT_CALC]    - Timestamp is BEFORE this skip. No change to shift.`
      );
    }
  }

  console.log(
    `[SHIFT_CALC] Final calculated shift for ${timestamp} is: ${shift}`
  );
  return shift;
  // --- End of Instrumented Function ---
}

async function enrichInputChangeEvents(logEntries, rrwebEvents) {
  console.log("Attempting to enrich input change events with rrweb data...");
  try {
    console.log(
      `[DEBUG][enrichInputChangeEvents] rrwebEvents total: ${
        rrwebEvents?.length || 0
      }`
    );
    const rrStartAdj = rrwebEvents?.[0]?.timestamp;
    const rrStartOrig = rrwebEvents?.[0]?.originalTimestamp || rrStartAdj;
    if (rrStartAdj) {
      console.log(
        `[DEBUG][enrichInputChangeEvents] rrweb start adjusted: ${rrStartAdj} (${
          new Date(rrStartAdj).toISOString?.() || rrStartAdj
        })`
      );
    }
    if (rrStartOrig) {
      console.log(
        `[DEBUG][enrichInputChangeEvents] rrweb start original: ${rrStartOrig} (${
          new Date(rrStartOrig).toISOString?.() || rrStartOrig
        })`
      );
    }
  } catch (e) {
    console.log(
      "[DEBUG][enrichInputChangeEvents] failed to log rrweb starts:",
      e?.message || e
    );
  }

  // 1. Find all input change events from PostHog log
  const inputChangeEvents = logEntries.filter(
    (entry) => entry.text && entry.text.includes("Some input changed")
  );
  try {
    const earliestPh = inputChangeEvents
      .map((e) => e.originalAbsoluteMs)
      .filter((v) => typeof v === "number")
      .sort((a, b) => a - b)[0];
    console.log(
      `[DEBUG][enrichInputChangeEvents] PH input events: ${
        inputChangeEvents.length
      }, earliest: ${earliestPh} (${
        earliestPh ? new Date(earliestPh).toISOString() : "n/a"
      })`
    );
  } catch (e) {
    console.log(
      "[DEBUG][enrichInputChangeEvents] failed to compute earliest PH input:",
      e?.message || e
    );
  }

  // 2. Find all rrweb input events that have been processed for time shifting
  const rrwebInputEvents = rrwebEvents.filter(
    (ev) =>
      ev.type === 3 && // IncrementalSnapshot
      ev.data.source === 5 && // Input
      ev.originalTimestamp // Ensure we only use events that have the raw timestamp preserved
  );
  try {
    const earliestRrInOrig = rrwebInputEvents
      .map((e) => e.originalTimestamp)
      .filter((v) => typeof v === "number")
      .sort((a, b) => a - b)[0];
    const earliestRrInAdj = rrwebInputEvents
      .map((e) => e.timestamp)
      .filter((v) => typeof v === "number")
      .sort((a, b) => a - b)[0];
    console.log(
      `[DEBUG][enrichInputChangeEvents] rrweb input events: ${
        rrwebInputEvents.length
      }, earliest original: ${earliestRrInOrig} (${
        earliestRrInOrig ? new Date(earliestRrInOrig).toISOString() : "n/a"
      }), earliest adjusted: ${earliestRrInAdj} (${
        earliestRrInAdj ? new Date(earliestRrInAdj).toISOString() : "n/a"
      })`
    );
  } catch (e) {
    console.log(
      "[DEBUG][enrichInputChangeEvents] failed to compute rrweb input earliest:",
      e?.message || e
    );
  }

  console.log("fjewhfewjhfwjehf ", inputChangeEvents, rrwebInputEvents);

  // ===== NEW DEBUG LOGGING START =====
  console.log("--- PostHog 'Input Changed' Events (pre-enrichment) ---");
  logEntries.forEach((entry) => {
    if (entry.text && entry.text.includes("Some input changed")) {
      console.log(
        `[PH] timestamp: ${entry.originalAbsoluteMs}, adjustedMs: ${
          entry.adjustedMs
        }, eventIndex: ${
          entry.eventIndex
        }, preRecording: ${!!entry.isPreRecording}, text: "${entry.text}"`
      );
    }
  });

  if (inputChangeEvents.length === 0) {
    console.log("No 'Some input changed' events found to enrich.");
    return logEntries;
  }
  console.log(
    `Found ${inputChangeEvents.length} input change events to enrich.`
  );

  // 3. Build a map of all nodes from the rrweb events.
  // This map will contain every node that ever existed in the recording.
  function buildRrwebNodeMap(events) {
    const fullSnapshot = events.find((e) => e.type === EventType.FullSnapshot);
    if (!fullSnapshot || !fullSnapshot.data || !fullSnapshot.data.node) {
      console.warn(
        "No valid full snapshot event found. Cannot build node map."
      );
      return null;
    }

    const idNodeMap = new Map();

    // Helper to recursively add a node and its children to the map
    function addNodeToMap(node) {
      if (!node) return;
      if (node.id !== undefined) {
        // We store the serialized node itself, which contains its attributes.
        idNodeMap.set(node.id, node);
      }
      if (node.childNodes) {
        for (const child of node.childNodes) {
          addNodeToMap(child);
        }
      }
    }

    // Start with the full snapshot
    addNodeToMap(fullSnapshot.data.node);

    // Add nodes from incremental mutations
    const mutationEvents = events.filter(
      (e) =>
        e.type === EventType.IncrementalSnapshot &&
        e.data.source === IncrementalSource.Mutation
    );

    for (const event of mutationEvents) {
      if (event.data.adds) {
        for (const added of event.data.adds) {
          addNodeToMap(added.node);
        }
      }
      // We don't need to process `removes` because we want to find any node
      // that *ever* existed. An input event could have occurred on a node
      // that was later removed.
    }
    return idNodeMap;
  }

  console.log("Building comprehensive node map from rrweb events...");
  const idNodeMap = buildRrwebNodeMap(rrwebEvents);

  if (!idNodeMap || idNodeMap.size === 0) {
    console.warn(
      "Failed to build rrweb node map or map is empty. Cannot enrich input events."
    );
    return logEntries;
  }
  console.log(`Successfully built node map with ${idNodeMap.size} nodes.`);

  // 4. For each PostHog input event, find the best rrweb match using timestamp-only matching
  // Enforce one-to-one matching: track rrweb input events already claimed
  const consumedRrwebInputKeys = new Set();
  const rrwebInputKey = (ev) =>
    `${ev.originalTimestamp || ev.timestamp}|${ev.data?.id ?? ""}|${
      ev.data?.text ?? ""
    }`;
  for (const phEvent of inputChangeEvents) {
    console.log(
      `\n[ENRICH_DEBUG] --------------------------------------------------`
    );
    console.log(
      `[ENRICH_DEBUG] Processing PostHog Event #${phEvent.eventIndex}: "${phEvent.text}"`
    );
    console.log(
      `[ENRICH_DEBUG]   PH original=${phEvent.originalAbsoluteMs} (${new Date(
        phEvent.originalAbsoluteMs
      ).toISOString()}), adjustedMs=${
        phEvent.adjustedMs
      }, preRecording=${!!phEvent.isPreRecording}`
    );
    console.log(
      `[ENRICH_DEBUG]   -> Matching against ${rrwebInputEvents.length} rrweb input events.`
    );
    console.log(
      `[ENRICH_DEBUG]   -> Search window: [${
        phEvent.originalAbsoluteMs - 2500
      } (${new Date(phEvent.originalAbsoluteMs - 2500).toISOString()}), ${
        phEvent.originalAbsoluteMs
      } (${new Date(phEvent.originalAbsoluteMs).toISOString()})]`
    );

    // Find rrweb input events that happened leading up to the PostHog event
    // PostHog often sends one 'change' event after the user stops typing
    const potentialMatchesAll = rrwebInputEvents.filter((rrwebEvent) => {
      // Use the UN-SHIFTED originalTimestamp for matching against the UN-SHIFTED PostHog timestamp
      const timeDiff =
        rrwebEvent.originalTimestamp - phEvent.originalAbsoluteMs;
      // Look in a window from 2500ms BEFORE to 0ms AFTER the PostHog event (i.e., only leading up to the event)
      return timeDiff >= -2500 && timeDiff <= 0;
    });
    // Exclude rrweb inputs already consumed by earlier PostHog events
    const potentialMatches = potentialMatchesAll.filter(
      (m) => !consumedRrwebInputKeys.has(rrwebInputKey(m))
    );

    console.log(
      `[ENRICH_DEBUG]   -> Found ${potentialMatches.length} potential rrweb matches.`
    );
    try {
      potentialMatches.slice(0, 10).forEach((m, i) => {
        const dRaw = m.originalTimestamp - phEvent.originalAbsoluteMs;
        const dAdj =
          (m.timestamp ?? m.originalTimestamp) - phEvent.originalAbsoluteMs;
        console.log(
          `     [CAND ${i + 1}] rrweb original=${
            m.originalTimestamp
          } (${new Date(m.originalTimestamp).toISOString()}), adjusted=${
            m.timestamp
          } (${
            m.timestamp ? new Date(m.timestamp).toISOString() : "n/a"
          }), rawDelta=${dRaw}ms, adjDelta=${dAdj}ms, value="${
            m.data?.text ?? ""
          }" id=${m.data?.id}`
        );
      });
      if (potentialMatches.length > 10) {
        console.log(`     ... ${potentialMatches.length - 10} more candidates`);
      }
    } catch (_) {}

    // EXTRA DIAGNOSTICS: nearest before/after and delta distributions
    try {
      const phAbs = phEvent.originalAbsoluteMs;

      // Find nearest rrweb input overall (before and after)
      let nearestBefore = null;
      let nearestAfter = null;
      let minBeforeDelta = Infinity;
      let minAfterDelta = Infinity;
      for (const ev of rrwebInputEvents) {
        const o = ev.originalTimestamp;
        if (typeof o !== "number") continue;
        if (o <= phAbs) {
          const d = phAbs - o;
          if (d < minBeforeDelta) {
            minBeforeDelta = d;
            nearestBefore = ev;
          }
        } else {
          const d = o - phAbs;
          if (d < minAfterDelta) {
            minAfterDelta = d;
            nearestAfter = ev;
          }
        }
      }
      if (nearestBefore) {
        console.log(
          `     [NEAREST-BEFORE] rrweb.orig=${
            nearestBefore.originalTimestamp
          } (${new Date(
            nearestBefore.originalTimestamp
          ).toISOString()}), rrweb.adj=${nearestBefore.timestamp} (${new Date(
            nearestBefore.timestamp
          ).toISOString()}), delta=${minBeforeDelta}ms, value="${
            nearestBefore?.data?.text ?? ""
          }" id=${nearestBefore?.data?.id}`
        );
      } else {
        console.log("     [NEAREST-BEFORE] none");
      }
      if (nearestAfter) {
        console.log(
          `     [NEAREST-AFTER] rrweb.orig=${
            nearestAfter.originalTimestamp
          } (${new Date(
            nearestAfter.originalTimestamp
          ).toISOString()}), rrweb.adj=${nearestAfter.timestamp} (${new Date(
            nearestAfter.timestamp
          ).toISOString()}), delta=${minAfterDelta}ms, value="${
            nearestAfter?.data?.text ?? ""
          }" id=${nearestAfter?.data?.id}`
        );
      } else {
        console.log("     [NEAREST-AFTER] none");
      }

      // Stats for candidates in the current window
      if (potentialMatches.length > 0) {
        const deltas = potentialMatches
          .map((m) => Math.abs(m.originalTimestamp - phAbs))
          .sort((a, b) => a - b);
        const minDelta = deltas[0];
        const maxDelta = deltas[deltas.length - 1];
        const mid = Math.floor(deltas.length / 2);
        const median =
          deltas.length % 2 === 0
            ? Math.round((deltas[mid - 1] + deltas[mid]) / 2)
            : deltas[mid];
        console.log(
          `     [CAND-STATS] count=${deltas.length} min=${minDelta}ms median=${median}ms max=${maxDelta}ms`
        );
        // Also show first and last candidate absolute times
        const firstC = potentialMatches[0];
        const lastC = potentialMatches[potentialMatches.length - 1];
        console.log(
          `     [CAND-EDGE] first.orig=${firstC.originalTimestamp} (${new Date(
            firstC.originalTimestamp
          ).toISOString()}) Δ=${Math.abs(
            firstC.originalTimestamp - phAbs
          )}ms | last.orig=${lastC.originalTimestamp} (${new Date(
            lastC.originalTimestamp
          ).toISOString()}) Δ=${Math.abs(lastC.originalTimestamp - phAbs)}ms`
        );
      }
    } catch (e) {
      console.log(
        "   [DEBUG] failed to compute nearest/stats:",
        e?.message || e
      );
    }

    let bestMatch = null;

    if (potentialMatches.length > 0) {
      // NEW LOGIC: Filter potential matches by element properties first to isolate the correct input field.
      const phOrig = phEvent?.originalEvent;
      const phProps = phOrig?.properties || {};
      let phId = null,
        phName = null,
        phTag = null,
        phPlaceholder = null,
        phType = null;

      // Extract properties from PostHog event to identify the element
      if (Array.isArray(phEvent?.elements) && phEvent.elements.length > 0) {
        const el = phEvent.elements[0];
        phTag = el.tag_name || null;
        const attrs = el.attributes || {};
        phId = attrs.id || null;
        phName = attrs.name || null;
        phPlaceholder = attrs.placeholder || null;
        phType = attrs.type || null;
      } else if (typeof phOrig?.elements_chain === "string") {
        const parsed = parsePostHogElementChain(phOrig.elements_chain) || [];
        const pref = parsed.find((e) => e?.tag_name === "input") || parsed[0];
        if (pref) {
          phTag = pref.tag_name || null;
          const attrs = pref.attributes || {};
          phId = attrs.id || null;
          phName = attrs.name || null;
          phPlaceholder = attrs.placeholder || null;
          phType = attrs.type || null;
        }
      } else {
        phTag = phProps.$el_tag_name || null;
        phId = phProps.$el_id || null;
        phName = phProps.$el_name || null;
        phPlaceholder = phProps.$el_placeholder || null;
        phType = phProps.$el_type || null;
      }

      const elementSpecificMatches = potentialMatches.filter((rrwebEvt) => {
        const nodeId = rrwebEvt.data.id;
        const node = idNodeMap.get(nodeId);
        if (!node) return false;
        const a = node?.attributes || {};

        // Match if at least one common, non-null identifier matches.
        if (phId && a.id) return phId === a.id;
        if (phName && a.name) return phName === a.name;
        if (phPlaceholder && a.placeholder)
          return phPlaceholder === a.placeholder;

        // If PostHog event has no strong identifiers, we can't reliably filter by element.
        // In that case, we fall back to using all time-based matches.
        if (!phId && !phName && !phPlaceholder) {
          return true;
        }

        return false;
      });

      console.log(
        `[ENRICH_DEBUG]   -> Found ${elementSpecificMatches.length} property-matching events out of ${potentialMatches.length}.`
      );

      // Prefer element-specific matches. If none, fall back to all potential matches in the time window.
      const sourceForBestMatch =
        elementSpecificMatches.length > 0
          ? elementSpecificMatches
          : potentialMatches;

      if (sourceForBestMatch.length > 0) {
        // The best match is the LAST event in the burst, as it has the final value
        const lastMatch = sourceForBestMatch[sourceForBestMatch.length - 1];
        bestMatch = {
          rrwebEvent: lastMatch,
          similarity: 0, // No text similarity since we're not comparing text
          timeDiff: Math.abs(
            lastMatch.originalTimestamp - phEvent.originalAbsoluteMs
          ),
          rrwebValue: lastMatch.data.text || "",
        };
        console.log(
          `[ENRICH_DEBUG]   -> ✅ Best match (last of relevant): value="${bestMatch.rrwebValue}", time diff=${bestMatch.timeDiff}ms, rrweb_ts=${lastMatch.originalTimestamp}`
        );
      }
    }

    // Apply the enrichment if we found a match
    if (bestMatch) {
      // Reserve this rrweb input for uniqueness across PH events
      try {
        const key = rrwebInputKey(bestMatch.rrwebEvent);
        if (consumedRrwebInputKeys.has(key)) {
          console.log(
            `   [UNIQUE][WARN] Selected rrweb input already consumed: ${key}`
          );
        } else {
          consumedRrwebInputKeys.add(key);
          console.log(`   [UNIQUE] Reserving rrweb input ${key}`);
        }
      } catch (_) {}
      const nodeId = bestMatch.rrwebEvent.data.id;
      const node = idNodeMap.get(nodeId);

      if (node) {
        try {
          const a = node?.attributes || {};
          let phTag = null,
            phId = null,
            phName = null,
            phPlaceholder = null,
            phType = null,
            phText = null;
          const phOrig = phEvent?.originalEvent;
          const phProps = phOrig?.properties || {};
          if (Array.isArray(phEvent?.elements) && phEvent.elements.length > 0) {
            const el = phEvent.elements[0];
            phTag = el.tag_name || null;
            phText = el.textContent || null;
            const attrs = el.attributes || {};
            phId = attrs.id || null;
            phName = attrs.name || null;
            phPlaceholder = attrs.placeholder || null;
            phType = attrs.type || null;
          } else if (typeof phOrig?.elements_chain === "string") {
            const parsed =
              parsePostHogElementChain(phOrig.elements_chain) || [];
            const pref =
              parsed.find((e) => e?.tag_name === "input") || parsed[0];
            if (pref) {
              phTag = pref.tag_name || null;
              phText = pref.textContent || null;
              const attrs = pref.attributes || {};
              phId = attrs.id || null;
              phName = attrs.name || null;
              phPlaceholder = attrs.placeholder || null;
              phType = attrs.type || null;
            }
          } else {
            phTag = phProps.$el_tag_name || null;
            phId = phProps.$el_id || null;
            phName = phProps.$el_name || null;
            phPlaceholder = phProps.$el_placeholder || null;
            phType = phProps.$el_type || null;
            phText = phProps.$el_text || null;
          }
          const matchId = phId && a.id && a.id === phId;
          const matchName = phName && a.name && a.name === phName;
          const matchPlaceholder =
            phPlaceholder && a.placeholder && a.placeholder === phPlaceholder;
          console.log(
            `     [BEST-NODE] nodeId=${nodeId} attrs: id=${a.id ?? ""} name=${
              a.name ?? ""
            } placeholder=${a.placeholder ?? ""} type=${
              a.type ?? ""
            } | PH-hints tag=${phTag ?? ""} id=${phId ?? ""} name=${
              phName ?? ""
            } placeholder=${phPlaceholder ?? ""} type=${phType ?? ""} text=${(
              phText ?? ""
            ).slice(
              0,
              60
            )} | matches: id=${!!matchId} name=${!!matchName} placeholder=${!!matchPlaceholder}`
          );
        } catch (_) {}
        // We have the serialized node, now let's describe it as best we can.
        const attributes = node.attributes || {};
        let elementDescription = `an input field`;

        if (attributes.placeholder) {
          elementDescription = `input with placeholder "${attributes.placeholder}"`;
        } else if (attributes["aria-label"]) {
          elementDescription = `input with label "${attributes["aria-label"]}"`;
        } else if (attributes.name) {
          elementDescription = `input named "${attributes.name}"`;
        } else if (attributes.id) {
          elementDescription = `input with id "${attributes.id}"`;
        }

        // Use the rrweb value since PostHog doesn't provide it
        const finalValue = bestMatch.rrwebValue;

        let newLogText = `changed ${elementDescription}`;
        if (
          finalValue !== undefined &&
          finalValue !== null &&
          finalValue !== ""
        ) {
          newLogText += ` to "${finalValue}"`;
        }

        // Update the log entry in the main array
        const targetEntry = logEntries.find(
          (entry) => entry.eventIndex === phEvent.eventIndex
        );
        if (targetEntry) {
          const prefix = targetEntry.isPreRecording
            ? "[PRE-RECORDING (mention in the output)] "
            : "";
          targetEntry.text = prefix + newLogText;

          // Store the matched rrweb event for potential use in activity log
          targetEntry.matchedRrwebEvent = bestMatch.rrwebEvent;
          targetEntry.matchConfidence = 0; // No text similarity confidence
          targetEntry.matchMethod = "timestamp-only";

          console.log(
            `   ✅ Enriched input event #${phEvent.eventIndex}: "${newLogText}"`
          );
          console.log(
            `   📊 Match method: ${targetEntry.matchMethod}, time diff: ${bestMatch.timeDiff}ms`
          );
        }
      } else {
        console.log(
          `[ENRICH_DEBUG]   -> ⚠️ Match found, but node ID ${nodeId} not in map.`
        );
      }
    } else {
      console.log(
        `   ❌ No suitable rrweb match found for PostHog event #${phEvent.eventIndex}`
      );
    }
  }

  /*  await new Promise((resolve) => setTimeout(resolve, 10000000)); */

  /*   console.log("nfewfwnfwkn ", logEntries); */
  return logEntries;
}
/**
 * Matches PostHog navigation events ($pageview/$pageleave) with rrweb navigation events
 * for accurate timing while preserving PostHog metadata
 * @param {Array} postHogLogEntries - PostHog log entries including navigation events
 * @param {Array} rrwebEvents - All rrweb events
 * @param {Array} skips - Time skip adjustments
 * @param {string} recordingStartTime - Recording start timestamp
 * @returns {Array} Matched navigation events with rrweb timing
 */
function matchPostHogNavigationWithRrweb(
  postHogLogEntries,
  rrwebEvents,
  skips = [],
  recordingStartTime = null
) {
  console.log("🔍 Matching PostHog navigation events with rrweb events...");

  // Extract PostHog navigation events
  const postHogNavigationEvents = postHogLogEntries.filter((entry) => {
    return entry.text && entry.text.includes("Page view:");
  });

  console.log(
    `Found ${postHogNavigationEvents.length} PostHog navigation events`
  );

  // DEBUG: Show all PostHog navigation events found
  console.log(`📋 PostHog navigation events found:`);
  postHogNavigationEvents.forEach((event, i) => {
    console.log(
      `  ${i + 1}. "${event.text}" (eventIndex: ${event.eventIndex})`
    );
  });

  /*   let i = 0;
  while (i < 100000000000) {
    i++;
  } */

  // Extract rrweb navigation events
  const rrwebNavigationEvents = [];

  rrwebEvents.forEach((event, index) => {
    // Meta events (type 4) - actual page loads
    if (event.type === 4 && event.data?.href) {
      rrwebNavigationEvents.push({
        ...event,
        url: event.data.href,
        navigationType: "META_EVENT",
        eventIndex: index,
      });
    }

    // Custom events (type 5) with $pageview tag - SPA navigation
    if (
      event.type === 5 &&
      event.data?.tag === "$pageview" &&
      event.data?.payload?.href
    ) {
      rrwebNavigationEvents.push({
        ...event,
        url: event.data.payload.href,
        navigationType: "SPA_NAVIGATION",
        eventIndex: index,
      });
    }
  });

  console.log(`Found ${rrwebNavigationEvents.length} rrweb navigation events`);

  // DEBUG: Show all rrweb navigation events found
  console.log(`📋 rrweb navigation events found:`);
  rrwebNavigationEvents.forEach((event, i) => {
    console.log(
      `  ${i + 1}. "${event.url}" at ${event.timestamp}ms (type: ${
        event.navigationType
      })`
    );
  });

  if (
    postHogNavigationEvents.length === 0 ||
    rrwebNavigationEvents.length === 0
  ) {
    console.log("No navigation events to match - returning empty array");
    return [];
  }

  // Parse recording start time if provided
  let recordingStartTimeMs = null;
  if (recordingStartTime) {
    recordingStartTimeMs = new Date(recordingStartTime).getTime();
  }

  const rrwebSessionStart = rrwebEvents[0]?.timestamp;
  const matchedEvents = [];

  for (const phNavEvent of postHogNavigationEvents) {
    console.log(`\n🔍 Matching PostHog event: "${phNavEvent.text}"`);

    // Extract URL from PostHog event text
    const urlMatch = phNavEvent.text.match(/Page (?:view:|leave:) (.+)$/);
    if (!urlMatch) {
      console.log(`   ❌ Could not extract URL from PostHog event text`);
      continue;
    }

    const phUrl = urlMatch[1];
    const isEntry = phNavEvent.text.includes("Page view:");

    console.log(`   📍 URL: ${phUrl}`);
    console.log(`   📍 Type: ${isEntry ? "ENTRY" : "EXIT"}`);
    console.log(`   📍 PostHog timestamp: ${phNavEvent.originalAbsoluteMs}ms`);

    // Find matching rrweb events with same URL
    const candidateRrwebEvents = rrwebNavigationEvents.filter((rrwebEvent) => {
      return rrwebEvent.url === phUrl;
    });

    console.log(
      `   🔍 Found ${candidateRrwebEvents.length} rrweb events with matching URL`
    );

    if (candidateRrwebEvents.length === 0) {
      console.log(`   ❌ No rrweb events found with matching URL`);
      continue;
    }

    // Find the closest rrweb event by timestamp
    let bestMatch = null;
    let minTimeDelta = Infinity;

    for (const rrwebEvent of candidateRrwebEvents) {
      const timeDelta = Math.abs(
        rrwebEvent.originalTimestamp - phNavEvent.originalAbsoluteMs
      );

      console.log(
        "ejfnwejfnwejfnwejfnw ",
        rrwebEvent.originalTimestamp,
        " ",
        rrwebEvent.timestamp
      );
      console.log(
        `      - rrweb event at ${rrwebEvent.originalTimestamp}ms, delta: ${timeDelta}ms`
      );

      if (timeDelta < minTimeDelta) {
        minTimeDelta = timeDelta;
        bestMatch = rrwebEvent;
      }
    }

    if (bestMatch && minTimeDelta < 3000) {
      console.log(`   ✅ Best match found: delta ${minTimeDelta}ms`);

      // Calculate timing for the matched rrweb event (already adjusted)
      const adjustedTimestamp = bestMatch.timestamp;
      const isPreRecording =
        recordingStartTimeMs && adjustedTimestamp < recordingStartTimeMs;

      // Calculate timing relative to rrweb session start (no extra shift)
      const adjustedRelativeTimeMs = adjustedTimestamp - rrwebSessionStart;
      const originalRelativeTimeMs = adjustedRelativeTimeMs;

      // Create the matched event using PostHog text but rrweb timing
      const eventText = isEntry
        ? `Page view: ${phUrl}`
        : `Page leave: ${phUrl}`;
      const flaggedText = isPreRecording
        ? `[PRE-RECORDING (mention in the output)] ${eventText}`
        : eventText;

      const matchedEvent = {
        adjustedMs: adjustedRelativeTimeMs,
        originalMs: originalRelativeTimeMs,
        originalAbsoluteMs: adjustedTimestamp, // Use rrweb timing (adjusted)
        text: flaggedText,
        isPreRecording: isPreRecording,
        eventIndex: `matched-nav-${bestMatch.eventIndex}`,
        originalEvent: bestMatch, // Store the rrweb event
        source: "matched-navigation",
        navigationType: bestMatch.navigationType,
        matchedFromPostHog: phNavEvent.eventIndex, // Reference to original PostHog event
        timeDelta: minTimeDelta,
      };

      matchedEvents.push(matchedEvent);

      console.log(
        `   📝 Added matched navigation event at ${(
          adjustedRelativeTimeMs / 1000
        ).toFixed(3)}s`
      );
    } else {
      console.log(
        `   ❌ No suitable match found (best delta: ${minTimeDelta}ms)`
      );
      if (candidateRrwebEvents.length > 0) {
        console.log(`   🔍 Candidate rrweb events for this URL:`);
        candidateRrwebEvents.forEach((candidate, i) => {
          const delta = Math.abs(
            candidate.timestamp - phNavEvent.originalAbsoluteMs
          );
          console.log(
            `      ${i + 1}. timestamp: ${
              candidate.timestamp
            }ms, delta: ${delta}ms`
          );
        });
      }
    }
  }

  console.log(
    `✅ Successfully matched ${matchedEvents.length} navigation events`
  );
  return matchedEvents;
}

/**
 * Adds matched rrweb input events as separate entries in the activity log
 * @param {Array} logEntries - Current log entries
 * @param {Array} rrwebEvents - All rrweb events
 * @param {Array} skips - Time skip adjustments
 * @returns {Array} Enhanced log entries with rrweb input events added
 */
function addRrwebInputEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding matched rrweb input events to activity log...");

  // DEBUG: Log all entries to see what we're working with
  console.log(`   📊 Total log entries to examine: ${logEntries.length}`);

  // DEBUG: Check each entry individually
  const inputRelatedEntries = logEntries.filter(
    (entry) => entry.text && entry.text.includes("input")
  );
  console.log(
    `   📊 Entries with "input" in text: ${inputRelatedEntries.length}`
  );

  const entriesWithMatchedRrweb = logEntries.filter(
    (entry) => entry.matchedRrwebEvent
  );
  console.log(
    `   📊 Entries with matchedRrwebEvent: ${entriesWithMatchedRrweb.length}`
  );

  // DEBUG: Show details of entries with matchedRrwebEvent
  entriesWithMatchedRrweb.forEach((entry, index) => {
    console.log(
      `     Entry ${index + 1}: eventIndex=${
        entry.eventIndex
      }, text="${entry.text?.substring(0, 80)}..."`
    );
  });

  // Find log entries that have matched rrweb events
  const enrichedInputEntries = logEntries.filter(
    (entry) =>
      entry.matchedRrwebEvent && entry.text && entry.text.includes("input")
  );

  if (enrichedInputEntries.length === 0) {
    console.log("   No enriched input entries found - nothing to add");
    return logEntries;
  }

  console.log(
    `   Found ${enrichedInputEntries.length} enriched input entries to process`
  );

  // Build node map for element descriptions
  const idNodeMap = buildRrwebNodeMap(rrwebEvents);
  if (!idNodeMap) {
    console.log("   Could not build node map - skipping rrweb event addition");
    return logEntries;
  }

  // Get the initial recording start timestamp
  const initialRecordingStart = rrwebEvents[0]?.timestamp;
  if (!initialRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  try {
    console.log(
      `   [TIMELINE] initialRecordingStart adjusted=${initialRecordingStart} (${new Date(
        initialRecordingStart
      ).toISOString()})`
    );
    const firstInputEvt = rrwebEvents.find(
      (e) => e?.type === 3 && e?.data?.source === 5
    );
    if (firstInputEvt) {
      console.log(
        `   [TIMELINE] first rrweb input original=${
          firstInputEvt.originalTimestamp
        } (${
          firstInputEvt.originalTimestamp
            ? new Date(firstInputEvt.originalTimestamp).toISOString()
            : "n/a"
        }), adjusted=${firstInputEvt.timestamp} (${
          firstInputEvt.timestamp
            ? new Date(firstInputEvt.timestamp).toISOString()
            : "n/a"
        })`
      );
    }
  } catch (e) {
    console.log(
      "   [DEBUG] failed to log initialRecordingStart:",
      e?.message || e
    );
  }

  const newLogEntries = [...logEntries];

  for (const enrichedEntry of enrichedInputEntries) {
    const rrwebEvent = enrichedEntry.matchedRrwebEvent;
    const node = idNodeMap.get(rrwebEvent.data.id);

    if (!node) {
      console.log(
        `   ⚠️ Could not find node for rrweb event (nodeId: ${rrwebEvent.data.id})`
      );
      continue;
    }

    // Calculate timing for the rrweb event
    const adjustedAbsoluteMs = rrwebEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - initialRecordingStart;
    const originalRelativeTimeMs = adjustedRelativeTimeMs;
    try {
      const rawDelta =
        (rrwebEvent.originalTimestamp ?? adjustedAbsoluteMs) -
        (enrichedEntry.originalAbsoluteMs ?? enrichedEntry.adjustedMs);
      console.log(
        `   [ADD-LOG] PH original=${enrichedEntry.originalAbsoluteMs} (${
          enrichedEntry.originalAbsoluteMs
            ? new Date(enrichedEntry.originalAbsoluteMs).toISOString()
            : "n/a"
        }), rrweb original=${rrwebEvent.originalTimestamp} (${
          rrwebEvent.originalTimestamp
            ? new Date(rrwebEvent.originalTimestamp).toISOString()
            : "n/a"
        }), rrweb adjusted=${adjustedAbsoluteMs} (${new Date(
          adjustedAbsoluteMs
        ).toISOString()}), rawDelta=${rawDelta}ms`
      );
    } catch (_) {}

    // Attribute-based override: prefer candidates whose node attributes match PostHog element hints
    try {
      if (potentialMatches.length > 0) {
        // Build PostHog element hints (id/name/placeholder/type/tag)
        let phTag = null,
          phId = null,
          phName = null,
          phPlaceholder = null,
          phType = null;
        const phOrig = phEvent?.originalEvent;
        const phProps = phOrig?.properties || {};
        if (Array.isArray(phEvent?.elements) && phEvent.elements.length > 0) {
          const el = phEvent.elements[0];
          phTag = el.tag_name || null;
          const attrs = el.attributes || {};
          phId = attrs.id || null;
          phName = attrs.name || null;
          phPlaceholder = attrs.placeholder || null;
          phType = attrs.type || null;
        } else if (typeof phOrig?.elements_chain === "string") {
          const parsed = parsePostHogElementChain(phOrig.elements_chain) || [];
          const pref = parsed.find((e) => e?.tag_name === "input") || parsed[0];
          if (pref) {
            phTag = pref.tag_name || null;
            const attrs = pref.attributes || {};
            phId = attrs.id || null;
            phName = attrs.name || null;
            phPlaceholder = attrs.placeholder || null;
            phType = attrs.type || null;
          }
        } else {
          phTag = phProps.$el_tag_name || null;
          phId = phProps.$el_id || null;
          phName = phProps.$el_name || null;
          phPlaceholder = phProps.$el_placeholder || null;
          phType = phProps.$el_type || null;
        }
      }
    } catch (e) {}

    // Create descriptive text for the rrweb input event
    const attributes = node.attributes || {};
    const inputValue = rrwebEvent.data.text || "";

    let elementDescription = "input field";
    if (attributes.placeholder) {
      elementDescription = `input with placeholder "${attributes.placeholder}"`;
    } else if (attributes["aria-label"]) {
      elementDescription = `input with label "${attributes["aria-label"]}"`;
    } else if (attributes.name) {
      elementDescription = `input named "${attributes.name}"`;
    } else if (attributes.id) {
      elementDescription = `input with id "${attributes.id}"`;
    }

    const rrwebEventText = `changed ${elementDescription}: "${inputValue}"`;

    // Check if this would be a pre-recording event
    const isPreRecording = enrichedEntry.isPreRecording;
    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${rrwebEventText}`
      : rrwebEventText;

    // Create new log entry for the rrweb event
    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: adjustedAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-input-${rrwebEvent.timestamp}`,
      originalEvent: rrwebEvent,
      source: "rrweb-input",
      matchMethod: enrichedEntry.matchMethod,
      matchConfidence: enrichedEntry.matchConfidence,
      relatedPostHogEventIndex: enrichedEntry.eventIndex,
    };

    newLogEntries.push(newLogEntry);

    console.log(
      `   ✅ Added rrweb input event: "${inputValue}" at ${(
        adjustedRelativeTimeMs / 1000
      ).toFixed(3)}s`
    );
    console.log(`      Related to PostHog event #${enrichedEntry.eventIndex}`);
    console.log(
      `      Match method: ${enrichedEntry.matchMethod}, confidence: ${(
        enrichedEntry.matchConfidence * 100
      ).toFixed(1)}%`
    );
  }

  // Sort all entries by adjusted time
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  const addedCount = newLogEntries.length - logEntries.length;
  console.log(`✅ Added ${addedCount} rrweb input events to the activity log`);
  console.log(
    `📊 Total log entries: ${newLogEntries.length} (was ${logEntries.length})`
  );

  return newLogEntries;
}

function addRrwebScrollEventsToLog(logEntries, rrwebEvents, skips = []) {
  console.log("🔍 Adding rrweb scroll events to activity log...");

  if (!rrwebEvents || rrwebEvents.length === 0) {
    console.log("   No rrweb events provided - nothing to add");
    return logEntries;
  }

  // Filter for scroll events: type 3 (Incremental Snapshot) and data.source 3 (Scroll)
  const scrollEvents = rrwebEvents.filter(
    (event) => event.type === 3 && event.data?.source === 3
  );

  if (scrollEvents.length === 0) {
    console.log("   No scroll events found in rrweb data");
    return logEntries;
  }

  console.log(`   Found ${scrollEvents.length} scroll events to process`);

  // Get the initial recording start timestamp from the ADJUSTED timeline
  const adjustedRecordingStart = rrwebEvents[0]?.timestamp;
  if (!adjustedRecordingStart) {
    console.log("   Could not determine recording start time - skipping");
    return logEntries;
  }
  // Also get the original start time for calculating originalRelativeTimeMs
  const originalRecordingStart =
    rrwebEvents[0]?.originalTimestamp || adjustedRecordingStart;

  const newLogEntries = [...logEntries];

  // Track previous scroll positions to determine direction and distance
  let previousScrollY = null;
  let previousScrollX = null;

  for (const scrollEvent of scrollEvents) {
    // scrollEvent.timestamp is the ADJUSTED absolute timestamp
    const adjustedAbsoluteMs = scrollEvent.timestamp;
    const adjustedRelativeTimeMs = adjustedAbsoluteMs - adjustedRecordingStart;

    // We need original timestamps to calculate originalRelativeTimeMs correctly
    const originalAbsoluteMs =
      scrollEvent.originalTimestamp || scrollEvent.timestamp;
    const originalRelativeTimeMs = originalAbsoluteMs - originalRecordingStart;

    // Extract scroll data
    const scrollData = scrollEvent.data;
    const scrollX = scrollData.x || 0;
    const scrollY = scrollData.y || 0;

    // Determine scroll direction and amount based on available data
    let scrollText;

    if (previousScrollY !== null && scrollData.y !== undefined) {
      const deltaY = scrollY - previousScrollY;
      const deltaX = scrollX - previousScrollX;

      if (Math.abs(deltaY) > Math.abs(deltaX)) {
        // Vertical scrolling is dominant
        if (deltaY > 0) {
          scrollText = `scrolled down by ${Math.abs(deltaY).toFixed(0)}px`;
        } else if (deltaY < 0) {
          scrollText = `scrolled up by ${Math.abs(deltaY).toFixed(0)}px`;
        } else {
          scrollText = "scrolled";
        }
      } else if (Math.abs(deltaX) > 0) {
        // Horizontal scrolling
        if (deltaX > 0) {
          scrollText = `scrolled right by ${Math.abs(deltaX).toFixed(0)}px`;
        } else {
          scrollText = `scrolled left by ${Math.abs(deltaX).toFixed(0)}px`;
        }
      } else {
        scrollText = "scrolled";
      }
    } else {
      // First scroll event or no position data - use generic format
      scrollText = "scrolled";
    }

    // Update previous positions for next iteration
    previousScrollY = scrollY;
    previousScrollX = scrollX;

    // Check if this would be a pre-recording event (using session start time logic)
    const isPreRecording = false; // Scroll events are from rrweb so they're during recording

    const flaggedText = isPreRecording
      ? `[PRE-RECORDING (mention in the output)] ${scrollText}`
      : scrollText;

    // Create new log entry for the scroll event
    const newLogEntry = {
      adjustedMs: adjustedRelativeTimeMs,
      originalMs: originalRelativeTimeMs,
      originalAbsoluteMs: originalAbsoluteMs,
      text: flaggedText,
      isPreRecording: isPreRecording,
      eventIndex: `rrweb-scroll-${scrollEvent.timestamp}`,
      originalEvent: scrollEvent,
      source: "rrweb-scroll",
      scrollData: {
        x: scrollX,
        y: scrollY,
        deltaX: previousScrollX !== null ? scrollX - previousScrollX : 0,
        deltaY: previousScrollY !== null ? scrollY - previousScrollY : 0,
      },
    };

    newLogEntries.push(newLogEntry);
  }

  // Sort by time to maintain chronological order
  newLogEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  console.log(`   📊 Added ${scrollEvents.length} scroll events to the log`);

  return newLogEntries;
}
function generateStandardLog(
  responseData,
  recordingDuration,
  skips = [],
  recordingStartTime = null,
  rrwebEvents = null
) {
  const logEntries = [];
  let events = responseData.results;
  if (!events || events.length === 0) return [];

  // Log navigation event source
  if (rrwebEvents && rrwebEvents.length > 0) {
    console.log(
      "📄 Matching PostHog navigation events to rrweb events for accurate timing"
    );
  } else {
    console.log("📄 Using PostHog events for page navigation (entries/exits)");
  }

  // NEW: Filter out noisy events that don't represent direct user actions for the log.
  const eventsToFilterOut = ["$rageclick", "$identify"];
  const originalEventCount = events.length;
  events = events.filter((event) => !eventsToFilterOut.includes(event.event));

  events.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  const sessionStartTime = new Date(events[0].timestamp).getTime();

  // Make postHogClickTimestamps a local variable to prevent race conditions
  const postHogClickTimestamps = [];

  // Parse recording start time if provided
  let recordingStartTimeMs = null;
  if (recordingStartTime) {
    recordingStartTimeMs = new Date(recordingStartTime).getTime();
    console.log(`\n📊 SESSION REPLAY SYNCHRONIZATION ANALYSIS:`);
    console.log(
      `Session Start Time: ${new Date(sessionStartTime).toISOString()}`
    );
    console.log(
      `Recording Start Time: ${new Date(recordingStartTimeMs).toISOString()}`
    );
    console.log(
      `Pre-recording Duration: ${(
        (recordingStartTimeMs - sessionStartTime) /
        1000
      ).toFixed(2)} seconds`
    );
    console.log(`=`.repeat(60));
  }

  const adjustedSessionStartTime = sessionStartTime; // Use original session start time

  let preRecordingEventCount = 0;
  let recordingAvailableEventIndex = -1;

  events.forEach((event, index) => {
    const properties = event.properties;
    const eventTime = new Date(event.timestamp);
    const originalTimestamp = eventTime.getTime();

    // Determine if this event occurred before recording started
    const isPreRecording =
      recordingStartTimeMs && originalTimestamp < recordingStartTimeMs;
    if (isPreRecording) {
      preRecordingEventCount++;
    }

    // Mark the first event where recording becomes available
    if (
      recordingStartTimeMs &&
      recordingAvailableEventIndex === -1 &&
      originalTimestamp >= recordingStartTimeMs
    ) {
      recordingAvailableEventIndex = index;
    }

    // Calculate the total time shift for this event's timestamp based on the skips
    const timeShift = calculateTimeShiftForTimestamp(originalTimestamp, skips);

    // The event's original position in time relative to the session start
    const originalRelativeTimeMs = originalTimestamp - sessionStartTime;
    // The event's new position in the processed video's timeline
    const adjustedRelativeTimeMs = originalRelativeTimeMs - timeShift;

    const primaryElement =
      event.elements && event.elements[0] ? event.elements[0] : null;

    let logText = null;

    switch (event.event) {
      case "$pageview":
        // Always include PostHog pageviews - they will be matched to rrweb events later
        logText = `Page view: ${properties.$current_url}`;
        break;
      case "$pageleave":
        // Always include PostHog pageleaves - they will be matched to rrweb events later
        logText = `Page leave: ${properties.$current_url}`;
        break;
      case "$autocapture": {
        const eventType = properties.$event_type; // 'click', 'change', 'submit'

        const tagName = primaryElement?.tag_name;
        const text = properties?.$el_text;

        // Store enriched element data back to the event for later use
        if (eventType === "click" && (tagName || text)) {
          if (!event.elements || event.elements.length === 0) {
            event.elements = [
              {
                tag_name: tagName || "unknown",
                textContent: text || "",
                attributes: primaryElement?.attributes || {},
              },
            ];
          }
        }

        let elDesc;
        if (text) {
          // Truncate long text for readability in logs
          const truncatedText =
            text.length > 50 ? text.substring(0, 47) + "..." : text;
          elDesc = `${tagName} with text: "${truncatedText}"`;
        } else {
          elDesc = `something (find in the recording)`;
        }

        if (eventType === "click") {
          logText = `Clicked ${elDesc}`;
        } else if (eventType === "change") {
          console.log("Fmewkfmwefkwm ", properties);
          const value = properties.$el_value;
          if (value !== undefined && value !== null) {
            logText = `Some input changed to "${value}"`;
          } else {
            logText = `Some input changed`;
          }
        } else if (eventType === "submit") {
          logText = `Submitted form via ${elDesc}`;
        }
        break;
      }
      default:
        // This will log any other events, such as custom events, $identify, etc.
        logText = `Event: "${event.event}"`;
        break;
    }

    if (logText) {
      // Add visual flag for events that occurred before recording started
      const flaggedText = isPreRecording
        ? `[PRE-RECORDING (mention in the output)] ${logText}`
        : logText;

      // CRITICAL DEBUG: Track PostHog timestamps to check for duplicates
      if (logText.includes("Clicked")) {
        // Check if we're getting duplicate timestamps
        postHogClickTimestamps.push({
          index: index,
          rawTimestamp: event.timestamp,
          originalTimestamp: originalTimestamp,
          logText: logText,
        });

        if (postHogClickTimestamps.length > 1) {
          const allSame = postHogClickTimestamps.every(
            (t) =>
              t.originalTimestamp ===
              postHogClickTimestamps[0].originalTimestamp
          );
          if (allSame) {
            // FIX: Add small offset to make timestamps unique
            let offsetMs = (postHogClickTimestamps.length - 1) * 1000; // 1 second apart
            originalTimestamp = originalTimestamp + offsetMs;
            originalRelativeTimeMs = originalTimestamp - sessionStartTime;
            adjustedRelativeTimeMs = originalRelativeTimeMs - timeShift;

            console.log(
              `   Fixed timestamp for event ${index}: ${originalTimestamp}ms (offset: +${offsetMs}ms)`
            );
          } else {
            /*  console.log(`✅ PostHog events have different timestamps:`);
            postHogClickTimestamps.forEach((t) => {
              console.log(`      Event ${t.index}: ${t.originalTimestamp}ms`);
            }); */
          }
        }
      }

      logEntries.push({
        adjustedMs: adjustedRelativeTimeMs,
        originalMs: originalRelativeTimeMs,
        originalAbsoluteMs: originalTimestamp, // The absolute original timestamp (possibly fixed)
        text: flaggedText,
        isPreRecording: isPreRecording,
        eventIndex: index,
        // 🔥 FIX: Preserve the original PostHog event data including elements and properties
        originalEvent: event, // This contains the enriched element data from log generation
        properties: event.properties, // Direct access to properties for convenience
        elements: event.elements, // Direct access to elements for convenience
      });
    }
  });

  /*   let i = 0
  while (i < 100000000000000) {
    i++
  } */

  // Add a marker for when replay becomes available if we have recording start time
  if (recordingStartTimeMs && recordingAvailableEventIndex !== -1) {
    const recordingStartRelativeMs = recordingStartTimeMs - sessionStartTime;
    const timeShift = calculateTimeShiftForTimestamp(
      recordingStartTimeMs,
      skips
    );
    const adjustedRecordingStartMs = recordingStartRelativeMs - timeShift;

    logEntries.push({
      adjustedMs: adjustedRecordingStartMs,
      originalMs: recordingStartRelativeMs,
      text: "🎥 [REPLAY AVAILABLE] Session recording starts here",
      isPreRecording: false,
      eventIndex: -1, // Use -1 to avoid conflicts with real event indices
      isReplayMarker: true,
    });
  }

  // Navigation events are handled by matching PostHog events to rrweb events
  // This ensures accurate timing from rrweb while preserving PostHog metadata
  console.log(
    "📄 Navigation events will use PostHog→rrweb matching for accurate timing"
  );

  // Decide which duration to use
  let finalDurationMs;
  if (recordingDuration) {
    const parsedDurationMs = Math.round(parseFloat(recordingDuration) * 1000);
    if (!isNaN(parsedDurationMs)) {
      finalDurationMs = parsedDurationMs;
    }
  }

  // Fallback if recordingDuration was not available/valid
  if (finalDurationMs === undefined && events && events.length > 0) {
    const sessionStartTime = new Date(events[0].timestamp).getTime();
    const lastEvent = events[events.length - 1];
    const lastEventTime = new Date(lastEvent.timestamp).getTime();
    finalDurationMs = lastEventTime - sessionStartTime;
  }

  /*   if (finalDurationMs !== undefined) {
    const sessionEndTime = sessionStartTime + finalDurationMs;
    const totalShift = calculateTimeShiftForTimestamp(sessionEndTime, skips);
    const adjustedDurationMs = finalDurationMs - totalShift;
    const originalDurationMs = finalDurationMs;

    logEntries.push({
      adjustedMs: adjustedDurationMs > 0 ? adjustedDurationMs : 0,
      originalMs: originalDurationMs,
      text: "Session ended",
    });
  } */

  // Sort entries by adjusted time to ensure proper chronological order
  logEntries.sort((a, b) => a.adjustedMs - b.adjustedMs);

  // Print synchronization summary if we have recording start time
  if (recordingStartTimeMs) {
    console.log(`\n📋 SYNCHRONIZATION SUMMARY:`);
    console.log(`Total Events: ${events.length}`);
    console.log(`Pre-recording Events: ${preRecordingEventCount}`);
    console.log(
      `Post-recording Events: ${events.length - preRecordingEventCount}`
    );
    console.log(
      `Recording Available at Event Index: ${recordingAvailableEventIndex}`
    );
    console.log(`=`.repeat(60));

    // Show a sample of pre-recording events
    const preRecordingEntries = logEntries.filter(
      (entry) => entry.isPreRecording
    );
    if (preRecordingEntries.length > 0) {
      console.log(
        `\n🔴 PRE-RECORDING EVENTS (${preRecordingEntries.length} total):`
      );
      preRecordingEntries.slice(0, 5).forEach((entry, i) => {
        const displayText = entry.text
          ? entry.text.replace("[PRE-RECORDING (mention in the output)] ", "")
          : "[REMOVED]";
        console.log(
          `  ${i + 1}. [${formatMilliseconds(entry.originalMs)}] ${displayText}`
        );
      });
      if (preRecordingEntries.length > 5) {
        console.log(
          `  ... and ${
            preRecordingEntries.length - 5
          } more pre-recording events`
        );
      }
      console.log(``);
    }
  }

  return logEntries;
}

/**
 * Injects custom "marker" events into the rrweb event stream to denote clip boundaries.
 * These markers can be useful for debugging the recording and clipping process.
 * @param {Array<Object>} events - The original rrweb event stream.
 * @param {Array<Object>} clips_folder - The array of clip segments from defineLogicalClipSegments.
 * @returns {Array<Object>} A new event stream with marker events injected and sorted.
 */
function injectClipMarkersIntoEventStream(
  events,
  clips_folder,
  opts = {}
) {
  if (
    !events ||
    events.length === 0 ||
    !clips_folder ||
    clips_folder.length === 0
  ) {
    console.warn(
      "[Marker Injection] No events or clips_folder provided, skipping injection."
    );
    return events;
  }

  const markerEvents = [];
  let yellowSeq = 0; // global order for yellow markers in this stream
  const sessionId =
    (opts && opts.sessionId) ||
    (typeof window === "undefined" ? process?.env?.SESSION_ID : null) ||
    null;
  const nextBeaconId = () => deterministicBeaconId(sessionId, yellowSeq++);
  // The clip start/end times are relative to the first event. We need the absolute timestamp of the first event.
  const firstEventTimestamp = events[0].timestamp;

  // Overlay windows to align yellow markers with extended boundaries
  const overlayInstructions = Array.isArray(opts?.overlayInstructions)
    ? opts.overlayInstructions
    : [];
  const overlayWindows = overlayInstructions
    .map((ov) => ({ startMs: ov.showAt, endMs: ov.hideAt }))
    .filter(
      (w) =>
        Number.isFinite(w.startMs) &&
        Number.isFinite(w.endMs) &&
        w.endMs > w.startMs
    )
    .sort((a, b) => a.startMs - b.startMs);
  // Tolerance window for overlays starting just after a planned end
  const TOL_AFTER_END_MS =
    typeof opts?.overlayToleranceAfterEndMs === "number"
      ? opts.overlayToleranceAfterEndMs
      : 0;
  const EPS_MS = typeof opts?.epsilonMs === "number" ? opts.epsilonMs : 1;

  clips_folder.forEach((clip, index) => {
    // Guard against missing logical times
    const logicalStart = Number.isFinite(clip.logicalStart)
      ? clip.logicalStart
      : clip.start;
    const logicalEnd = Number.isFinite(clip.logicalEnd)
      ? clip.logicalEnd
      : clip.end;
    if (!Number.isFinite(logicalStart) || !Number.isFinite(logicalEnd)) {
      console.warn(
        `[Marker Injection] Skipping clip index ${index} due to invalid logical start/end times.`
      );
      return;
    }
    // Convert relative logical seconds back to absolute rrweb timestamps (in milliseconds)
    const startTimestamp = firstEventTimestamp + logicalStart * 1000;
    const endTimestamp = firstEventTimestamp + logicalEnd * 1000;
    const clipDurationMs = endTimestamp - startTimestamp;
    console.log(
      `[DEBUG] injectClipMarkers: Processing clip #${index}, type: ${
        clip.type
      }, duration: ${clipDurationMs.toFixed(
        2
      )}ms, logicalStart: ${logicalStart.toFixed(
        3
      )}s, logicalEnd: ${logicalEnd.toFixed(3)}s`
    );

    if (clip.type === "click") {
      // Show yellow markers for click clip boundaries (start and end).
      const endMarkerType = "END_CLICK_CLIP";
      const color = "yellow";

      // END marker (visible 2s)
      const endBeaconId = nextBeaconId();

      // ==================================================================
      // <<< 🎯 ADD THIS CONSOLE.LOG FOR CLICK CLIPS 🎯 >>>
      // ==================================================================
      console.log(
        `[YELLOW BEACON PLAN] Marking end of '${clip.type}' clip #${
          clip.originalEvent?.eventIndex || `click-${index}`
        } at timestamp ${endTimestamp - 1}ms (relative: ${
          (endTimestamp - 1 - firstEventTimestamp) / 1000
        }s)`
      );
      // ==================================================================

      // END marker (1ms prior to avoid collision)
      markerEvents.push({
        type: 5,
        timestamp: endTimestamp - 1,
        data: {
          tag: "CLIP_MARKER",
          payload: {
            markerType: endMarkerType,
            clipIndex: clip.originalEvent?.eventIndex || `click-${index}`,
            clipType: clip.type,
            color,
            beaconId: endBeaconId,
            displayDurationMs: 2000,
            description: `End of ${clip.type} clip #${
              clip.originalEvent?.eventIndex || `click-${index}`
            }`,
          },
        },
      });
      return;
    }

    // For noclick: inject yellow beacons for every 10s sub-clip
    const maxSeconds =
      (typeof NONCLICK_MAX_CLIP_DURATION_MS === "number"
        ? NONCLICK_MAX_CLIP_DURATION_MS
        : 10000) / 1000;
    const totalLen = Math.max(0, logicalEnd - logicalStart);
    if (totalLen < 0.05) return;

    let subStart = logicalStart;
    let subIndex = 0;
    while (subStart < logicalEnd - 1e-6) {
      let subEnd = Math.min(subStart + maxSeconds, logicalEnd);
      const subStartTs = firstEventTimestamp + subStart * 1000;
      let subEndTs = firstEventTimestamp + subEnd * 1000;

      // If overlays cross or begin just after the planned end, extend the sub-clip end
      if (overlayWindows.length > 0) {
        let adjustedEndTs = subEndTs;
        // Overlays that cross the end boundary
        const crossing = overlayWindows.filter(
          (w) => w.startMs < adjustedEndTs && w.endMs > adjustedEndTs
        );
        if (crossing.length > 0) {
          const maxEnd = Math.max(...crossing.map((w) => w.endMs));
          adjustedEndTs = Math.max(adjustedEndTs, maxEnd + EPS_MS);
        }
        // Overlays that start shortly after planned end
        const nearAfter = overlayWindows.filter(
          (w) =>
            w.startMs >= adjustedEndTs &&
            w.startMs <= adjustedEndTs + TOL_AFTER_END_MS &&
            w.endMs > adjustedEndTs
        );
        if (nearAfter.length > 0) {
          const maxEnd = Math.max(...nearAfter.map((w) => w.endMs));
          adjustedEndTs = Math.max(adjustedEndTs, maxEnd + EPS_MS);
        }
        // Clamp to the logical end of the parent clip (already adjusted elsewhere)
        const clipEndAbs = firstEventTimestamp + logicalEnd * 1000;
        if (adjustedEndTs > subEndTs) {
          const clamped = Math.min(adjustedEndTs, clipEndAbs);
          if (clamped !== subEndTs) {
            console.log(
              `[YELLOW BEACON PLAN] Extending noclick sub-clip #${index}.${subIndex} end from ${subEndTs}ms to ${clamped}ms due to inactivity overlay alignment.`
            );
            subEndTs = clamped;
            subEnd = (subEndTs - firstEventTimestamp) / 1000;
          }
        }
      }

      // ==================================================================
      // <<< 🎯 ADD THIS CONSOLE.LOG FOR NO-CLICK CLIPS 🎯 >>>
      // ==================================================================
      console.log(
        `[YELLOW BEACON PLAN] Marking end of '${
          clip.type
        }' sub-clip #${index}.${subIndex} at timestamp ${
          subEndTs - 1
        }ms (relative: ${(subEndTs - 1 - firstEventTimestamp) / 1000}s)`
      );
      // ==================================================================

      // END marker for noclick sub-clip (yellow)
      const beaconId = nextBeaconId();
      markerEvents.push({
        type: 5,
        timestamp: subEndTs - 1,
        data: {
          tag: "CLIP_MARKER",
          payload: {
            markerType: "END_NOCLICK_SUBCLIP",
            clipIndex:
              clip.originalEvent?.eventIndex || `noclick-${index}-${subIndex}`,
            clipType: "noclick",
            color: "yellow",
            beaconId,
            displayDurationMs: 2000,
            description: `End of noclick clip #${index}.${subIndex}`,
          },
        },
      });

      // Add a dummy "anchor" event immediately after the marker to ensure a
      // consistent pause duration, regardless of any subsequent user inactivity.
      markerEvents.push({
        type: 5, // Custom event
        timestamp: subEndTs,
        data: {
          tag: "PAUSE_ANCHOR",
          payload: { forClipIndex: `noclick-${index}-${subIndex}` },
        },
      });

      subIndex += 1;
      subStart = subEnd;
    }
  });

  // Combine original events with new marker events
  const combinedEvents = [...events, ...markerEvents];

  // Re-sort this stream by timestamp to ensure chronological order
  combinedEvents.sort((a, b) => a.timestamp - b.timestamp);

  console.log(
    `✅ Successfully injected and sorted ${markerEvents.length} marker events.`
  );
  return combinedEvents;
}

/**
 * Builds a comprehensive map of all rrweb node IDs to their node data
 * @param {Array} events - rrweb events
 * @returns {Map|null} Map of node IDs to node objects, or null if failed
 */
function buildRrwebNodeMap(events) {
  const fullSnapshot = events.find((e) => e.type === EventType.FullSnapshot);
  if (!fullSnapshot || !fullSnapshot.data || !fullSnapshot.data.node) {
    console.warn("No valid full snapshot event found. Cannot build node map.");
    return null;
  }

  const idNodeMap = new Map();

  // Helper to recursively add a node and its children to the map
  function addNodeToMap(node) {
    if (!node) return;
    if (node.id !== undefined) {
      // We store the serialized node itself, which contains its attributes.
      idNodeMap.set(node.id, node);
    }
    if (node.childNodes) {
      for (const child of node.childNodes) {
        addNodeToMap(child);
      }
    }
  }

  // Start with the full snapshot
  addNodeToMap(fullSnapshot.data.node);

  // Add nodes from incremental mutations
  const mutationEvents = events.filter(
    (e) =>
      e.type === EventType.IncrementalSnapshot &&
      e.data.source === IncrementalSource.Mutation
  );

  for (const event of mutationEvents) {
    if (event.data.adds) {
      for (const added of event.data.adds) {
        addNodeToMap(added.node);
      }
    }
    // We don't need to process `removes` because we want to find any node
    // that *ever* existed. An input event could have occurred on a node
    // that was later removed.
  }
  return idNodeMap;
}

/**
 * Merges short 'noclick' segments that immediately precede a 'click' segment
 * into the previous 'noclick' segment. This prevents the creation of tiny,
 * unwanted pre-click clips_folder.
 *
 * @param {Array<Object>} allClips - The initial array of logically defined clips_folder.
 * @param  [mergeThresholdSeconds=3.0] - The max duration for a noclick clip to be considered for merging.
 * @returns {Array<Object>} A new array of clips_folder with the short segments merged.
 */
function mergeShortPreClickSegments(allClips, mergeThresholdSeconds = 3.0) {
  if (!allClips || allClips.length < 2) {
    return allClips;
  }

  const mergedClips = [];
  for (let i = 0; i < allClips.length; i++) {
    const current = allClips[i];
    const next = i < allClips.length - 1 ? allClips[i + 1] : null;

    // Check for the specific pattern: a short 'noclick' clip followed by a 'click' clip.
    const isShortPreClickGap =
      current.type === "noclick" &&
      current.end - current.start < mergeThresholdSeconds &&
      next &&
      next.type === "click";

    if (isShortPreClickGap) {
      const previousClip =
        mergedClips.length > 0 ? mergedClips[mergedClips.length - 1] : null;

      // If the previous clip was also a 'noclick' clip, extend it to cover this short gap.
      if (previousClip && previousClip.type === "noclick") {
        console.log(
          `[MERGE-LOGIC] Merging short pre-click gap (${(
            current.end - current.start
          ).toFixed(2)}s) into previous noclick clip.`
        );
        // Extend the end time of the *previous* clip to the end time of the *current* short clip.
        previousClip.end = current.end;
        previousClip.logicalEnd = current.logicalEnd;
        // We do NOT add the 'current' clip to the mergedClips array, effectively deleting it.
      } else {
        // This short noclick clip is at the beginning of the video, keep it.
        mergedClips.push(current);
      }
    } else {
      // This clip does not meet the merge criteria, so add it to the results.
      mergedClips.push(current);
    }
  }

  console.log(
    `[MERGE-LOGIC] Initial clips_folder: ${allClips.length}, Merged clips_folder: ${mergedClips.length}`
  );
  return mergedClips;
}

/**
 * Adjusts overlay instruction timestamps to account for injected pauses.
 * @param {Array<Object>} overlayInstructions - The original overlay instructions.
 * @param {Array<Object>} eventsBeforePauses - The event stream before pauses were injected (but with markers).
 * @param {number} pauseDurationMs - The duration of the pause to inject in milliseconds.
 * @returns {Array<Object>} A new array of overlay instructions with adjusted timestamps.
 */
function adjustOverlayInstructionsForPauses(
  overlayInstructions,
  eventsBeforePauses,
  pauseDurationMs = 2000
) {
  console.log(
    "[OVERLAY_ADJUST] Adjusting overlay instructions for injected pauses..."
  );
  if (!overlayInstructions || overlayInstructions.length === 0) {
    return [];
  }

  // Create a sorted list of pause marker timestamps for efficient lookup
  const pauseMarkerTimestamps = eventsBeforePauses
    .filter(
      (event) =>
        event.type === 5 &&
        event.data?.tag === "CLIP_MARKER" &&
        event.data?.payload?.color === "yellow"
    )
    .map((event) => event.timestamp)
    .sort((a, b) => a - b);

  if (pauseMarkerTimestamps.length === 0) {
    console.log(
      "[OVERLAY_ADJUST] No pause markers found, no adjustments needed."
    );
    return overlayInstructions;
  }

  const calculateShift = (targetTimestamp) => {
    let shift = 0;
    for (const markerTs of pauseMarkerTimestamps) {
      if (markerTs < targetTimestamp) {
        // Two pauses per yellow marker: 2s before, 2s at marker
        shift += pauseDurationMs * 2;
      } else {
        break; // Timestamps are sorted
      }
    }
    return shift;
  };

  const adjustedInstructions = overlayInstructions.map((ov) => {
    const showShift = calculateShift(ov.showAt);
    const originalDuration = ov.hideAt - ov.showAt;
    const newShowAt = ov.showAt + showShift;
    const newHideAt = newShowAt + originalDuration;

    console.log(
      `[OVERLAY_ADJUST] Adjusting overlay: showAt ${ov.showAt} -> ${newShowAt} (+${showShift}ms), duration preserved at ${originalDuration}ms`
    );

    return {
      ...ov,
      showAt: newShowAt,
      hideAt: newHideAt,
    };
  });

  console.log(
    `[OVERLAY_ADJUST] Finished adjusting ${adjustedInstructions.length} overlay instructions.`
  );
  return adjustedInstructions;
}
async function removeSegmentsFromVideo(
  inputPath,
  outputPath,
  intervalsToRemove
) {
  console.log(
    `[POST-PROCESS] Attempting to remove ${intervalsToRemove.length} freeze segments from video.`
  );
  if (!intervalsToRemove || intervalsToRemove.length === 0) {
    console.log(
      `[POST-PROCESS] No segments to remove. The clip will not be changed.`
    );
    // If no changes, we need to ensure the output path exists.
    // The caller expects the final clip at outputPath.
    try {
      await fs.copyFile(inputPath, outputPath);
    } catch (copyError) {
      console.error(
        `[POST-PROCESS] Error copying clip for no-op removal:`,
        copyError
      );
      return false;
    }
    return true;
  }

  const ffmpegCommand = ffmpegStatic || "ffmpeg";
  const ffprobeCommand = ffprobeStatic?.path || "ffprobe";
  const tempDir = os.tmpdir();
  const tempClipPaths = [];
  let concatListPath = "";

  try {
    // 1. Get total duration of the video
    const probeResult = spawnSync(
      ffprobeCommand,
      [
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        inputPath,
      ],
      { encoding: "utf-8" }
    );

    if (probeResult.status !== 0 || !probeResult.stdout) {
      throw new Error(
        `ffprobe failed to get clip duration. Stderr: ${probeResult.stderr}`
      );
    }
    const totalDuration = parseFloat(probeResult.stdout);

    // 2. Define the "good" segments (the parts we want to keep)
    const goodSegments = [];
    let lastEndTime = 0;

    intervalsToRemove.sort((a, b) => a.start - b.start);

    for (const interval of intervalsToRemove) {
      if (interval.start > lastEndTime) {
        goodSegments.push({ start: lastEndTime, end: interval.start });
      }
      lastEndTime = Math.max(lastEndTime, interval.end);
    }

    if (lastEndTime < totalDuration) {
      goodSegments.push({ start: lastEndTime, end: totalDuration });
    }

    console.log(
      `[POST-PROCESS] Defined ${goodSegments.length} segments to keep.`
    );
    if (goodSegments.length === 0) {
      console.warn(
        "[POST-PROCESS] All clip content was marked as a freeze segment. Aborting removal."
      );
      return false;
    }

    // 3. Trim the clip into clips_folder for each "good" segment
    for (let i = 0; i < goodSegments.length; i++) {
      const segment = goodSegments[i];
      const duration = segment.end - segment.start;
      if (duration < 0.1) continue; // Skip tiny segments

      const tempClipPath = path.join(
        tempDir,
        `good_segment_${i}_${crypto.randomBytes(4).toString("hex")}.mp4`
      );
      const trimSuccess = await trimVideo(
        inputPath,
        tempClipPath,
        segment.start,
        duration
      );
      if (trimSuccess) {
        tempClipPaths.push(tempClipPath);
      } else {
        console.warn(
          `[POST-PROCESS] Failed to trim segment ${i}. It will be excluded.`
        );
      }
    }

    if (tempClipPaths.length === 0) {
      console.error(
        "[POST-PROCESS] Failed to create any good segments. Aborting removal."
      );
      return false;
    }

    // 4. Create a concat list file
    concatListPath = path.join(
      tempDir,
      `concat_list_${crypto.randomBytes(6).toString("hex")}.txt`
    );
    const concatContent = tempClipPaths
      .map((p) => `file '${p.replace(/\\/g, "/")}'`)
      .join("\n");
    await fs.writeFile(concatListPath, concatContent);

    // 5. Concatenate the clips_folder using stream copy
    const concatArgs = [
      "-f",
      "concat",
      "-safe",
      "0",
      "-i",
      concatListPath,
      "-c",
      "copy",
      "-y",
      outputPath,
    ];

    const concatResult = spawnSync(ffmpegCommand, concatArgs, {
      stdio: "inherit",
    });
    if (concatResult.status !== 0) {
      throw new Error(
        `ffmpeg concatenation failed with code ${concatResult.status}.`
      );
    }

    console.log(
      `[POST-PROCESS] Successfully rebuilt clip without freeze segments at: ${outputPath}`
    );
    return true;
  } catch (error) {
    console.error("[POST-PROCESS] Error removing freeze segments:", error);
    return false;
  } finally {
    // 6. Cleanup temporary files
    if (concatListPath) await fs.unlink(concatListPath).catch(() => {});
    for (const clipPath of tempClipPaths) {
      await fs.unlink(clipPath).catch(() => {});
    }
  }
}

/**
 * V3C: Adjusts noclick clip boundaries to fully contain inactivity overlays that cross
 * the clip end, or start just after it. Also pushes the start of the next clip forward
 * to the new end to keep the timeline seamless. This avoids overlays spanning two clips_folder.
 *
 * @param {Array<Object>} initialClips - The clip segments from defineLogicalClipSegments.
 * @param {Array<Object>} overlayInstructions - The inactivity overlay instructions.
 * @param {number} firstEventTimestamp - The absolute timestamp of the first rrweb event.
 * @returns {Array<Object>} A new array of clips_folder with adjusted boundaries.
 */
function adjustClipBoundariesForOverlays_V3C(
  initialClips,
  overlayInstructions,
  firstEventTimestamp
) {
  console.log(
    "\n=================================================================="
  );
  console.log(
    "====== [DEBUG] Running adjustClipBoundariesForOverlays_V3C (Crossing + Near-After) ======"
  );
  console.log(
    "=================================================================="
  );

  if (!overlayInstructions || overlayInstructions.length === 0) {
    console.log(
      "[DEBUG] No overlay instructions provided. No adjustments will be made."
    );
    return initialClips;
  }

  const adjustedClips = JSON.parse(JSON.stringify(initialClips)); // Deep copy
  const overlayWindows = overlayInstructions.map((ov) => ({
    startMs: ov.showAt,
    endMs: ov.hideAt,
    message: ov.message,
  }));

  // Iterate up to the second-to-last clip, as we need to adjust the 'next' clip
  for (let i = 0; i < adjustedClips.length - 1; i++) {
    const currentClip = adjustedClips[i];
    const nextClip = adjustedClips[i + 1];

    console.log(
      `\n--- [DEBUG] Analyzing Clip #${i} (Type: ${currentClip.type}) ---`
    );

    // Adjust any clip type (click or noclick) if an inactivity overlay crosses its end.
    // This ensures overlays don't span two adjacent clips_folder.

    // Convert clip's relative seconds to absolute milliseconds for comparison
    let clipEndMs = firstEventTimestamp + currentClip.logicalEnd * 1000;
    console.log(`  Initial Clip End (Absolute): ${clipEndMs.toFixed(0)}ms`);

    let extendedInLastPass;
    let iteration = 0;
    do {
      iteration++;
      extendedInLastPass = false;

      const TOLERANCE_AFTER_END_MS = 200; // pull-in window for overlays starting just after end
      console.log(
        `  [Iteration ${iteration}] Checking overlays around ${clipEndMs.toFixed(
          0
        )}ms...`
      );

      // 1) Extend if any overlay already crosses the current end.
      const crossingAtEnd = overlayWindows.filter(
        (w) => w.startMs < clipEndMs && w.endMs > clipEndMs
      );
      if (crossingAtEnd.length > 0) {
        let targetEndMs = clipEndMs;
        for (const w of crossingAtEnd) {
          if (w.endMs > targetEndMs) targetEndMs = w.endMs;
        }
        const EPS_MS = 1;
        const newEndMs = targetEndMs + EPS_MS;
        console.log(
          `  [TRIM-OVERLAY] Crossing overlay detected (count=${
            crossingAtEnd.length
          }). Extending end from ${clipEndMs.toFixed(
            0
          )}ms to ${newEndMs.toFixed(0)}ms.`
        );
        clipEndMs = newEndMs;
        extendedInLastPass = true;
        continue; // re-check for chained overlays after extension
      }

      // 2) Pull in overlays that start just AFTER the end within tolerance.
      const nearAfter = overlayWindows.filter(
        (w) =>
          w.startMs >= clipEndMs &&
          w.startMs <= clipEndMs + TOLERANCE_AFTER_END_MS &&
          w.endMs > clipEndMs
      );
      if (nearAfter.length > 0) {
        console.log(
          `  -> Found ${nearAfter.length} overlay(s) starting shortly after end.`
        );
        let targetEndMs = clipEndMs;
        for (const w of nearAfter) {
          if (w.endMs > targetEndMs) targetEndMs = w.endMs;
        }
        const EPS_MS = 1;
        const newEndMs = targetEndMs + EPS_MS;
        if (newEndMs > clipEndMs) {
          console.log(
            `  \u0007 EXTENDING CLIP: End time will be updated from ${clipEndMs.toFixed(
              0
            )}ms to ${newEndMs.toFixed(0)}ms.`
          );
          clipEndMs = newEndMs;
          extendedInLastPass = true;
        }
      } else {
        console.log("  -> No overlays within near-after tolerance.");
      }
    } while (extendedInLastPass);

    // Apply updates to current and next clips_folder if we extended
    const newLogicalEnd = (clipEndMs - firstEventTimestamp) / 1000;
    if (newLogicalEnd > currentClip.logicalEnd) {
      console.log(
        `  FINAL UPDATE for Clip #${i}: logicalEnd ${currentClip.logicalEnd.toFixed(
          3
        )}s -> ${newLogicalEnd.toFixed(3)}s`
      );
      currentClip.logicalEnd = newLogicalEnd;
      currentClip.end = newLogicalEnd;

      console.log(
        `  SHIFTING NEXT Clip #${
          i + 1
        }: logicalStart ${nextClip.logicalStart.toFixed(
          3
        )}s -> ${newLogicalEnd.toFixed(3)}s`
      );
      nextClip.logicalStart = newLogicalEnd;
      nextClip.start = newLogicalEnd;
    } else {
      console.log(`  No changes made to Clip #${i}.`);
    }
  }

  console.log(
    "\n=================================================================="
  );
  console.log(
    "====== [DEBUG] Finished adjustClipBoundariesForOverlays_V3C ======"
  );
  console.log(
    "=================================================================="
  );
  return adjustedClips;
}
