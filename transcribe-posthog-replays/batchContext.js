import path from "path";

function ensureBucket(storage, bucketName) {
  if (!bucketName) {
    throw new Error(
      "[BATCH-CONTEXT] GCS_BUCKET_NAME is not configured; cannot persist batch context."
    );
  }
  return storage.bucket(bucketName);
}

function buildContextPath(sessionId, analysisId, suffix = "context") {
  const safeSession = String(sessionId || "unknown").replace(
    /[^a-zA-Z0-9_-]/g,
    "_"
  );
  const safeAnalysis = String(analysisId || "unknown").replace(
    /[^a-zA-Z0-9_-]/g,
    "_"
  );
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  return path
    .join(
      "batch-context",
      safeSession,
      `${safeAnalysis}-${suffix}-${timestamp}.json`
    )
    .replace(/\\/g, "/");
}

export async function saveBatchContext({
  storage,
  bucketName,
  sessionId,
  analysisId,
  payload,
}) {
  if (!payload || typeof payload !== "object") {
    throw new Error("[BATCH-CONTEXT] Missing payload for context save.");
  }
  const bucket = ensureBucket(storage, bucketName);
  const destination = buildContextPath(sessionId, analysisId);
  const file = bucket.file(destination);

  await file.save(JSON.stringify(payload), {
    metadata: { contentType: "application/json" },
    resumable: false,
  });

  return `gs://${bucketName}/${destination}`;
}

function parseGsUri(uri) {
  if (typeof uri !== "string" || !uri.startsWith("gs://")) {
    throw new Error(
      `[BATCH-CONTEXT] Invalid GCS URI "${uri}". Expected format: gs://bucket/path`
    );
  }
  const withoutScheme = uri.slice("gs://".length);
  const slashIndex = withoutScheme.indexOf("/");
  if (slashIndex === -1) {
    throw new Error(
      `[BATCH-CONTEXT] Invalid GCS URI "${uri}". Missing object path.`
    );
  }
  const bucketName = withoutScheme.slice(0, slashIndex);
  const objectPath = withoutScheme.slice(slashIndex + 1);
  return { bucketName, objectPath };
}

export async function loadBatchContext({ storage, contextUri, fallbackBucket }) {
  if (!contextUri) {
    throw new Error("[BATCH-CONTEXT] Missing context URI.");
  }

  const { bucketName, objectPath } = parseGsUri(contextUri);
  const bucket = storage.bucket(bucketName || fallbackBucket);
  const file = bucket.file(objectPath);
  const [contents] = await file.download();
  return JSON.parse(contents.toString("utf-8"));
}

export async function deleteBatchContext({ storage, contextUri }) {
  if (!contextUri) {
    return;
  }
 /*  const { bucketName, objectPath } = parseGsUri(contextUri); */
  try {
    /* await storage.bucket(bucketName).file(objectPath).delete({ ignoreNotFound: true }); */
  } catch (error) {
    console.warn(
      `[BATCH-CONTEXT] Failed to delete context file ${contextUri}:`,
      error?.message || error
    );
  }
}
