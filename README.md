# PostHog Session Replay Pipeline

An LLM pipeline that turns raw PostHog session replays into founder-ready insight reports with playable evidence clips.

The hard part wasn't calling Gemini. It was getting accuracy to the point where founders trust the output and keeping it affordable enough to run at scale.

> This was the production backend powering [UserSession.ai](https://www.usersession.ai). I've open-sourced it after winding the product down. The code is battle-damaged, not pretty. Most of what's here exists because something broke in a real customer session.

## UserSession.ai demo

https://github.com/user-attachments/assets/1faf2c1e-0e14-45db-9e2c-a5a0719eb3a4

## What it does

- **Ingests** PostHog session replays (including gzipped chunk handling and long-context sessions)
- **Renders** each session as video via headless Chrome + rrweb, injecting semantic color beacons to mark click boundaries and pause windows
- **Verifies** the exact on-screen location of every click by hit-testing inside the replayer iframe and cross-checking against rrweb's claimed target node
- **Cuts** the master recording into click and no-click clips based on the detected beacons, removes the beacon frames themselves, and re-encodes with silent audio
- **Transcribes** each clip via a two-pass Gemini flow (click-position extraction from zoomed frames → full clip description with rrweb event log as context)
- **Aggregates** transcripts across many sessions into structured insights, using explicit prompt caching and history-aware deduplication against the user's prior accepted/ignored/resolved reports
- **Stitches** evidence clips: for each insight's time range, finds overlapping clip files in GCS, trims with ffmpeg to exact boundaries, concatenates them, and returns a single playable URL per evidence item

## Why this was hard

Session replay transcription sounds like "feed video to Gemini." The accuracy tax is in the details. A few of the non-obvious problems this pipeline solves:

### Knowing where the user actually clicked

rrweb records a target node ID, but that doesn't tell you where on the rendered page the pointer was. The replayer scales and letterboxes the recorded viewport inside the output canvas, so naive `{x, y}` coordinates from the event stream drift by tens of pixels.

The pipeline resolves this by running `elementFromPoint` inside the replayer iframe at the moment of the click, translating coordinates through the host's scale transform, and cross-checking the resolved DOM node against rrweb's claimed target. When they disagree, it falls back to geometric containment via `getClientRects` with configurable tolerance, and surfaces the mismatch (plus a `highlightVisibility` object describing whether the claimed element is even in the viewport) to the downstream LLM prompt so it doesn't confidently describe the wrong element.

### Four timelines that don't line up

Any non-trivial session has at least four parallel timelines:

1. Raw rrweb timestamps
2. The skip-adjusted timeline after collapsing inactivity windows
3. The inflated visual timeline after injecting frozen click-hold sequences
4. Per-clip relative time

Every event — pointer events, page views, scrolls, inactivity overlays, synthetic boundary markers — has to be translated between these correctly or the final transcript describes the wrong moment. Functions like `calculateTimeShiftForTimestamp`, `cumulativePauseSecondsUpTo`, `getClickFreezeTimeSecondsUpTo`, and `mapToLogical` exist because every silent-failure mode in this translation eventually surfaces as "the AI described a click that happened three seconds before the one it was supposed to describe."

### Telling the LLM where a clip starts and ends

The rendered video carries embedded color-flash beacons — green before a click, red after, yellow at pause boundaries, blue elsewhere. They're detected after-the-fact by piping the master video through ffmpeg at low resolution (16×9, decimated FPS) and scanning the per-frame RGB averages from the raw stream.

Beacons serve three purposes:

- **Clip boundary detection.** Beacons pair up into click windows, so the same master video can be re-cut later without re-rendering.
- **Frame removal.** Once clips are cut, the beacon frames themselves are stripped so the final output looks clean.
- **Hidden subtitle metadata.** Yellow beacons are also written as `mov_text` subtitle cues with deterministic per-session IDs (`sha1(sessionId + seq)`), so boundary metadata round-trips through the video file.

### Preventing Gemini from inventing clicks

Early versions confidently hallucinated clicks that never happened. The fix is a constrained prompt per clip:

- The exact count of pointer events (`0`, `1`, or an explicit right-click count) is computed from rrweb and injected into the system instruction, and the model is told not to exceed it.
- The click's on-page location is pre-computed, verified, and passed in as a natural-language description so the model describes *that* element rather than whatever looks visually prominent.
- The rrweb event log (inputs, scrolls, navigations, inactivity overlays) within the clip window is passed in alongside the video with millisecond-accurate timestamps.
- When click verification fails, the prompt explicitly tells the model to describe the event outcome without naming a clicker, to avoid fabricating actor intent.

### Keeping costs sustainable

A transcribed session can easily contain 500k+ input tokens across many prompts. Without care this becomes unusable at any real volume.

- **Explicit prompt caching.** The large session-transcripts prompt is cached once per run via Gemini's `caches.create` API and reused across every insight task. Cached tokens bill at 10% of the standard input rate — tracked explicitly in the per-user cost report.
- **Dynamic FPS and resolution per clip.** Recording FPS is fixed at 20, but the FPS passed to Gemini's `videoMetadata` varies from 6 to 12 depending on clip duration. Every parameter — FPS, resolution, CRF, zoom crop size — was tuned against real cost-vs-accuracy trade-offs, not picked from a tutorial.
- **Per-user, per-session cost ledger** written to `gs://<bucket>/cost-logs-new/<userId>.log`, broken down by prompt label, model, token modality, cache usage, and long-context pricing tier. Lets operators answer "which user is expensive and why" in seconds.

### History-aware insight generation

A weekly report that repeats the same five insights every week is useless. The aggregation stage runs two explicit Gemini passes:

1. **Filter.** Candidate insights are compared against the user's *accepted*, *ignored*, and *recently resolved* reports pulled from Postgres. The model is instructed to exclude any candidate matching prior history.
2. **Merge.** Surviving candidates are merged with the user's current "new" reports to produce one deduplicated action plan.

Between sequential insight-extraction tasks within a single run, each task is also passed an ignore-list of titles already produced by earlier tasks, forcing the model to find new angles rather than rewording the same finding.

### Evidence clips, not dashboard links

Every insight references multiple sessions. A PostHog URL that opens a replay dashboard is not trustworthy evidence — the founder has to watch it themselves to check.

Instead, for each evidence time range, the pipeline:

1. Lists all clip files in GCS under the session's prefix.
2. Parses filenames (`from_Xs_to_Ys`) to find clips overlapping the range.
3. Picks a minimal covering set, preferring click-clips over no-click-clips and processed over raw.
4. Downloads each segment, trims with ffmpeg to the exact sub-range, concatenates (stream-copy with re-encode fallback), trims again to the exact total duration, re-encodes to `yuv420p` + `+faststart`, uploads, makes public, and appends the customer's `user_id`.

The founder clicks the evidence link and immediately sees the moment that supports the claim. No hunting.

### Resumability

Each expensive stage writes its output to a per-user JSON checkpoint (`merged-output-<userId>.json`, `dedupe-merge-output-<userId>.json`, `dedupe-merge-output-with-clips-<userId>.json`). Setting `USE_FILE`, `USE_FILE2`, or `USE_FILE3` resumes from the corresponding checkpoint. Debugging a late-stage bug doesn't mean re-burning the early-stage tokens.

## Architecture

```
┌─────────────────────────┐    ┌─────────────────────────┐
│  transcribe-posthog-    │    │  generate-insights/     │
│  replays/               │    │                         │
│                         │    │  daily.js               │
│  • Fetch + decompress   │    │  everyXdays.js          │
│    PostHog snapshots    │    │                         │
│  • Sanitize DOCTYPE     │    │  • Pull transcripts     │
│  • Prefetch + inline    │    │    from Postgres        │
│    images as data URIs  │    │  • Cache large prompt   │
│  • Inject clip markers  │    │  • Run insight tasks    │
│  • Render via headless  │    │    with ignore-list     │
│    Chrome + rrweb       │    │    chaining             │
│  • Beacon detection +   │    │  • Filter vs history    │
│    clip extraction      │    │    (accepted/ignored/   │
│  • Click verification   │    │    resolved)            │
│  • Two-pass Gemini      │    │  • Merge with current   │
│    transcription        │    │    'new' reports        │
│  • Write transcript +   │    │  • Stitch evidence      │
│    clip artifacts       │    │    clips via ffmpeg     │
│                         │    │  • Persist reports +    │
│                         │    │    per-session cost log │
└───────────┬─────────────┘    └──────────┬──────────────┘
            │                             │
            ▼                             ▼
┌──────────────────────────────────────────────────┐
│  PostgreSQL              GCS                     │
│  • sessionanalysis       • Clip files            │
│  • report                • Joined evidence clips │
│  • user2                 • Cost logs             │
└──────────────────────────────────────────────────┘
```

## Project structure

```
├── transcribe-posthog-replays/   Extraction + transcription engine
│   └── index.js                  Main orchestrator (~17k lines).
│                                 Owns ingestion, rendering, beacons,
│                                 click verification, Gemini prompts.
│
└── generate-insights/            Aggregation workers (cron-scheduled)
    ├── daily.js                  Short-window insight pass
    └── everyXdays.js             Longer-window insight pass with
                                  history filtering and evidence
                                  stitching
```

## Getting started

### Prerequisites

- Node.js 18+
- PostgreSQL with tables for `user2`, `sessionanalysis`, and `report`
- Google Cloud project with a GCS bucket (for clip storage, evidence outputs, and cost logs)
- Gemini API key
- ffmpeg available on the system (the transcription side uses `ffmpeg-static` and `ffprobe-static`; the insights side shells out to `ffmpeg` directly)

### Install

```
cd transcribe-posthog-replays && npm install
cd ../generate-insights && npm install
```

### Configure

Create a `.env` in each directory. Required variables include:

```
# Postgres
DB_USER=
DB_HOST=
DB_DATABASE=
DB_PASSWORD=
DB_PORT=

# GCP
GCS_BUCKET_NAME=
# google-credentials.json via GOOGLE_APPLICATION_CREDENTIALS

# Gemini
GEMINI_API_KEY_0=

# Encryption (for PostHog API keys stored in the DB)
ENCRYPTION_SECRET=
ENCRYPTION_SECRET_2=
ENCRYPTION_SECRET_3=
SALT=
```

`.env` files and `google-credentials.json` are gitignored. Don't commit them.

### Run

Transcribe a batch of sessions:

```
cd transcribe-posthog-replays
node index.js
```

Generate insights:

```
cd generate-insights
node daily.js
# or
node everyXdays.js
```

Useful flags:

- `USE_OUTPUTFILE=true` — skip PostHog fetch, use a local `outputFile.json` (useful for debugging the transcription pipeline)
- `USE_FILE=true` / `USE_FILE2=true` / `USE_FILE3=true` — resume the insights pipeline from a checkpoint
- `SKIP_CLIPS=true` — skip evidence clip generation (faster iteration on prompts)
- `DAYS_BACK=<n>` — override the session-window length

## Current state

The code is production code from a live product, not a reference implementation. Specifically:

- The main orchestrator in `transcribe-posthog-replays/index.js` is large (~17k lines in one file) because splitting it didn't earn its keep against the cost of making it easier to read later.
- There are dead branches, commented-out experiments, and at least one "V3C" suffix that testifies to how many times the clip-boundary logic was rewritten.
- Logging is verbose on purpose — every silent failure mode in this pipeline eventually needed tracing.

If you're reading this as a reference, the interesting parts are:
- `performClickVerification` and `freezePlayerAtClick` in the replayer bootstrap — click-position truth
- `findBeaconFrames_TrueStream` — beacon detection via decimated raw RGB decoding
- `cutVideoBasedOnBeacons` + `postProcessClip` — clip extraction with beacon-frame removal
- `adjustClipBoundariesForOverlays_V3C` — the overlay-aware boundary logic
- `ensurePromptCache` + the filter/merge flow in `everyXdays.js` — the cost-aware aggregation pattern
- `findClipPublicUrlsForRange` + `joinClipsAndUpload` — the ffmpeg-based evidence stitcher

## License

MIT. See `LICENSE`.

## Attribution

Built solo as the backend for [UserSession.ai](https://www.usersession.ai). Open-sourced after the product was wound down.
