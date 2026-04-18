# PostHog Session Replay Pipeline

An automated, AI-powered pipeline to transcribe, analyze, and extract deep insights from PostHog session replays.

This repository orchestrates a multi-step process utilizing Puppeteer for replay rendering, Google Gemini for visual and text analysis, and Google Cloud Storage for clip persistence, culminating in structured, queryable analytics in PostgreSQL.

## 📂 Project Structure

This project is divided into two primary sub-modules, each serving a distinct phase of the pipeline:

### 1. `transcribe-posthog-replays/`
The extraction and inference engine. It orchestrates headless browsers to replay PostHog user sessions. It coordinates timeline events, records video clips with `puppeteer-screen-recorder`, processes clips using `ffmpeg-static`, and sends contextual slices to Google Gemini to interpret user actions and transcribe exact behaviors into structured summaries in a PostgreSQL database.

### 2. `generate-insights/`
The scheduled analytics workers. Contains scripts intended to be run via cronjobs or schedulers (`daily.js`, `everyXdays.js`). These workers query the accumulated session transcripts and utilize AI to generate actionable business metrics, UI/UX assessments, and behavioral summaries over specified time horizons.

## 🚀 Getting Started

### Prerequisites
* **Node.js**: Recommended v18 or newer
* **PostgreSQL**: A running instance for storing transcripts and insights
* **Google Cloud Project**: With a GCS bucket available for storing video clips and Gemini intermediate batch processing
* **Gemini API Key**: For AI transcription and analysis logic

### Installation

Navigate into each directory and install the dependencies:

```bash
cd transcribe-posthog-replays
npm install

cd ../generate-insights
npm install
```

### Configuration

Both modules require environment variables to operate. You will need to create a `.env` file in **both** directories (`transcribe-posthog-replays/.env` and `generate-insights/.env`). 

Ensure the following variables are configured appropriately for your environment:

```env
# Database Configuration
DB_USER=
DB_HOST=
DB_DATABASE=
DB_PASSWORD=
DB_PORT=

# Google Cloud & Storage
GCS_BUCKET_NAME=

# Add any additional environment keys specific to PostHog APIs and Google Gemini auth here
```

> **Note:** `.env` files and `google-credentials.json` are strictly ignored in git via `.gitignore` to prevent credential leaks. Do not commit these files.

## 🔧 Usage

**Running the Transcription Pipeline**
```bash
cd transcribe-posthog-replays
node index.js
```

**Running the Insight Generators**
```bash
cd generate-insights
node daily.js
# OR
node everyXdays.js
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).
