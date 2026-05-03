#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import Papa from "papaparse";
import { OpenRouter } from "@openrouter/sdk";

const LABELS = [
  "Defense / Military",
  "Economy / Budget / Taxes",
  "Healthcare",
  "Social Policy",
  "Foreign Policy",
  "Other"
];

const HEURISTIC_RULES = {
  "Defense / Military": [
    ["department of defense", 4],
    ["armed forces", 4],
    ["military", 3],
    ["defense authorization", 4],
    ["defense appropriations", 4],
    ["national defense", 4],
    ["army", 3],
    ["navy", 3],
    ["air force", 3],
    ["marine corps", 3],
    ["war powers", 4],
    ["weapon", 3],
    ["missile", 3],
    ["troops", 3],
    ["combat", 3],
    ["veterans", 2]
  ],
  "Economy / Budget / Taxes": [
    ["appropriation", 3],
    ["appropriations", 3],
    ["budget", 3],
    ["continuing resolution", 4],
    ["revenue", 3],
    ["tax", 3],
    ["tariff", 4],
    ["duty on", 4],
    ["treasury", 3],
    ["debt", 3],
    ["deficit", 3],
    ["bank", 3],
    ["banking", 3],
    ["currency", 3],
    ["monetary", 3],
    ["commerce", 2],
    ["economic", 2],
    ["finance", 2]
  ],
  Healthcare: [
    ["health care", 4],
    ["healthcare", 4],
    ["health insurance", 4],
    ["public health", 4],
    ["medicare", 4],
    ["medicaid", 4],
    ["hospital", 3],
    ["medical", 3],
    ["medicine", 3],
    ["drug", 3],
    ["pharmaceutical", 3],
    ["disease", 3],
    ["vaccine", 3],
    ["mental health", 4],
    ["physician", 3]
  ],
  "Social Policy": [
    ["education", 3],
    ["school", 2],
    ["student", 2],
    ["civil rights", 4],
    ["voting rights", 4],
    ["immigration", 4],
    ["immigrant", 3],
    ["abortion", 4],
    ["labor", 3],
    ["employment", 2],
    ["crime", 3],
    ["criminal", 3],
    ["family", 2],
    ["marriage", 3],
    ["housing", 2],
    ["welfare", 3],
    ["equal opportunity", 3]
  ],
  "Foreign Policy": [
    ["foreign affairs", 4],
    ["foreign policy", 4],
    ["international", 3],
    ["treaty", 4],
    ["sanctions", 4],
    ["ambassador", 3],
    ["embassy", 3],
    ["diplomatic", 3],
    ["united nations", 4],
    ["nato", 4],
    ["foreign aid", 4],
    ["recognition of", 3],
    ["trade agreement", 3],
    ["export control", 3]
  ],
  Other: [
    ["suspend the rules", 5],
    ["motion to recommit", 5],
    ["motion to table", 5],
    ["motion to adjourn", 5],
    ["previous question", 5],
    ["quorum", 5],
    ["journal", 4],
    ["rule providing", 4],
    ["providing for consideration", 4],
    ["house resolution", 3],
    ["senate resolution", 3],
    ["elect the speaker", 5],
    ["entitled to his seat", 5],
    ["point of order", 5],
    ["committee on rules", 4],
    ["yeas and nays", 4]
  ]
};

function loadDotenv(dotenvPath = ".env") {
  if (!fs.existsSync(dotenvPath)) return;
  const lines = fs.readFileSync(dotenvPath, "utf8").split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const [key, ...rest] = line.split("=");
    const value = rest.join("=").trim().replace(/^['"]|['"]$/g, "");
    if (!(key in process.env)) process.env[key.trim()] = value;
  }
}

function parseArgs(argv) {
  const options = {
    input: "data/all/HSall_rollcalls.csv",
    output: "data/all/HSall_rollcalls_categorized.csv",
    model: "openai/gpt-oss-120b:free",
    workers: 3,
    batchSize: 15,
    retries: 6,
    startRow: 1,
    minCongress: null,
    activeMemberWindow: false,
    currentMembersPath: "site-data/HS119/HS119_members.csv",
    allMembersPath: "data/all/HSall_members.csv",
    overwrite: false,
    limit: null
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === "--input") options.input = next, i += 1;
    else if (arg === "--output") options.output = next, i += 1;
    else if (arg === "--model") options.model = next, i += 1;
    else if (arg === "--workers") options.workers = Number(next), i += 1;
    else if (arg === "--batch-size") options.batchSize = Number(next), i += 1;
    else if (arg === "--retries") options.retries = Number(next), i += 1;
    else if (arg === "--start-row") options.startRow = Number(next), i += 1;
    else if (arg === "--min-congress") options.minCongress = Number(next), i += 1;
    else if (arg === "--limit") options.limit = Number(next), i += 1;
    else if (arg === "--current-members-path") options.currentMembersPath = next, i += 1;
    else if (arg === "--all-members-path") options.allMembersPath = next, i += 1;
    else if (arg === "--active-member-window") options.activeMemberWindow = true;
    else if (arg === "--overwrite") options.overwrite = true;
    else if (arg === "--help") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function printHelp() {
  console.log(`Usage:
node classify_rollcalls_openrouter.mjs \\
  --input data/all/HSall_rollcalls.csv \\
  --output data/all/HSall_rollcalls_categorized.csv \\
  --active-member-window \\
  --workers 3 \\
  --batch-size 15`);
}

function parseCsvFile(filePath) {
  const text = fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, "");
  const result = Papa.parse(text, { header: true, skipEmptyLines: "greedy" });
  if (result.errors?.length) {
    const fatal = result.errors.find((err) => err.code !== "TooFewFields" && err.code !== "TooManyFields");
    if (fatal) throw new Error(`${fatal.message} (${filePath})`);
  }
  return result.data;
}

function buildRollcallText(row) {
  const parts = [];
  for (const field of ["dtl_desc", "vote_question", "vote_desc", "bill_number"]) {
    const value = String(row[field] ?? "").trim();
    if (value) parts.push(`${field}: ${value}`);
  }
  return parts.length ? parts.join("\n") : "No description available.";
}

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function heuristicClassify(row) {
  const text = buildRollcallText(row).toLowerCase().replace(/\s+/g, " ");
  const scores = Object.fromEntries(LABELS.map((label) => [label, 0]));
  const matched = Object.fromEntries(LABELS.map((label) => [label, []]));

  for (const [label, rules] of Object.entries(HEURISTIC_RULES)) {
    for (const [phrase, weight] of rules) {
      const pattern = new RegExp(`\\b${escapeRegex(phrase).replace(/\\ /g, "\\s+")}\\b`);
      if (pattern.test(text)) {
        scores[label] += weight;
        matched[label].push(phrase);
      }
    }
  }

  const bestLabel = LABELS.reduce((best, current) => scores[current] > scores[best] ? current : best, LABELS[0]);
  const bestScore = scores[bestLabel];
  const orderedScores = Object.values(scores).sort((a, b) => b - a);
  const secondScore = orderedScores[1] ?? 0;

  if (bestScore < 4) return null;
  if (bestScore - secondScore < 2 && bestLabel !== "Other") return null;

  return {
    topic_category: bestLabel,
    topic_confidence: "",
    topic_reason: "",
    topic_model: "heuristic"
  };
}

function computeActiveMemberWindowStart(currentMembersPath, allMembersPath) {
  const currentRows = parseCsvFile(currentMembersPath);
  const activeIcpsr = new Set(
    currentRows
      .filter((row) => ["House", "Senate"].includes(row.chamber) && row.icpsr)
      .map((row) => String(row.icpsr))
  );
  let minCongress = null;
  for (const row of parseCsvFile(allMembersPath)) {
    if (!activeIcpsr.has(String(row.icpsr))) continue;
    if (!["House", "Senate"].includes(row.chamber)) continue;
    const congress = Number(row.congress);
    if (!Number.isFinite(congress)) continue;
    if (minCongress === null || congress < minCongress) minCongress = congress;
  }
  if (minCongress === null) throw new Error("Could not compute active-member window start.");
  return minCongress;
}

function rowKey(index, row) {
  return `${index}|${row.congress ?? ""}|${row.chamber ?? ""}|${row.rollnumber ?? ""}`;
}

function loadExistingKeys(outputPath) {
  if (!fs.existsSync(outputPath)) return new Set();
  const rows = parseCsvFile(outputPath);
  return new Set(rows.map((row) => rowKey(row._source_row, row)));
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function appendHeaderIfNeeded(outputPath, fieldnames, overwrite) {
  if (overwrite || !fs.existsSync(outputPath) || fs.statSync(outputPath).size === 0) {
    fs.writeFileSync(outputPath, Papa.unparse([], { columns: fieldnames, header: true }) + "\n");
  }
}

function appendRows(outputPath, rows, fieldnames) {
  if (!rows.length) return;
  const csv = Papa.unparse(rows, { columns: fieldnames, header: false });
  fs.appendFileSync(outputPath, `${csv}\n`);
}

function buildBatchPrompt(rows) {
  const blocks = rows.map(({ id, row }) => `ROW_ID: ${id}
congress: ${row.congress ?? ""}
chamber: ${row.chamber ?? ""}
rollnumber: ${row.rollnumber ?? ""}
date: ${row.date ?? ""}
text:
${buildRollcallText(row)}`);

  return `Classify each congressional roll call below into exactly one label.

Allowed labels:
1. Defense / Military
2. Economy / Budget / Taxes
3. Healthcare
4. Social Policy
5. Foreign Policy
6. Other

Return exactly one JSON array and nothing else.
Each item must be:
{"row_id":"...","category":"one of the six labels"}

Keep one output item for every input ROW_ID, in the same order.

Roll calls:

${blocks.join("\n\n---\n\n")}`;
}

function parseJsonPayload(text) {
  const candidate = text.trim();
  const startPositions = [candidate.indexOf("["), candidate.indexOf("{")].filter((value) => value >= 0);
  if (!startPositions.length) {
    throw new Error(`API response was not JSON. Response starts with: ${JSON.stringify(candidate.slice(0, 200))}`);
  }
  const start = Math.min(...startPositions);
  const parsed = JSON.parse(candidate.slice(start));
  return parsed;
}

function normalizeBatchResult(parsed, expectedIds, model) {
  if (!Array.isArray(parsed)) throw new Error("Batch response root was not an array.");
  if (parsed.length !== expectedIds.length) {
    throw new Error(`Batch response returned ${parsed.length} items, expected ${expectedIds.length}.`);
  }
  return parsed.map((item, index) => {
    if (!item || typeof item !== "object") throw new Error(`Batch item ${index} was not an object.`);
    if (item.row_id !== expectedIds[index]) {
      throw new Error(`Batch row_id mismatch at ${index}: expected ${expectedIds[index]}, got ${item.row_id}`);
    }
    if (!LABELS.includes(item.category)) {
      throw new Error(`Invalid category returned: ${item.category}`);
    }
    return {
      topic_category: item.category,
      topic_confidence: "",
      topic_reason: "",
      topic_model: model
    };
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeRateLimitSleep(err) {
  const reset = err?.error?.metadata?.headers?.["X-RateLimit-Reset"];
  if (!reset) return null;
  const resetMs = Number(reset);
  if (!Number.isFinite(resetMs)) return null;
  return Math.max(1000, resetMs - Date.now());
}

async function classifyRowsSdk(openrouter, rows, model, retries) {
  const expectedIds = rows.map(({ id }) => id);

  for (let attempt = 1; attempt <= retries; attempt += 1) {
    try {
      const stream = await openrouter.chat.send({
        httpReferer: "http://localhost",
        appTitle: "political-analysis-rollcall-classifier",
        chatRequest: {
          model,
          messages: [
            {
              role: "system",
              content: "You classify congressional roll calls. Return only compact JSON with row_id and category."
            },
            {
              role: "user",
              content: buildBatchPrompt(rows)
            }
          ],
          stream: true
        }
      });

      let content = "";
      for await (const chunk of stream) {
        const piece = chunk.choices?.[0]?.delta?.content;
        if (piece) content += piece;
      }

      const parsed = parseJsonPayload(content);
      return normalizeBatchResult(parsed, expectedIds, model);
    } catch (err) {
      const message = String(err?.message ?? err);
      const status = err?.status ?? err?.error?.code;
      if ((status === 429 || message.includes("429")) && attempt < retries) {
        const waitMs = Math.min((computeRateLimitSleep(err) ?? 20000) + Math.random() * 500, 600000);
        console.error(`Rate limited on batch; sleeping ${(waitMs / 1000).toFixed(1)}s before retry ${attempt + 1}/${retries}.`);
        await sleep(waitMs);
        continue;
      }
      if (attempt < retries) {
        const backoff = Math.min(6000, 800 * (2 ** (attempt - 1))) + Math.random() * 350;
        await sleep(backoff);
        continue;
      }
      throw err;
    }
  }

  throw new Error("Batch classification exhausted retries.");
}

async function classifyWithFallback(openrouter, batch, model, retries) {
  try {
    const rows = batch.map(([index, row]) => ({ id: String(index), row }));
    const results = await classifyRowsSdk(openrouter, rows, model, retries);
    return batch.map(([index, row], idx) => [index, row, results[idx]]);
  } catch (err) {
    if (batch.length === 1) throw err;
    const midpoint = Math.floor(batch.length / 2);
    console.error(`Non-JSON or invalid batch response for rows ${batch[0][0]}-${batch.at(-1)[0]}; splitting into ${midpoint} and ${batch.length - midpoint}.`);
    const left = await classifyWithFallback(openrouter, batch.slice(0, midpoint), model, retries);
    const right = await classifyWithFallback(openrouter, batch.slice(midpoint), model, retries);
    return [...left, ...right];
  }
}

function createBatches(rows, options, processedKeys, outputPath, fieldnames) {
  const batches = [];
  const directRows = [];
  let attempted = 0;
  let currentBatch = [];

  for (let i = 0; i < rows.length; i += 1) {
    const index = i + 1;
    const row = rows[i];
    if (index < options.startRow) continue;
    if (options.limit !== null && attempted >= options.limit) break;
    if (options.minCongress !== null && Number(row.congress) < options.minCongress) continue;

    const key = rowKey(index, row);
    if (processedKeys.has(key)) continue;
    attempted += 1;

    const heuristic = heuristicClassify(row);
    if (heuristic) {
      directRows.push({ ...row, ...heuristic, _source_row: String(index) });
      processedKeys.add(key);
      continue;
    }

    currentBatch.push([index, row]);
    if (currentBatch.length >= Math.max(1, options.batchSize)) {
      batches.push(currentBatch);
      currentBatch = [];
    }
  }

  if (currentBatch.length) batches.push(currentBatch);
  if (directRows.length) appendRows(outputPath, directRows, fieldnames);

  return { batches, directCount: directRows.length };
}

async function runWorkerPool(batches, workerCount, workerFn, onBatchDone) {
  let cursor = 0;
  let completedBatches = 0;
  let completedRows = 0;

  async function worker() {
    while (cursor < batches.length) {
      const batchIndex = cursor;
      cursor += 1;
      const result = await workerFn(batches[batchIndex]);
      completedBatches += 1;
      completedRows += result.length;
      if (onBatchDone) {
        await onBatchDone(result, {
          batchIndex,
          completedBatches,
          totalBatches: batches.length,
          completedRows
        });
      }
    }
  }

  const pool = Array.from({ length: Math.max(1, workerCount) }, () => worker());
  await Promise.all(pool);
}

async function main() {
  loadDotenv();
  const options = parseArgs(process.argv.slice(2));
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("Missing OPENROUTER_API_KEY in .env or environment.");

  if (options.activeMemberWindow) {
    options.minCongress = computeActiveMemberWindowStart(options.currentMembersPath, options.allMembersPath);
    console.error(`Active-member window starts at Congress ${options.minCongress}.`);
  }

  const inputRows = parseCsvFile(options.input);
  const processedKeys = options.overwrite ? new Set() : loadExistingKeys(options.output);
  const fieldnames = [
    ...Object.keys(inputRows[0] ?? {}),
    "topic_category",
    "topic_confidence",
    "topic_reason",
    "topic_model",
    "_source_row"
  ].filter((value, index, array) => array.indexOf(value) === index);

  ensureDir(options.output);
  appendHeaderIfNeeded(options.output, fieldnames, options.overwrite);

  const { batches, directCount } = createBatches(inputRows, options, processedKeys, options.output, fieldnames);
  console.error(`Prepared ${batches.length} API batches. Wrote ${directCount} heuristic rows immediately.`);

  const openrouter = new OpenRouter({ apiKey });
  await runWorkerPool(
    batches,
    options.workers,
    async (batch) => classifyWithFallback(openrouter, batch, options.model, options.retries)
    ,
    async (batchResults, progress) => {
      const outputRows = batchResults.map(([index, row, result]) => ({
        ...row,
        ...result,
        _source_row: String(index)
      }));
      appendRows(options.output, outputRows, fieldnames);
      console.error(
        `Wrote batch ${progress.completedBatches}/${progress.totalBatches} `
        + `(${progress.completedRows} SDK rows total).`
      );
    }
  );
  console.error(`Finished SDK classification for ${options.output}.`);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
