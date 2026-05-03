    #!/usr/bin/env python3
"""
Classify congressional roll calls with OpenRouter and write a new CSV.

The script reads OPENROUTER_API_KEY from .env or the environment, calls the
OpenRouter chat completions API in batches, and appends topic columns:

- topic_category
- topic_confidence
- topic_reason
- topic_model

Example:
  python3 classify_rollcalls_ai.py \
    --input data/all/HSall_rollcalls.csv \
    --output data/all/HSall_rollcalls_categorized.csv \
    --model tencent/hy3-preview:free \
    --limit 100 \
    --active-member-window
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LABELS = [
    "Defense / Military",
    "Economy / Budget / Taxes",
    "Healthcare",
    "Social Policy",
    "Foreign Policy",
    "Other",
]

SYSTEM_PROMPT = (
    "You classify congressional roll call descriptions into exactly one topic "
    "label. Return only compact JSON with row_id and category."
)

try:
    import certifi
except ImportError:
    certifi = None

_SSL_CONTEXT: Optional[ssl.SSLContext] = None
HEURISTIC_RULES = {
    "Defense / Military": [
        ("department of defense", 4),
        ("armed forces", 4),
        ("military", 3),
        ("defense authorization", 4),
        ("defense appropriations", 4),
        ("national defense", 4),
        ("army", 3),
        ("navy", 3),
        ("air force", 3),
        ("marine corps", 3),
        ("war powers", 4),
        ("weapon", 3),
        ("missile", 3),
        ("troops", 3),
        ("combat", 3),
        ("veterans", 2),
    ],
    "Economy / Budget / Taxes": [
        ("appropriation", 3),
        ("appropriations", 3),
        ("budget", 3),
        ("continuing resolution", 4),
        ("revenue", 3),
        ("tax", 3),
        ("tariff", 4),
        ("duty on", 4),
        ("treasury", 3),
        ("debt", 3),
        ("deficit", 3),
        ("bank", 3),
        ("banking", 3),
        ("currency", 3),
        ("monetary", 3),
        ("commerce", 2),
        ("economic", 2),
        ("finance", 2),
    ],
    "Healthcare": [
        ("health care", 4),
        ("healthcare", 4),
        ("health insurance", 4),
        ("public health", 4),
        ("medicare", 4),
        ("medicaid", 4),
        ("hospital", 3),
        ("medical", 3),
        ("medicine", 3),
        ("drug", 3),
        ("pharmaceutical", 3),
        ("disease", 3),
        ("vaccine", 3),
        ("mental health", 4),
        ("physician", 3),
    ],
    "Social Policy": [
        ("education", 3),
        ("school", 2),
        ("student", 2),
        ("civil rights", 4),
        ("voting rights", 4),
        ("immigration", 4),
        ("immigrant", 3),
        ("abortion", 4),
        ("labor", 3),
        ("employment", 2),
        ("crime", 3),
        ("criminal", 3),
        ("family", 2),
        ("marriage", 3),
        ("housing", 2),
        ("welfare", 3),
        ("equal opportunity", 3),
    ],
    "Foreign Policy": [
        ("foreign affairs", 4),
        ("foreign policy", 4),
        ("international", 3),
        ("treaty", 4),
        ("sanctions", 4),
        ("ambassador", 3),
        ("embassy", 3),
        ("diplomatic", 3),
        ("united nations", 4),
        ("nato", 4),
        ("foreign aid", 4),
        ("recognition of", 3),
        ("trade agreement", 3),
        ("export control", 3),
    ],
    "Other": [
        ("suspend the rules", 5),
        ("motion to recommit", 5),
        ("motion to table", 5),
        ("motion to adjourn", 5),
        ("previous question", 5),
        ("quorum", 5),
        ("journal", 4),
        ("rule providing", 4),
        ("providing for consideration", 4),
        ("house resolution", 3),
        ("senate resolution", 3),
        ("elect the speaker", 5),
        ("entitled to his seat", 5),
        ("point of order", 5),
        ("committee on rules", 4),
        ("yeas and nays", 4),
    ],
}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_rollcall_text(row: Dict[str, str]) -> str:
    parts = []
    for field in ("dtl_desc", "vote_question", "vote_desc", "bill_number"):
        value = (row.get(field) or "").strip()
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts) if parts else "No description available."


def heuristic_classify(row: Dict[str, str]) -> Optional[Dict[str, str]]:
    text = build_rollcall_text(row).lower()
    text = re.sub(r"\s+", " ", text)
    scores = {label: 0 for label in LABELS}
    matched = {label: [] for label in LABELS}

    for label, rules in HEURISTIC_RULES.items():
        for phrase, weight in rules:
            pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, text):
                scores[label] += weight
                matched[label].append(phrase)

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score <= 0:
        return None

    ordered_scores = sorted(scores.values(), reverse=True)
    second_score = ordered_scores[1] if len(ordered_scores) > 1 else 0

    if best_score < 4:
        return None
    if best_score - second_score < 2 and best_label != "Other":
        return None

    confidence = 0.92 if best_score >= 6 else 0.84
    reason = f"heuristic keyword match: {', '.join(matched[best_label][:3])}"
    return {
        "topic_category": best_label,
        "topic_confidence": f"{confidence:.3f}",
        "topic_reason": reason,
        "topic_model": "heuristic",
    }


def build_user_prompt(row: Dict[str, str]) -> str:
    return f"""Classify this congressional roll call into exactly one label.

Allowed labels:
1. Defense / Military
2. Economy / Budget / Taxes
3. Healthcare
4. Social Policy
5. Foreign Policy
6. Other

Rules:
- Defense / Military: war, armed forces, military operations, defense spending, weapons, military procurement, veterans if clearly military.
- Economy / Budget / Taxes: tariffs, duties, taxes, appropriations, budgets, debt, banking, currency, commerce, revenue, economic regulation.
- Healthcare: health, medicine, hospitals, disease, insurance, public health, Medicare, Medicaid.
- Social Policy: education, civil rights, immigration, abortion, labor, crime, family policy, voting rights.
- Foreign Policy: treaties, diplomacy, sanctions, recognition, international agreements, foreign trade relations, but not war or direct military operations.
- Other: chamber procedure, seating/elections, internal administration, parliamentary rules, or genuinely unclear subject matter.

Return exactly this JSON shape:
{{"category":"one of the six labels"}}

Roll call metadata:
- congress: {row.get("congress", "")}
- chamber: {row.get("chamber", "")}
- rollnumber: {row.get("rollnumber", "")}
- date: {row.get("date", "")}

Roll call text:
{build_rollcall_text(row)}
"""


def build_batch_user_prompt(rows: List[Tuple[str, Dict[str, str]]]) -> str:
    blocks = []
    for row_id, row in rows:
        blocks.append(
            f"""ROW_ID: {row_id}
congress: {row.get("congress", "")}
chamber: {row.get("chamber", "")}
rollnumber: {row.get("rollnumber", "")}
date: {row.get("date", "")}
text:
{build_rollcall_text(row)}"""
        )

    joined = "\n\n---\n\n".join(blocks)
    return f"""Classify each congressional roll call below into exactly one label.

Allowed labels:
1. Defense / Military
2. Economy / Budget / Taxes
3. Healthcare
4. Social Policy
5. Foreign Policy
6. Other

Rules:
- Defense / Military: war, armed forces, military operations, defense spending, weapons, military procurement, veterans if clearly military.
- Economy / Budget / Taxes: tariffs, duties, taxes, appropriations, budgets, debt, banking, currency, commerce, revenue, economic regulation.
- Healthcare: health, medicine, hospitals, disease, insurance, public health, Medicare, Medicaid.
- Social Policy: education, civil rights, immigration, abortion, labor, crime, family policy, voting rights.
- Foreign Policy: treaties, diplomacy, sanctions, recognition, international agreements, foreign trade relations, but not war or direct military operations.
- Other: chamber procedure, seating/elections, internal administration, parliamentary rules, or genuinely unclear subject matter.

Return exactly one JSON array and nothing else.
Each array item must have this shape:
{{"row_id":"...","category":"one of the six labels"}}

Keep one output item for every input ROW_ID, in the same order.

Roll calls:

{joined}
"""


def extract_json_object(text: str) -> Dict[str, object]:
    candidate = text.strip()
    if candidate.startswith("```"):
      candidate = candidate.strip("`")
      if "\n" in candidate:
          candidate = candidate.split("\n", 1)[1]
      if candidate.endswith("```"):
          candidate = candidate[:-3]
      candidate = candidate.strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return JSON: {text[:200]!r}")
    return json.loads(candidate[start:end + 1])


def normalize_result(parsed: Dict[str, object], model: str) -> Dict[str, str]:
    category = parsed.get("category")

    if category not in LABELS:
        raise ValueError(f"Invalid category returned: {category!r}")

    return {
        "topic_category": str(category),
        "topic_confidence": "",
        "topic_reason": "",
        "topic_model": model,
    }


def normalize_batch_results(
    parsed: object,
    expected_row_ids: List[str],
    model: str,
) -> List[Dict[str, str]]:
    if not isinstance(parsed, list):
        raise ValueError(f"Batch classification did not return a JSON array: {type(parsed).__name__}")

    if len(parsed) != len(expected_row_ids):
        raise ValueError(
            f"Batch classification returned {len(parsed)} items, expected {len(expected_row_ids)}."
        )

    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Batch classification item {idx} was not an object.")
        row_id = item.get("row_id")
        if row_id != expected_row_ids[idx]:
            raise ValueError(
                f"Batch classification row_id mismatch at position {idx}: "
                f"expected {expected_row_ids[idx]!r}, got {row_id!r}"
            )
        normalized.append(normalize_result(item, model))
    return normalized


def build_ssl_context() -> ssl.SSLContext:
    global _SSL_CONTEXT
    if _SSL_CONTEXT is not None:
        return _SSL_CONTEXT
    if certifi is None:
        raise RuntimeError(
            "Missing certifi. Install it with `python3 -m pip install certifi` "
            "or run the macOS Python certificate installer."
        )
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    return _SSL_CONTEXT


def parse_api_json_text(raw_text: str) -> Dict[str, object]:
    candidate = raw_text.lstrip("\ufeff \t\r\n")
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start_positions = [pos for pos in (candidate.find("{"), candidate.find("[")) if pos != -1]
        if not start_positions:
            snippet = candidate[:300].strip()
            raise ValueError(f"API response was not JSON. Response starts with: {snippet!r}")
        start = min(start_positions)
        parsed, _ = decoder.raw_decode(candidate[start:])
    return parsed


def compute_rate_limit_sleep_seconds(error_body: str) -> Optional[float]:
    try:
        parsed = json.loads(error_body)
    except json.JSONDecodeError:
        return None

    reset_raw = (
        parsed.get("error", {})
        .get("metadata", {})
        .get("headers", {})
        .get("X-RateLimit-Reset")
    )
    if reset_raw is None:
        return None

    try:
        reset_ms = int(str(reset_raw))
    except ValueError:
        return None

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return max(1.0, (reset_ms - now_ms) / 1000.0)


def classify_rows(
    rows: List[Tuple[str, Dict[str, str]]],
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
) -> Dict[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_user_prompt(rows)},
        ],
        "temperature": 0,
    }

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "political-analysis-rollcall-classifier",
        },
        method="POST",
    )

    ssl_context = build_ssl_context()

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
            body = parse_api_json_text(raw_text)

            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"OpenRouter response missing choices: {body}")

            message = choices[0].get("message", {})
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"OpenRouter response missing message content: {body}")
            expected_row_ids = [row_id for row_id, _ in rows]
            parsed = parse_api_json_text(content)
            return normalize_batch_results(parsed, expected_row_ids, model)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            last_error = err
            if err.code == 429 and attempt < retries:
                wait_seconds = compute_rate_limit_sleep_seconds(detail)
                if wait_seconds is None:
                    wait_seconds = min(20.0, 1.5 * (2 ** (attempt - 1)))
                wait_seconds = min(wait_seconds + random.uniform(0.0, 0.5), 600.0)
                print(
                    f"Rate limited on batch; sleeping {wait_seconds:.1f}s before retry {attempt + 1}/{retries}.",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(
                f"OpenRouter HTTP error: {err.code} {err.reason}\n{detail}"
            ) from err
        except (
            json.JSONDecodeError,
            ValueError,
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.HTTPException,
        ) as err:
            last_error = err
            if attempt >= retries:
                raise
            sleep_seconds = min(6.0, 0.8 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.35)
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


def row_key(index: int, row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        str(index),
        str(row.get("congress", "")),
        str(row.get("chamber", "")),
        str(row.get("rollnumber", "")),
    )


def classify_row_with_context(
    batch: List[Tuple[int, Dict[str, str]]],
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
) -> List[Tuple[int, Dict[str, str], Dict[str, str]]]:
    labeled_rows = [(str(index), row) for index, row in batch]
    try:
        results = classify_rows(labeled_rows, api_key=api_key, model=model, timeout=timeout, retries=retries)
        return [(index, row, result) for (index, row), result in zip(batch, results)]
    except ValueError as err:
        if len(batch) == 1:
            raise
        midpoint = len(batch) // 2
        left = batch[:midpoint]
        right = batch[midpoint:]
        print(
            f"Non-JSON or invalid batch response for rows {batch[0][0]}-{batch[-1][0]}; "
            f"splitting into {len(left)} and {len(right)}.",
            file=sys.stderr,
        )
        left_results = classify_row_with_context(left, api_key=api_key, model=model, timeout=timeout, retries=retries)
        right_results = classify_row_with_context(right, api_key=api_key, model=model, timeout=timeout, retries=retries)
        return left_results + right_results


def compute_active_member_window_start(
    current_members_path: Path,
    all_members_path: Path,
) -> int:
    with current_members_path.open(newline="", encoding="utf-8") as infile:
        current_rows = list(csv.DictReader(infile))

    active_icpsr = {
        row["icpsr"]
        for row in current_rows
        if row.get("chamber") in {"House", "Senate"} and row.get("icpsr")
    }
    if not active_icpsr:
        raise RuntimeError(f"No active House/Senate members found in {current_members_path}")

    min_congress: Optional[int] = None
    with all_members_path.open(newline="", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            if row.get("icpsr") not in active_icpsr:
                continue
            if row.get("chamber") not in {"House", "Senate"}:
                continue
            congress_text = str(row.get("congress", "")).strip()
            if not congress_text:
                continue
            congress = int(congress_text)
            if min_congress is None or congress < min_congress:
                min_congress = congress

    if min_congress is None:
        raise RuntimeError(
            "Could not compute the first Congress served by the current roster. "
            f"Checked {current_members_path} against {all_members_path}."
        )

    return min_congress


def load_existing_keys(output_path: Path) -> set[Tuple[str, str, str, str]]:
    if not output_path.exists():
        return set()

    keys = set()
    with output_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            index = row.get("_source_row", "")
            keys.add((index, row.get("congress", ""), row.get("chamber", ""), row.get("rollnumber", "")))
    return keys


def classify_csv(
    input_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    limit: Optional[int],
    start_row: int,
    min_congress: Optional[int],
    timeout: int,
    retries: int,
    delay_seconds: float,
    overwrite: bool,
    workers: int,
    batch_size: int,
) -> None:
    processed_keys = set() if overwrite else load_existing_keys(output_path)
    mode = "w" if overwrite else "a"

    with input_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        for extra in ("topic_category", "topic_confidence", "topic_reason", "topic_model", "_source_row"):
            if extra not in fieldnames:
                fieldnames.append(extra)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode, newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            if overwrite or output_path.stat().st_size == 0:
                writer.writeheader()

            written = 0
            attempted = 0
            reader_iter = enumerate(reader, start=1)

            def next_batch() -> Optional[List[Tuple[int, Dict[str, str]]]]:
                nonlocal attempted, written
                batch: List[Tuple[int, Dict[str, str]]] = []
                for index, row in reader_iter:
                    if index < start_row:
                        continue
                    if limit is not None and attempted >= limit:
                        break
                    if min_congress is not None:
                        congress_text = str(row.get("congress", "")).strip()
                        if not congress_text or int(congress_text) < min_congress:
                            continue
                    key = row_key(index, row)
                    if key in processed_keys:
                        continue
                    attempted += 1
                    heuristic = heuristic_classify(row)
                    if heuristic is not None:
                        row.update(heuristic)
                        row["_source_row"] = str(index)
                        writer.writerow(row)
                        written += 1
                        processed_keys.add(row_key(index, row))
                        print(
                            f"row {index}: {row.get('chamber', '')} {row.get('congress', '')}-"
                            f"{row.get('rollnumber', '')} -> {heuristic['topic_category']} [heuristic]",
                            file=sys.stderr,
                        )
                        continue
                    batch.append((index, row))
                    if len(batch) >= max(1, batch_size):
                        break
                if batch:
                    outfile.flush()
                return batch or None

            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                in_flight = {}

                while len(in_flight) < max(1, workers):
                    candidate_batch = next_batch()
                    if candidate_batch is None:
                        break
                    future = executor.submit(
                        classify_row_with_context,
                        candidate_batch,
                        api_key,
                        model,
                        timeout,
                        retries,
                    )
                    in_flight[future] = candidate_batch

                while in_flight:
                    done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)

                    for future in done:
                        candidate_batch = in_flight.pop(future)
                        batch_start = candidate_batch[0][0]
                        try:
                            batch_results = future.result()
                        except urllib.error.HTTPError as err:
                            detail = err.read().decode("utf-8", errors="replace")
                            raise RuntimeError(
                                f"OpenRouter HTTP error on batch starting row {batch_start}: {err.code} {err.reason}\n{detail}"
                            ) from err
                        except urllib.error.URLError as err:
                            raise RuntimeError(f"OpenRouter network error on batch starting row {batch_start}: {err}") from err

                        for index, row, result in batch_results:
                            row.update(result)
                            row["_source_row"] = str(index)
                            writer.writerow(row)
                            written += 1
                            processed_keys.add(row_key(index, row))

                            print(
                                f"row {index}: {row.get('chamber', '')} {row.get('congress', '')}-"
                                f"{row.get('rollnumber', '')} -> {result['topic_category']}",
                                file=sys.stderr,
                            )
                        outfile.flush()

                        if delay_seconds > 0:
                            time.sleep(delay_seconds)

                        candidate_batch = next_batch()
                        if candidate_batch is None:
                            continue
                        next_future = executor.submit(
                            classify_row_with_context,
                            candidate_batch,
                            api_key,
                            model,
                            timeout,
                            retries,
                        )
                        in_flight[next_future] = candidate_batch

    print(f"Wrote {written} classified rows to {output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to HSall_rollcalls.csv")
    parser.add_argument("--output", required=True, type=Path, help="Path to the categorized CSV")
    parser.add_argument(
        "--model",
        default="tencent/hy3-preview:free",
        help="OpenRouter model slug. Defaults to the free model you referenced.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only classify this many new rows")
    parser.add_argument("--start-row", type=int, default=1, help="1-based row number to start from")
    parser.add_argument(
        "--min-congress",
        type=int,
        default=None,
        help="Only classify rollcalls from this Congress onward.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per row for malformed responses or transient network failures.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional pause between requests to reduce rate-limit pressure",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the output file from scratch instead of resuming/appending",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel requests to run at once. Increase carefully to avoid rate limits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of roll calls to classify in one API request.",
    )
    parser.add_argument(
        "--active-member-window",
        action="store_true",
        help=(
            "Automatically start at the earliest Congress served by any member "
            "of the current 119th Congress roster."
        ),
    )
    parser.add_argument(
        "--current-members-path",
        type=Path,
        default=Path("site-data/HS119/HS119_members.csv"),
        help="Current Congress roster file used with --active-member-window.",
    )
    parser.add_argument(
        "--all-members-path",
        type=Path,
        default=Path("data/all/HSall_members.csv"),
        help="Historical members file used with --active-member-window.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY. Put it in .env or export it in your shell.")

    min_congress = args.min_congress
    if args.active_member_window:
        min_congress = compute_active_member_window_start(
            current_members_path=args.current_members_path,
            all_members_path=args.all_members_path,
        )
        print(f"Active-member window starts at Congress {min_congress}.", file=sys.stderr)

    classify_csv(
        input_path=args.input,
        output_path=args.output,
        api_key=api_key,
        model=args.model,
        limit=args.limit,
        start_row=args.start_row,
        min_congress=min_congress,
        timeout=args.timeout,
        retries=args.retries,
        delay_seconds=args.delay_seconds,
        overwrite=args.overwrite,
        workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
