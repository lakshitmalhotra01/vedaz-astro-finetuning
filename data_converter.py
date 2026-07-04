#!/usr/bin/env python3
"""
data_converter.py
=================
Converts astrologer chat data from multiple formats into ChatML-style
messages format required by Qwen2.5-7B-Instruct.

Supported input formats:
  - JSON / JSONL  (messages array, or flat prompt/response)
  - CSV           (columns: user/assistant, prompt/response, human/bot)
  - TXT           (Human:/Assistant: alternating blocks)
  - Excel (.xlsx) (same column patterns as CSV)

Output: JSONL file where each line is:
  {"messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
  ]}

Usage:
  python data_converter.py --input_dir ./raw_data --output ./dataset_chatml.jsonl
  python data_converter.py --input_file chat_data.json --output ./dataset_chatml.jsonl
  python data_converter.py --inspect --input_dir ./raw_data   # just report format
"""

import os
import json
import csv
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default system prompt for the astrologer persona
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are Vedaz, an expert Vedic astrologer with deep knowledge of Jyotish shastra. "
    "You have mastered kundli (birth chart) analysis, rashi (zodiac signs), nakshatra (lunar mansions), "
    "planetary positions, dasha systems (Vimshottari, Yogini), bhava (houses), yogas, and divisional charts. "
    "You speak with warmth, empathy, and authority. When a user shares their birth details, you carefully "
    "analyze their chart before giving predictions. You always mention specific dates, planetary transits, "
    "or dasha periods when predicting future events. You balance spiritual wisdom with practical guidance."
)


# ---------------------------------------------------------------------------
# Format Detectors
# ---------------------------------------------------------------------------

def detect_json_structure(data: Any) -> str:
    """Detect the internal structure of JSON/JSONL data."""
    if isinstance(data, list) and len(data) > 0:
        sample = data[0]
        if isinstance(sample, dict):
            keys = set(sample.keys())
            if "messages" in keys:
                return "messages_array"        # [{messages: [...]}]
            if "conversations" in keys:
                return "conversations_array"   # [{conversations: [...]}]
            if "prompt" in keys and "response" in keys:
                return "prompt_response"       # [{prompt: ..., response: ...}]
            if "user" in keys and "assistant" in keys:
                return "user_assistant"        # [{user: ..., assistant: ...}]
            if "human" in keys and "gpt" in keys:
                return "human_gpt"             # ShareGPT format
            if "instruction" in keys and "output" in keys:
                return "alpaca"                # Alpaca format
    elif isinstance(data, dict):
        if "messages" in data:
            return "single_messages"
    return "unknown"


# ---------------------------------------------------------------------------
# Parsers for each format
# ---------------------------------------------------------------------------

def parse_messages_array(sample: Dict) -> Optional[List[Dict]]:
    """Parse [{messages: [{"role": ..., "content": ...}]}]"""
    msgs = sample.get("messages") or sample.get("conversations") or []
    if not msgs:
        return None
    normalized = []
    for m in msgs:
        role = m.get("role") or m.get("from", "")
        content = m.get("content") or m.get("value", "")
        if role.lower() in ("human", "user"):
            role = "user"
        elif role.lower() in ("gpt", "assistant", "bot", "model"):
            role = "assistant"
        elif role.lower() == "system":
            role = "system"
        else:
            continue
        normalized.append({"role": role, "content": content.strip()})
    return normalized if len(normalized) >= 2 else None


def parse_prompt_response(sample: Dict) -> Optional[List[Dict]]:
    """Parse flat {prompt: ..., response: ...} or {user: ..., assistant: ...}"""
    user_content = (
        sample.get("prompt") or sample.get("user") or
        sample.get("human") or sample.get("input") or ""
    )
    asst_content = (
        sample.get("response") or sample.get("assistant") or
        sample.get("gpt") or sample.get("output") or ""
    )
    if not user_content or not asst_content:
        return None
    system_content = sample.get("system") or sample.get("system_prompt") or ""
    msgs = []
    if system_content:
        msgs.append({"role": "system", "content": system_content.strip()})
    msgs.append({"role": "user", "content": user_content.strip()})
    msgs.append({"role": "assistant", "content": asst_content.strip()})
    return msgs


def parse_alpaca(sample: Dict) -> Optional[List[Dict]]:
    """Parse Alpaca-style {instruction, input, output}"""
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")
    if not instruction or not output:
        return None
    user_msg = instruction.strip()
    if inp:
        user_msg += f"\n\n{inp.strip()}"
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output.strip()},
    ]


def parse_txt_conversation(text: str) -> Optional[List[Dict]]:
    """
    Parse text files with alternating Human:/Assistant: blocks.
    Also handles User:/Bot: and similar patterns.
    """
    # Regex to split on role markers
    pattern = re.compile(
        r"(Human|User|H|Assistant|Bot|A)\s*:\s*",
        re.IGNORECASE,
    )
    parts = pattern.split(text.strip())
    # parts = [pre_text, role1, content1, role2, content2, ...]
    msgs = []
    i = 1
    while i + 1 < len(parts):
        role_raw = parts[i].strip().lower()
        content = parts[i + 1].strip()
        if role_raw in ("human", "user", "h"):
            role = "user"
        elif role_raw in ("assistant", "bot", "a"):
            role = "assistant"
        else:
            i += 2
            continue
        if content:
            msgs.append({"role": role, "content": content})
        i += 2
    return msgs if len(msgs) >= 2 else None


# ---------------------------------------------------------------------------
# Main file readers
# ---------------------------------------------------------------------------

def read_json_file(filepath: Path) -> List[List[Dict]]:
    """Read JSON or JSONL, return list of message sequences."""
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Try JSONL first
            lines = f.readlines()
        
        samples = []
        if len(lines) > 1:
            # Attempt JSONL
            try:
                for line in lines:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            except json.JSONDecodeError:
                # Fallback: load as single JSON
                samples = json.loads("".join(lines))
        else:
            samples = json.loads("".join(lines))

        if isinstance(samples, dict):
            samples = [samples]

        fmt = detect_json_structure(samples)
        logger.info(f"  [{filepath.name}] Detected JSON structure: {fmt} ({len(samples)} samples)")

        for sample in samples:
            msgs = None
            if fmt in ("messages_array", "conversations_array", "single_messages"):
                msgs = parse_messages_array(sample)
            elif fmt in ("prompt_response", "user_assistant", "human_gpt"):
                msgs = parse_prompt_response(sample)
            elif fmt == "alpaca":
                msgs = parse_alpaca(sample)
            else:
                # Try all parsers
                msgs = (
                    parse_messages_array(sample) or
                    parse_prompt_response(sample) or
                    parse_alpaca(sample)
                )
            if msgs:
                results.append(msgs)

    except Exception as e:
        logger.error(f"  Error reading {filepath}: {e}")
    return results


def read_csv_file(filepath: Path) -> List[List[Dict]]:
    """Read CSV/Excel, auto-detect column names."""
    results = []
    try:
        if filepath.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)

        df.columns = [c.strip().lower() for c in df.columns]
        logger.info(f"  [{filepath.name}] CSV columns: {list(df.columns)}")

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            msgs = parse_prompt_response(row_dict)
            if msgs:
                results.append(msgs)

    except Exception as e:
        logger.error(f"  Error reading {filepath}: {e}")
    return results


def read_txt_file(filepath: Path) -> List[List[Dict]]:
    """Read TXT file, split into conversations."""
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Split conversations by double newline + separator patterns
        separator_patterns = [
            r"\n\s*---+\s*\n",
            r"\n\s*===+\s*\n",
            r"\n\s*\*\*\*+\s*\n",
            r"\n{3,}",
        ]
        conversations = [text]
        for sep in separator_patterns:
            new_conversations = []
            for conv in conversations:
                parts = re.split(sep, conv)
                new_conversations.extend(parts)
            conversations = new_conversations

        logger.info(f"  [{filepath.name}] Found {len(conversations)} conversation blocks")

        for conv_text in conversations:
            if len(conv_text.strip()) < 20:
                continue
            msgs = parse_txt_conversation(conv_text)
            if msgs:
                results.append(msgs)

    except Exception as e:
        logger.error(f"  Error reading {filepath}: {e}")
    return results


# ---------------------------------------------------------------------------
# Ensure each conversation has a system prompt
# ---------------------------------------------------------------------------

def normalize_conversation(
    msgs: List[Dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    add_system_if_missing: bool = True,
) -> List[Dict]:
    """Ensure the conversation starts with a system message."""
    if not msgs:
        return msgs
    if msgs[0]["role"] != "system":
        if add_system_if_missing:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def validate_conversation(msgs: List[Dict]) -> bool:
    """Check conversation is valid: has user+assistant turns, non-empty."""
    roles = [m["role"] for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return False
    if any(not m["content"].strip() for m in msgs if m["role"] != "system"):
        return False
    return True


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_directory(
    input_dir: str,
    output_path: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    add_system: bool = True,
) -> int:
    """Process all supported files in a directory."""
    input_path = Path(input_dir)
    all_conversations = []

    supported_exts = {".json", ".jsonl", ".csv", ".txt", ".xlsx", ".xls"}
    files = sorted([
        f for f in input_path.rglob("*")
        if f.suffix.lower() in supported_exts and f.is_file()
    ])

    if not files:
        logger.warning(f"No supported files found in {input_dir}")
        return 0

    logger.info(f"Found {len(files)} file(s) to process:")

    for filepath in files:
        logger.info(f"Processing: {filepath}")
        ext = filepath.suffix.lower()
        
        if ext in (".json", ".jsonl"):
            convs = read_json_file(filepath)
        elif ext in (".csv", ".xlsx", ".xls"):
            convs = read_csv_file(filepath)
        elif ext == ".txt":
            convs = read_txt_file(filepath)
        else:
            continue

        logger.info(f"  → Extracted {len(convs)} conversations")
        all_conversations.extend(convs)

    # Normalize & validate
    valid_conversations = []
    for conv in all_conversations:
        conv = normalize_conversation(conv, system_prompt, add_system)
        if validate_conversation(conv):
            valid_conversations.append(conv)

    logger.info(f"\nTotal: {len(all_conversations)} raw → {len(valid_conversations)} valid conversations")

    # Write output JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in valid_conversations:
            obj = {"messages": conv}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info(f"Saved to: {output_path}")
    return len(valid_conversations)


def inspect_directory(input_dir: str) -> None:
    """Report on data format without converting."""
    input_path = Path(input_dir)
    supported_exts = {".json", ".jsonl", ".csv", ".txt", ".xlsx", ".xls"}
    files = sorted([
        f for f in input_path.rglob("*")
        if f.suffix.lower() in supported_exts and f.is_file()
    ])

    print(f"\n{'='*60}")
    print(f"INSPECTION REPORT: {input_dir}")
    print(f"{'='*60}")
    print(f"Files found: {len(files)}")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    print()

    for filepath in files[:3]:  # Inspect first 3 files
        print(f"\n--- {filepath.name} ---")
        ext = filepath.suffix.lower()
        if ext in (".json", ".jsonl"):
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                # Show first sample
                sample = json.loads(lines[0]) if len(lines) > 1 else json.loads("".join(lines))
                if isinstance(sample, list):
                    sample = sample[0]
                print(f"  Format: JSON/JSONL | Total lines: {len(lines)}")
                print(f"  Keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
                if isinstance(sample, dict) and "messages" in sample:
                    msgs = sample["messages"]
                    print(f"  Sample has {len(msgs)} turns")
                    for m in msgs[:2]:
                        print(f"    [{m.get('role')}]: {str(m.get('content', ''))[:100]}...")
            except Exception as e:
                print(f"  Error: {e}")
        elif ext == ".csv":
            try:
                df = pd.read_csv(filepath, nrows=5)
                print(f"  Format: CSV | Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Sample row:\n{df.iloc[0].to_dict()}")
            except Exception as e:
                print(f"  Error: {e}")
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read(500)
            print(f"  Format: TXT | Preview:\n{content}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert astrologer chat data to ChatML format for Qwen2.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, help="Directory containing raw chat data files")
    group.add_argument("--input_file", type=str, help="Single input file to convert")

    parser.add_argument(
        "--output", type=str, default="./dataset_chatml.jsonl",
        help="Output JSONL file path (default: ./dataset_chatml.jsonl)"
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="Override default system prompt (optional)"
    )
    parser.add_argument(
        "--no_system", action="store_true",
        help="Don't add system prompt if missing"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect data format and exit without converting"
    )

    args = parser.parse_args()

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    if args.inspect:
        target = args.input_dir or str(Path(args.input_file).parent)
        inspect_directory(target)
        return

    if args.input_dir:
        count = convert_directory(
            args.input_dir, args.output, system_prompt, not args.no_system
        )
    else:
        # Single file: wrap in a temp dir approach
        filepath = Path(args.input_file)
        ext = filepath.suffix.lower()
        if ext in (".json", ".jsonl"):
            convs = read_json_file(filepath)
        elif ext in (".csv", ".xlsx", ".xls"):
            convs = read_csv_file(filepath)
        elif ext == ".txt":
            convs = read_txt_file(filepath)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return

        valid_conversations = []
        for conv in convs:
            conv = normalize_conversation(conv, system_prompt, not args.no_system)
            if validate_conversation(conv):
                valid_conversations.append(conv)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in valid_conversations:
                f.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")

        logger.info(f"Converted {len(valid_conversations)} conversations → {args.output}")
        count = len(valid_conversations)

    print(f"\n✅ Done! {count} conversations saved to: {args.output}")


if __name__ == "__main__":
    main()
