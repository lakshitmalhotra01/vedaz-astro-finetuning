#!/usr/bin/env python3
"""
evaluate_model.py
=================
Sanity-check evaluation for the fine-tuned Qwen2.5-7B-Instruct astrologer model.

Runs sample generations on a set of astrology-themed prompts and saves:
  - evaluate_results.json   : structured results
  - evaluate_results.md     : human-readable report

Usage:
  # Evaluate merged model
  python evaluate_model.py --model_path ./output/qwen25_astro_merged

  # Evaluate using LoRA adapter (no merge needed)
  python evaluate_model.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_path ./output/qwen25_astro_qlora

  # Interactive mode (chat with the model)
  python evaluate_model.py --model_path ./output/qwen25_astro_merged --interactive
"""

import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Vedaz, an expert Vedic astrologer with deep knowledge of Jyotish shastra. "
    "You have mastered kundli (birth chart) analysis, rashi (zodiac signs), nakshatra (lunar mansions), "
    "planetary positions, dasha systems (Vimshottari, Yogini), bhava (houses), yogas, and divisional charts. "
    "You speak with warmth, empathy, and authority. When a user shares their birth details, you carefully "
    "analyze their chart before giving predictions. You always mention specific dates, planetary transits, "
    "or dasha periods when predicting future events. You balance spiritual wisdom with practical guidance."
)

# ---------------------------------------------------------------------------
# Evaluation prompts (astrology-themed)
# ---------------------------------------------------------------------------
EVAL_PROMPTS = [
    {
        "id": "career_eval",
        "category": "Career",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Namaste ji! I was born on March 15, 1992, at 6:30 AM in Delhi. My career has been very unstable for the past 2 years. Can you look at my kundli and tell me when things will improve?"},
        ],
        "expected_elements": ["rashi", "dasha", "saturn", "house", "date"],
    },
    {
        "id": "marriage_eval",
        "category": "Marriage",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Namaste. My DOB is July 22, 1995, time 11:45 PM, born in Mumbai. I am worried about my marriage. My family has been looking for a suitable match for 3 years but nothing is working out. When will I get married?"},
        ],
        "expected_elements": ["venus", "7th house", "dasha", "nakshatra", "year"],
    },
    {
        "id": "health_eval",
        "category": "Health",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "My birth details: 5 October 1988, 3:15 AM, Jaipur. I have been suffering from some health issues and doctors cannot find the cause. Can astrology tell me something about my health?"},
        ],
        "expected_elements": ["6th house", "mars", "saturn", "dasha", "remedies"],
    },
    {
        "id": "finance_eval",
        "category": "Finance",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Pranam Guruji. My DOB is 14 February 1990, born at 8:20 AM in Pune. I run a small business but have been facing huge financial losses. Is there any planetary combination causing this? When will my financial situation improve?"},
        ],
        "expected_elements": ["2nd house", "jupiter", "rahu", "dasha", "month"],
    },
    {
        "id": "overseas_eval",
        "category": "Foreign Travel/Settlement",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Hello. My birth date is November 28, 1997, at 2:00 PM, Hyderabad. I have been trying to go abroad for higher studies for 2 years but keep facing rejections and delays. Does my kundli show foreign settlement?"},
        ],
        "expected_elements": ["12th house", "rahu", "nakshatra", "dasha", "year"],
    },
    {
        "id": "brief_query",
        "category": "Quick Query",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is the significance of Mars in the 8th house?"},
        ],
        "expected_elements": ["8th house", "mars", "longevity", "transformation"],
    },
    {
        "id": "multi_turn",
        "category": "Multi-turn Conversation",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Namaste ji. Can you help me with my kundli reading?"},
            {"role": "assistant", "content": "Namaste! Of course, I would be happy to help you with your kundli reading. Please share your birth details — your date of birth, exact time of birth, and place of birth — and I will analyze your chart carefully."},
            {"role": "user", "content": "My name is Priya. I was born on September 4, 1993, at 7:45 AM in Chennai. I want to know about my career prospects."},
        ],
        "expected_elements": ["kanya", "virgo", "dasha", "10th house", "date"],
    },
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, adapter_path: str = None, use_4bit: bool = False):
    """Load the fine-tuned model (merged or with adapter)."""
    logger.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if use_4bit:
        logger.info("Loading model in 4-bit for memory efficiency...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        logger.info("Loading model in float16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    if adapter_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Adapter loaded (model not merged - running inference with adapter)")

    model.eval()
    logger.info("Model ready for inference")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> tuple[str, float]:
    """Generate a response for a conversation."""
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # True for inference (add assistant turn start)
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time

    # Decode only the new tokens (skip the input prompt)
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    tokens_per_sec = len(new_tokens) / elapsed if elapsed > 0 else 0
    return response, tokens_per_sec


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_quality_score(response: str, expected_elements: list) -> dict:
    """Simple keyword-based quality check."""
    response_lower = response.lower()
    found = []
    missing = []
    for elem in expected_elements:
        if elem.lower() in response_lower:
            found.append(elem)
        else:
            missing.append(elem)
    
    score = len(found) / len(expected_elements) if expected_elements else 0
    return {
        "score": round(score, 2),
        "found_elements": found,
        "missing_elements": missing,
        "response_length": len(response.split()),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    model,
    tokenizer,
    output_dir: str = ".",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Run all evaluation prompts and save results."""
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING EVALUATION: {len(EVAL_PROMPTS)} prompts")
    logger.info(f"{'='*60}")

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        logger.info(f"\n[{i}/{len(EVAL_PROMPTS)}] {prompt['category']}: {prompt['id']}")
        
        response, tps = generate_response(
            model, tokenizer,
            prompt["messages"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        quality = compute_quality_score(response, prompt["expected_elements"])

        result = {
            "id": prompt["id"],
            "category": prompt["category"],
            "last_user_message": prompt["messages"][-1]["content"],
            "generated_response": response,
            "tokens_per_second": round(tps, 1),
            "quality": quality,
        }
        results.append(result)

        # Print to console
        print(f"\n{'─'*50}")
        print(f"Category: {prompt['category']}")
        print(f"User: {prompt['messages'][-1]['content'][:150]}...")
        print(f"\nAssistant: {response[:400]}...")
        print(f"\nQuality Score: {quality['score']:.0%} | Speed: {tps:.1f} tok/s")
        print(f"Found: {quality['found_elements']}")
        print(f"Missing: {quality['missing_elements']}")

    # Save JSON
    json_path = output_path / "evaluate_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {json_path}")

    # Save Markdown report
    md_path = output_path / "evaluate_results.md"
    _save_markdown_report(results, md_path)
    logger.info(f"Markdown report saved to: {md_path}")

    # Print summary
    avg_score = sum(r["quality"]["score"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Prompts evaluated:  {len(results)}")
    print(f"  Average quality:    {avg_score:.0%}")
    print(f"  Average speed:      {avg_tps:.1f} tokens/sec")
    print(f"{'='*60}")

    return results


def _save_markdown_report(results: list, path: Path):
    """Save evaluation results as Markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Fine-tuned Model Evaluation Report",
        f"*Generated: {now}*\n",
        "## Summary\n",
    ]
    
    avg_score = sum(r["quality"]["score"] for r in results) / len(results)
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total prompts | {len(results)} |")
    lines.append(f"| Average quality score | {avg_score:.0%} |")
    
    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
    lines.append(f"| Average generation speed | {avg_tps:.1f} tok/s |\n")
    
    lines.append("## Detailed Results\n")
    for r in results:
        lines.append(f"### {r['category']}: `{r['id']}`\n")
        lines.append(f"**User Query:**\n> {r['last_user_message']}\n")
        lines.append(f"**Model Response:**\n```\n{r['generated_response']}\n```\n")
        q = r["quality"]
        lines.append(f"**Quality:** {q['score']:.0%} | **Words:** {q['response_length']} | **Speed:** {r['tokens_per_second']} tok/s")
        lines.append(f"- ✅ Found: {', '.join(q['found_elements']) or 'none'}")
        lines.append(f"- ❌ Missing: {', '.join(q['missing_elements']) or 'none'}\n")
        lines.append("---\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_chat(model, tokenizer):
    """Interactive terminal chat with the model."""
    print("\n" + "="*60)
    print("INTERACTIVE ASTROLOGER CHAT (type 'quit' to exit, 'reset' to start new conversation)")
    print("="*60 + "\n")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[Conversation reset]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        
        print("Astrologer: ", end="", flush=True)
        response, tps = generate_response(model, tokenizer, messages)
        print(response)
        print(f"  [{tps:.1f} tok/s]\n")
        
        messages.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen2.5 astrologer model")
    parser.add_argument("--model_path", type=str, required=True,
        help="Path to merged model or base model ID")
    parser.add_argument("--adapter_path", type=str, default=None,
        help="Path to LoRA adapter (if not merged)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_output",
        help="Directory to save evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=512,
        help="Max tokens to generate per response")
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature (0.0 = greedy, 1.0 = more random)")
    parser.add_argument("--use_4bit", action="store_true",
        help="Load model in 4-bit for memory efficiency")
    parser.add_argument("--interactive", action="store_true",
        help="Start interactive chat mode instead of benchmark")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.adapter_path, args.use_4bit
    )

    if args.interactive:
        interactive_chat(model, tokenizer)
    else:
        run_evaluation(
            model, tokenizer,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
