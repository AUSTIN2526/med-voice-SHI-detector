#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen inference with full Chinese support and repeated match handling"""

import argparse
import json
import re
import string
from pathlib import Path
from datetime import datetime

import spacy
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

pair_re = re.compile(r"\s*、\s*")
token_re = re.compile(r"\w+")
punct_table = str.maketrans("", "", string.punctuation)


def is_chinese(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def tokenize(text):
    return token_re.findall(text.lower())


def merge_short_sentences(sents, min_chars=20):
    merged = []
    buff = ""
    for sent in sents:
        candidate = f"{buff} {sent}".strip() if buff else sent
        if len(candidate) < min_chars:
            buff = candidate
        else:
            merged.append(candidate)
            buff = ""
    if buff:
        merged.append(buff)
    return merged


def sliding_windows(sents, max_len):
    res = []
    n = len(sents)
    if n == 0:
        return res
    for l in range(1, min(max_len, n) + 1):
        res.append(" ".join(sents[0:l]))
    for i in range(1, n - max_len + 1):
        res.append(" ".join(sents[i: i + max_len]))
    if n >= 2:
        res.append(" ".join(sents[-2:]))
    res.append(sents[-1])
    seen = set()
    uniq = []
    for w in res:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq


def parse_output(text):
    pairs = []
    for pair in pair_re.split(text.strip()):
        if "|" in pair:
            phi, word = pair.split("|", 1)
            pairs.append((phi.strip(), word.strip()))
    return pairs


def build_flat_tokens(words):
    flat = []
    for idx, word in enumerate(words):
        for tok in tokenize(word["text"]):
            flat.append((tok, idx))
    return flat


def align_once(word, search_from, words, flat_tokens, used_pos, replaceable_by_start):
    search_tokens = tokenize(word)
    if not search_tokens or len(word) < 2:
        return None
    m = len(search_tokens)
    N = len(flat_tokens)
    i = search_from
    while i <= N - m:
        if flat_tokens[i][0] != search_tokens[0]:
            i += 1
            continue
        candidate_idxs = [i + k for k in range(m)]
        if any(flat_tokens[idx][0] != search_tokens[k] for k, idx in enumerate(candidate_idxs)):
            i += 1
            continue
        start_idx = flat_tokens[candidate_idxs[0]][1]
        end_idx = flat_tokens[candidate_idxs[-1]][1]
        overlap = any(idx in used_pos for idx in candidate_idxs)
        if not overlap:
            used_pos.update(candidate_idxs)
            return words[start_idx]["start"], words[end_idx]["end"], i + 1
        prev = replaceable_by_start.get(start_idx)
        if prev:
            prev_end_idx, prev_token_indices = prev
            if end_idx > prev_end_idx:
                used_pos.update(candidate_idxs)
                used_pos.update(prev_token_indices)
                replaceable_by_start[start_idx] = (end_idx, list(set(prev_token_indices) | set(candidate_idxs)))
                return words[start_idx]["start"], words[end_idx]["end"], i + 1
        i += 1
    return None


def process_json(json_path, model, tokenizer, system_prompt, max_window):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    words = data["words"]
    raw_text = "".join(word["text"] for word in words)
    fid = json_path.stem
    predictions = []
    output_pairs = []

    if is_chinese(raw_text):
        # 中文處理：整段處理 + 字元比對
        prompt = f"### PROMPT\n{system_prompt}\n\n### INPUT\n{raw_text}\n\n### OUTPUT\n"
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inp, max_new_tokens=300)
        out_text = tokenizer.decode(out_ids[0][inp.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        output_pairs.append((raw_text, out_text))

        flat_text = "".join(word["text"] for word in words)
        char_to_idx = []
        for idx, word in enumerate(words):
            for _ in word["text"]:
                char_to_idx.append(idx)

        if out_text:
            for phi, phrase in parse_output(out_text):
                for match in re.finditer(re.escape(phrase), flat_text):
                    start_char = match.start()
                    end_char = match.end() - 1
                    if end_char < len(char_to_idx):
                        s_idx = char_to_idx[start_char]
                        e_idx = char_to_idx[end_char]
                        s = words[s_idx]["start"]
                        e = words[e_idx]["end"]
                        predictions.append((fid, phi, s, e, phrase))
    else:
        # 英文處理：句子切分 + sliding windows
        nlp = spacy.load("en_core_web_lg")
        nlp.add_pipe("sentencizer")
        sents = [s.text.strip() for s in nlp(" ".join(word["text"] for word in words)).sents if s.text.strip()]
        sents = merge_short_sentences(sents, min_chars=10)
        windows = sliding_windows(sents, max_window)
        flat_tokens = build_flat_tokens(words)

        for win in windows:
            prompt = f"### PROMPT\n{system_prompt}\n\n### INPUT\n{win}\n\n### OUTPUT\n"
            inp = tokenizer(prompt, return_tensors="pt").to(model.device)
            out_ids = model.generate(**inp, max_new_tokens=100)
            out_text = tokenizer.decode(out_ids[0][inp.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            output_pairs.append((win, out_text))

            toks = tokenize(win)
            if toks:
                try:
                    fst = next(i for i, (t, _) in enumerate(flat_tokens) if t == toks[0])
                    lst = max(i for i, (t, _) in enumerate(flat_tokens) if t == toks[-1])
                    ws = words[flat_tokens[fst][1]]["start"]
                    we = words[flat_tokens[lst][1]]["end"]
                except:
                    ws, we = 0.0, words[-1]["end"]
            else:
                ws, we = 0.0, words[-1]["end"]

            if out_text:
                for phi, word in parse_output(out_text):
                    sf = 0
                    used = set()
                    rep = {}
                    while True:
                        span = align_once(word, sf, words, flat_tokens, used, rep)
                        if not span:
                            break
                        s, e, sf = span
                        if s >= ws and e <= we:
                            predictions.append((fid, phi, s, e, word))

    preds_sorted = sorted(predictions, key=lambda r: (r[3] - r[2]), reverse=True)
    chosen = []
    for rec in preds_sorted:
        _, _, s, e, _ = rec
        if not any(not (e <= cs or s >= ce) for _, _, cs, ce, _ in chosen):
            chosen.append(rec)
    predictions = sorted(chosen, key=lambda r: r[2])
    return predictions, output_pairs


def write_outputs(fid, output_pairs, cleaned_records, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = {
        "fid": fid,
        "all_outputs": [{"input": inp, "output": out} for inp, out in output_pairs],
        "predictions": [
            {"fid": fid, "phi": phi, "start": round(s, 2), "end": round(e, 2), "word": w}
            for fid, phi, s, e, w in cleaned_records
        ]
    }
    json_path = out_dir / f"{fid}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)


def load_model(base_model, lora_path, torch_dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="cpu")
    model = PeftModel.from_pretrained(base, lora_path, torch_dtype=torch_dtype, device_map="cpu")
    model = model.merge_and_unload().to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Qwen inference with Chinese support")
    parser.add_argument("json_path")
    parser.add_argument("--output-dir", default="ner_result")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B-Base")
    parser.add_argument("--lora-path", default="all_adapter/qwen_adapter")
    parser.add_argument("--system-prompt-file", default="aicup_data/prompt.txt")
    parser.add_argument("--max-window", type=int, default=3)
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 🔧 Loading model...")
    torch_dtype = getattr(torch, args.torch_dtype)
    model, tokenizer = load_model(args.base_model, args.lora_path, torch_dtype, args.device)
    print("✅ Model loaded successfully.")
    print("=" * 60)

    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    path = Path(args.json_path)
    files = list(path.glob("*.json")) if path.is_dir() else [path]

    for i, file in enumerate(files, 1):
        print(f"\n📄 [{i}/{len(files)}] Processing file: {file.name}")
        selected, output_pairs = process_json(file, model, tokenizer, system_prompt, args.max_window)
        write_outputs(file.stem, output_pairs, selected, Path(args.output_dir))
        print(f"✅ Done: {len(selected)} predictions saved for {file.stem}")
        print("-" * 50)

    print(f"\n🎉 All {len(files)} file(s) processed.")
    print(f"🗂️  Outputs saved to: {Path(args.output_dir).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
