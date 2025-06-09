import json
from collections import Counter
from pathlib import Path
import spacy

# 嘗試載入 spaCy 的大型英文模型，如果尚未安裝則自動下載
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    from spacy.cli import download as spacy_download
    spacy_download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# 讀取 task1_answer.txt，格式為「ID<TAB>文本內容」
def read_text_file(path):
    result = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            pid, *rest = line.split("\t", maxsplit=1)
            if rest:
                result[pid] = rest[0]
    print(f"Loaded {len(result):,} text rows from {Path(path).name}")
    return result

# 讀取 task2_answer.txt，格式為「ID<TAB>phi_type<TAB>...<TAB>phi_content」
def read_phi_file(path):
    result = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(f'task2_answer file is a Error format Error on -> {parts}')
            pid, phi_type, *_unused, phi_content = parts[:5]
            result.setdefault(pid, []).append((phi_type, phi_content.strip()))
    print(f"Loaded PHI records for {len(result):,} IDs from {Path(path).name}")
    return result

# 將文本依據句子切分，並嘗試將 PHI 條目對應到句子中
def find_phi_in_sentences(text, phi_items, case_sensitive=True, min_chars=10):
    doc = nlp(text)
    raw_sents = [(s.start_char, s.end_char, s.text) for s in doc.sents]

    # 對短句進行合併以避免斷句影響匹配
    merged = []
    i = 0
    while i < len(raw_sents):
        start_i, end_i, txt_i = raw_sents[i]
        if len(txt_i.strip()) < min_chars and i + 1 < len(raw_sents):
            _, end_next, txt_next = raw_sents[i + 1]
            merged_text = txt_i.rstrip() + " " + txt_next.lstrip()
            merged.append((start_i, end_next, merged_text))
            i += 2
        else:
            merged.append((start_i, end_i, txt_i))
            i += 1

    sentences = [span[2].strip() for span in merged]
    sentence_matches = [[] for _ in merged]
    phi_not_found = []
    current_offset = 0

    # 判斷是否為單字邊界的輔助函數
    def _is_boundary(ch):
        return not (ch.isalpha() or ch.isdigit())

    # 對每個 PHI 條目進行搜尋
    for phi_index, (phi_type, phi_text) in enumerate(phi_items):
        found_match = None
        needle = phi_text if case_sensitive else phi_text.lower()

        for sent_idx, (sent_start, sent_end, sent_text) in enumerate(merged):
            if sent_end <= current_offset:
                continue

            allowed_start_in_sent = max(0, current_offset - sent_start)
            haystack_full = sent_text[allowed_start_in_sent:]
            haystack = haystack_full if case_sensitive else haystack_full.lower()

            # 搜尋 PHI 字串
            pos = haystack.find(needle)
            while pos != -1:
                abs_start = sent_start + allowed_start_in_sent + pos
                abs_end = abs_start + len(phi_text)

                before = sent_text[abs_start - sent_start - 1] if abs_start - sent_start > 0 else ""
                after_i = abs_end - sent_start
                after = sent_text[after_i] if after_i < len(sent_text) else ""

                # 檢查邊界（避免誤匹配 now → know）
                left_ok = before == "" or _is_boundary(before)
                right_ok = after == "" or after in ("'", "’") or _is_boundary(after)

                if not (left_ok and right_ok):
                    pos = haystack.find(needle, pos + 1)
                    continue

                # 找到匹配，記錄位置與相關資訊
                found_match = {
                    "phi_type": phi_type,
                    "phi_content": phi_text,
                    "sentence_index": sent_idx,
                    "start": abs_start,
                    "end": abs_end,
                }
                sentence_matches[sent_idx].append(found_match)
                current_offset = abs_end
                break

            if found_match:
                break

        if not found_match:
            phi_not_found.append({"phi_type": phi_type, "phi_content": phi_text})

    return sentences, sentence_matches, phi_not_found

# 主流程：處理多個文本 ID，分析 PHI 與句子對應情況
def analyze_phi_locations(text_file, phi_file, target_id=None):
    texts = read_text_file(text_file)
    phis = read_phi_file(phi_file)

    ids = [target_id] if target_id and target_id in texts else sorted(texts.keys())

    master = {}
    not_found_summary = {}
    total_counter = Counter()
    not_found_counter = Counter()
    consistency_errors = []

    for pid in ids:
        text = texts[pid]
        phi_items = phis.get(pid, [])

        for p_type, _ in phi_items:
            total_counter[p_type] += 1

        sentences, matches, not_found = find_phi_in_sentences(text, phi_items, case_sensitive=True)

        matched_count = sum(len(s) for s in matches)
        if matched_count + len(not_found) != len(phi_items):
            msg = (
                f"PID {pid}: matched_count ({matched_count}) + not_found ({len(not_found)}) != declared PHI items ({len(phi_items)})"
            )
            print("[Consistency Warning] " + msg)
            consistency_errors.append(msg)

        master[pid] = {
            "text": text,
            "sentences": sentences,
            "phi_results": matches,
            "phi_not_found": not_found,
        }

        if not_found:
            not_found_summary[pid] = not_found
            for item in not_found:
                not_found_counter[item["phi_type"]] += 1

    # 儲存結果至 JSON 檔案
    out_path = Path("aicup_data/train/sentence_locations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(master, fh, ensure_ascii=False, indent=2)
    print(f"Saved JSON → {out_path.name}")

    # 顯示未匹配結果
    if not_found_summary:
        print("\n未匹配 PHI 摘要:")
        for pid, items in not_found_summary.items():
            print(f"文件 ID {pid} 中有 {len(items)} 個未匹配的 PHI:")
            for item in items:
                print(f"  • {item['phi_type']}: '{item['phi_content']}'")

    # 一致性檢查報告
    if consistency_errors:
        print("\n一致性檢查發現以下問題:")
        for err in consistency_errors:
            print("  - " + err)
    else:
        print("\n一致性檢查通過，所有 PID 的匹配總數與原始 PHI 條目數量一致。")

    return master

# 指令列介面
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="使用 spaCy 切分句子並輸出 PHI JSON 位置報表，"
                    "改為僅以大小寫相符的『完全字串相等』逐句搜尋，"
                    "並在英文中加入單詞邊界邏輯，避免 now/know 類誤判，同時允許 Austin's → Austin 的合法匹配。"
    )
    parser.add_argument("--data_dir", default="aicup_data/train", help="資料目錄，預設為 aicup_data/train")
    parser.add_argument("--text_file", default=None, help="task1_answer.txt 的檔案路徑")
    parser.add_argument("--phi_file", default=None, help="task2_answer.txt 的檔案路徑")
    parser.add_argument("--id", "-i", dest="target_id")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    text_file = Path(args.text_file) if args.text_file else data_dir / "task1_answer.txt"
    phi_file = Path(args.phi_file) if args.phi_file else data_dir / "task2_answer.txt"

    analyze_phi_locations(text_file, phi_file, args.target_id)
