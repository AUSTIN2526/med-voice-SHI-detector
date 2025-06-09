#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import zipfile
from pathlib import Path


def merge_and_sort_task1(input_dir: Path, output_file: Path):
    """
    合併 whisper_outputs 資料夾中的 JSON 檔案，提取 text 欄位，並依 ID 排序後寫入目標檔案。
    """
    dataset = [
        f"{p.stem}\t{json.loads(p.read_text(encoding='utf-8-sig'))['text']}"
        for p in input_dir.glob('*.json')
    ]
    dataset.sort(key=lambda x: int(x.split('\t', 1)[0]))
    output_file.write_text("\n".join(dataset), encoding='utf-8')
    print(f"✅ task1 合併並排序完成，輸出檔案位於：{output_file}")


def merge_and_sort_task2_json(input_dir: Path, pattern: str, output_file: Path):
    """
    合併 qwen_outputs 資料夾中的 *_results.json 檔案，提取 predictions 欄位，並依 ID 排序寫入 txt。
    """
    file_list = list(input_dir.glob(pattern))
    merged_lines = []

    for f in file_list:
        data = json.loads(f.read_text(encoding='utf-8'))
        for item in data.get("predictions", []):
            line = f"{item['fid']}\t{item['phi']}\t{item['start']:.2f}\t{item['end']:.2f}\t{item['word']}"
            merged_lines.append(line)

    merged_lines.sort(key=lambda l: int(l.split('\t', 1)[0]))
    output_file.write_text("\n".join(merged_lines), encoding='utf-8')
    print(f"✅ task2 合併並排序完成，輸出檔案位於：{output_file}")


def compress_outputs(files: list[Path], zip_path: Path):
    """
    將指定的檔案壓縮成一個 zip 檔案。
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in files:
            if file.exists():
                z.write(file, arcname=file.name)
            else:
                print(f"⚠️ 警告：檔案不存在，無法加入 ZIP：{file}")
    print(f"📦 壓縮完成，輸出 ZIP 檔案位於：{zip_path}")


def main():
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Task 1 ===
    task1_out = output_dir / 'task1_answer.txt'
    merge_and_sort_task1(Path('asr_result'), task1_out)

    # === Task 2（JSON 輸出版）===
    task2_out = output_dir / 'task2_answer.txt'
    merge_and_sort_task2_json(Path('ner_result'), '*_results.json', task2_out)

    # === 壓縮 ===
    zip_file = output_dir / 'submission.zip'
    compress_outputs([task1_out, task2_out], zip_file)


if __name__ == '__main__':
    main()
