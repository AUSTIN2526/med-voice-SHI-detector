import librosa
import json
from collections import Counter
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_dataset(audio_dir, transcript_file, target_sr=16000):
    dataset = []
    transcript_path = Path(transcript_file)

    # 檢查轉錄檔案是否存在
    if not transcript_path.is_file():
        raise FileNotFoundError(f"{transcript_path}")

    print('───────────────────────────────────────────────────────────────')
    with transcript_path.open('r', encoding="utf-8-sig") as f:
        raw_data = json.load(f)

    print(f'>>> 共有 {len(raw_data)} 筆紀錄，開始載入並篩選音訊...')
    for record_id_str, record in tqdm(raw_data.items(), desc="載入音訊中", unit="記錄"):
        try:
            file_id = int(record_id_str)
        except ValueError:
            file_id = record_id_str

        # 建立對應的音訊路徑
        wav_path = Path(audio_dir) / f"{file_id}.wav"
        if not wav_path.is_file():
            tqdm.write(f"[警告] 找不到音訊檔：{wav_path} ，此筆紀錄跳過。")
            continue

        try:
            # 使用 librosa 載入音訊並重採樣
            audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
        except Exception as e:
            tqdm.write(f"[錯誤] 載入失敗：{wav_path}，錯誤訊息：{e}")
            continue

        # 取得句子欄位資料
        sentences = record.get('sentences') or [record.get('sentence', '')]
        sentence_text = " ".join(sentences).strip()

        # 加入資料列表
        dataset.append({
            'audio': audio,
            'sentence': sentence_text,
            'file_id': file_id
        })

    print(f'>>> 篩選與載入完成，最終保留 {len(dataset)} / {len(raw_data)} 筆紀錄')
    print('───────────────────────────────────────────────────────────────\n')
    return dataset


# 載入 PHI 標註資料，過濾無標註或標註缺失的紀錄
def load_phi_sentences(filepath):
    discarded_types = []       # 被捨棄的 PHI 類型統計
    discarded_docs = 0         # 被捨棄的文件數
    total_type_counts = Counter()  # 所有 PHI 類型統計
    file_path = Path(filepath)

    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path}")

    with file_path.open('r', encoding='utf-8') as f:
        raw_data = json.load(f)

    results = {}
    for record_id, record in raw_data.items():
        phi_not_found = record.get('phi_not_found', [])
        phi_results = record.get('phi_results', [])

        # 如果全部都沒有標註，則跳過
        if all(not ents for ents in phi_results):
            discarded_docs += 1
            continue

        # 如果有遺漏的標註資料，也跳過
        if phi_not_found:
            discarded_docs += 1
            types = [ent['phi_type'] for ent in phi_not_found]
            discarded_types.extend(types)
            total_type_counts.update(types)
            continue

        # 處理正常紀錄
        sentences = [s.replace(r'\"', '') for s in record.get('sentences', [])]
        annotations = []

        for entities in phi_results:
            if entities:
                total_type_counts.update(ent['phi_type'] for ent in entities)
                annotations.append([
                    f"{ent['phi_type']}|{ent['phi_content']}" for ent in entities
                ])
            else:
                annotations.append(['PHI|NULL'])

        results[record_id] = {
            'sentences': sentences,
            'labels': annotations
        }

    # 計算保留與丟棄的統計
    discarded_counter = Counter(discarded_types)
    kept_counter = {
        phi_type: total_type_counts[phi_type] - discarded_counter[phi_type]
        for phi_type in total_type_counts
    }

    print('───────────────────────────────────────────────────────────────')
    print(f">>> 被跳過（丟棄）的文件數: {discarded_docs}")
    print('>>> 各 phi_type 統計 (共出現、丟棄、保留、丟棄比例、保留比例):')
    for phi_type, total in total_type_counts.items():
        discarded = discarded_counter[phi_type]
        kept = kept_counter[phi_type]
        print(f"  - {phi_type}: 總共 {total} 次，丟棄 {discarded} 次 ({discarded/total*100:.2f}%)，保留 {kept} 次 ({kept/total*100:.2f}%)")
    print('───────────────────────────────────────────────────────────────\n')
    return results


# 將資料轉為滑動視窗方式組合（每組包含 1 到 window_size 句子）
def sliding_window_transform(data: dict, window_size: int = 3):
    result = {}
    for id_, content in data.items():
        sentences, labels = content["sentences"], content["labels"]
        new_sentences, new_labels = [], []
        for start in range(len(sentences)):
            for end in range(start + 1, min(start + window_size + 1, len(sentences) + 1)):
                sentence_chunk = sentences[start:end]
                label_chunk = labels[start:end]

                # 將多個句子的標註合併
                label_merged = ["、".join(lbl) for lbl in label_chunk]
                filtered_labels = [lbl for lbl in label_merged if lbl != "PHI|NULL"]
                final_label = "、".join(filtered_labels) if filtered_labels else "PHI|NULL"

                new_sentences.append(" ".join(sentence_chunk))
                new_labels.append(final_label)

        result[id_] = {
            "sentences": new_sentences,
            "labels": new_labels
        }
    return result


# 過濾過多的 PHI|NULL 標註，保留比例可調整
def reduce_phi_null_data_sliding(data, ratio=0.1, seed=2526):
    all_sentences = []
    all_labels = []

    for entry in data.values():
        all_sentences.extend(entry["sentences"])
        all_labels.extend(entry["labels"])

    x_arr, y_arr = np.array(all_sentences), np.array(all_labels)
    is_null = (y_arr == 'PHI|NULL')

    null_x, null_y = x_arr[is_null], y_arr[is_null]
    phi_x, phi_y = x_arr[~is_null], y_arr[~is_null]

    print('───────────────────────────────────────────────────────────────')
    print(f'有 PHI 的資料: {len(phi_x)}，只有 PHI|NULL 的資料: {len(null_x)}')

    # 決定要保留多少 PHI|NULL 的樣本
    keep_num = min(int(len(phi_x) * ratio), len(null_x))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(null_x), size=keep_num, replace=False)

    # 合併保留的資料
    final_x = np.concatenate([phi_x, null_x[sampled_idx]])
    final_y = np.concatenate([phi_y, null_y[sampled_idx]])

    print(f'資料過濾後總數: {len(final_x)}')
    print('───────────────────────────────────────────────────────────────\n')
    return final_x.tolist(), final_y.tolist()
