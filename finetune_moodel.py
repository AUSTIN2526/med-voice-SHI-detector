import argparse  # 用於處理命令列參數
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 自訂的工具模組
from utils.preprocess_utils import (
    load_phi_sentences,          # 讀取 NER 訓練資料
    reduce_phi_null_data_sliding,# 去除無標註資料
    sliding_window_transform,    # 滑動視窗轉換文本
    load_dataset                 # 載入 ASR 音訊資料
)
from utils.load_model_tokenizer import (
    initialize_qwen_with_lora,  # 初始化 Qwen 模型並加載 LoRA
    initialize_whisper_with_lora # 初始化 Whisper 模型並加載 LoRA
)
from utils.enhance_pytorch_dataloader import (
    NERDataset,                 # NER 資料集處理類別
    SpeechSeq2SeqDataset       # ASR 資料集處理類別
)
from utils.unified_trainer import UnifiedLoRATrainer  # 統一的訓練器模組
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# 設定隨機種子，確保可重現性
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# NER 任務的訓練流程
def train_ner(args):
    set_seeds(2526)  # 設定種子
    # 初始化 Qwen 模型並應用 LoRA
    model, tokenizer, device, dtype = initialize_qwen_with_lora(
        model_name=args.qwen_model_name,
        enable_neftune=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    print(f"Model on {device} with dtype {dtype}")

    # 載入並預處理 NER 訓練資料
    results = load_phi_sentences(args.ner_data_path)
    results = sliding_window_transform(results, 3)  # 使用滑動視窗技術處理長句
    final_sentences, final_labels = reduce_phi_null_data_sliding(results, 0.1)  # 過濾無標註資料
    x_train, x_valid, y_train, y_valid = train_test_split(
        final_sentences, final_labels, train_size=0.8, random_state=2526, shuffle=True
    )

    # 載入 NER 提示詞
    with open(args.ner_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # 建立 PyTorch 資料集與 DataLoader
    trainset = NERDataset(tokenizer, x_train, y_train, system_prompt)
    validset = NERDataset(tokenizer, x_valid, y_valid, system_prompt)

    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=trainset.collate_fn)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False, collate_fn=validset.collate_fn)

    # 建立 adapter 存檔目錄
    adapter_save_path = os.path.join(args.adapter_path, "qwen_adapter")
    os.makedirs(adapter_save_path, exist_ok=True)

    # 使用統一訓練器訓練模型
    trainer = UnifiedLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.025,
        scheduler_type='cosine',
        early_stopping=5,
        adapter_path=adapter_save_path,
        enable_aug=False,
        enable_spike_detection=True,
    )
    trainer.train()

# ASR 任務的訓練流程
def train_asr(args):
    set_seeds(2526)
    # 初始化 Whisper 模型並應用 LoRA
    model, processor, device, dtype = initialize_whisper_with_lora(
        args.whisper_model_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'],
        freeze_encoder_modules=None,
        language=None
    )
    print(f"Model on {device} with dtype {dtype}")
    
    # 載入音訊資料
    target_sr = 16000
    audio_data = load_dataset(args.asr_audio_dir, args.asr_transcript_file, target_sr=target_sr)
    train_data, valid_data = train_test_split(audio_data, train_size=0.8, random_state=2526, shuffle=True)

    # 建立 ASR 的 PyTorch 資料集
    train_dataset = SpeechSeq2SeqDataset(train_data, processor, target_sr, augment=True)
    valid_dataset = SpeechSeq2SeqDataset(valid_data, processor, target_sr, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=valid_dataset.collate_fn)

    # 建立 adapter 存檔目錄
    adapter_save_path = os.path.join(args.adapter_path, "whisper_adapter")
    os.makedirs(adapter_save_path, exist_ok=True)

    # 使用統一訓練器訓練模型
    trainer = UnifiedLoRATrainer(
        model=model,
        tokenizer=processor,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=12,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        scheduler_type='cosine',
        early_stopping=3,
        adapter_path=adapter_save_path,
        enable_aug=True,
        enable_spike_detection=False,
    )
    trainer.train()

# 主程式入口點
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['ner', 'asr'], required=True, help='Training task: ner or asr')

    # LoRA adapter 輸出資料夾
    parser.add_argument('--adapter_path', type=str, default="all_adapter", help='Directory to store adapter models')

    # NER 模型與資料參數
    parser.add_argument('--qwen_model_name', type=str, default='Qwen/Qwen3-14B-Base')
    parser.add_argument('--ner_data_path', type=str, default='aicup_data/train/sentence_locations.json')
    parser.add_argument('--ner_prompt_path', type=str, default='aicup_data/prompt.txt')

    # ASR 模型與資料參數
    parser.add_argument('--whisper_model_name', type=str, default='nyrahealth/CrisperWhisper')
    parser.add_argument('--asr_audio_dir', type=str, default='aicup_data/train/audio')
    parser.add_argument('--asr_transcript_file', type=str, default='aicup_data/train/sentence_locations.json')

    # 解析參數並執行對應任務
    args = parser.parse_args()
    if args.task == 'ner':
        train_ner(args)
    elif args.task == 'asr':
        train_asr(args)
