import argparse
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.preprocess_utils import (
    load_phi_sentences, reduce_phi_null_data_sliding,
    sliding_window_transform, load_dataset
)
from utils.load_model_tokenizer import initialize_qwen_with_lora, initialize_whisper_with_lora
from utils.enhance_pytorch_dataloader import NERDataset, SpeechSeq2SeqDataset
from utils.unified_trainer import UnifiedLoRATrainer


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_ner(args):
    set_seeds(2526)
    model, tokenizer, device, dtype = initialize_qwen_with_lora(
        model_name=args.qwen_model_name,
        enable_neftune=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    print(f"Model on {device} with dtype {dtype}")

    results = load_phi_sentences(args.ner_data_path)
    results = sliding_window_transform(results, 3)
    final_sentences, final_labels = reduce_phi_null_data_sliding(results, 0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(
        final_sentences, final_labels, train_size=0.8, random_state=2526, shuffle=True
    )

    with open(args.ner_prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    trainset = NERDataset(tokenizer, x_train, y_train, system_prompt)
    validset = NERDataset(tokenizer, x_valid, y_valid, system_prompt)

    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=trainset.collate_fn)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False, collate_fn=validset.collate_fn)

    adapter_save_path = os.path.join(args.adapter_path, "qwen_adapter")
    os.makedirs(adapter_save_path, exist_ok=True)

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


def train_asr(args):
    set_seeds(2526)
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

    audio_data = load_dataset(args.asr_audio_dir, args.asr_transcript_file, target_sr=16000)
    train_data, valid_data = train_test_split(audio_data, train_size=0.8, random_state=2526, shuffle=True)
    train_dataset = SpeechSeq2SeqDataset(train_data, processor, 16000, augment=True)
    valid_dataset = SpeechSeq2SeqDataset(valid_data, processor, 16000, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=valid_dataset.collate_fn)

    adapter_save_path = os.path.join(args.adapter_path, "whisper_adapter")
    os.makedirs(adapter_save_path, exist_ok=True)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['ner', 'asr'], required=True, help='Training task: ner or asr')

    # Unified LoRA adapter output root directory
    parser.add_argument('--adapter_path', type=str, default="all_adapter", help='Directory to store adapter models')

    # NER args
    parser.add_argument('--qwen_model_name', type=str, default='Qwen/Qwen3-14B-Base')
    parser.add_argument('--ner_data_path', type=str, default='aicup_data/train/phi_sentence_locations.json')
    parser.add_argument('--ner_prompt_path', type=str, default='aicup_data/prompt.txt')

    # ASR args
    parser.add_argument('--whisper_model_name', type=str, default='nyrahealth/CrisperWhisper')
    parser.add_argument('--asr_audio_dir', type=str, default='aicup_data/train/audio')
    parser.add_argument('--asr_transcript_file', type=str, default='aicup_data/train/sentence_locations.json')

    args = parser.parse_args()

    if args.task == 'ner':
        train_ner(args)
    elif args.task == 'asr':
        train_asr(args)
