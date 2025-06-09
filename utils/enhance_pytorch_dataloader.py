import numpy as np
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 音訊增強類別：提供多種音訊增強策略，並可隨機混合使用
class AdvancedAudioAugmentation:
    
    def __init__(self):
        # 定義各種增強策略及其參數
        self.strategies = {
            'add_noise': {'weight': 0.30, 'min_factor': 0.0005, 'max_factor': 0.005},  # 添加雜訊
            'time_shift': {'weight': 0.30, 'min_shift': 0.03, 'max_shift': 0.08},      # 時間平移
            'time_mask': {'weight': 0.25, 'min_mask': 0.01, 'max_mask': 0.05},         # 隱藏部分音訊
            'volume_change': {'weight': 0.15, 'min_factor': 0.8, 'max_factor': 1.2},   # 音量改變
        }

        # 定義單一/雙重/三重混合增強的機率
        self.mix_strategies = {
            'single': 0.6,
            'double': 0.3,
            'triple': 0.1,
        }

    # 主增強函式：根據強度與機率，隨機應用 1~3 種增強方式
    def apply_mixed_augmentation(self, audio: np.ndarray, sr: int, 
                               intensity: float = 0.5, 
                               apply_prob: float = 0.6) -> np.ndarray:
        if np.random.random() > apply_prob:
            return audio  # 按照機率不進行增強

        adjusted_intensity = self._adjust_intensity_by_epoch(intensity)

        mix_type = np.random.choice(
            list(self.mix_strategies.keys()), 
            p=list(self.mix_strategies.values())
        )

        num_augmentations = {
            'single': 1, 'double': 2, 'triple': 3
        }[mix_type]

        # 根據策略權重隨機選擇策略
        strategy_names = list(self.strategies.keys())
        strategy_weights = [self.strategies[name]['weight'] for name in strategy_names]

        selected_strategies = np.random.choice(
            strategy_names, 
            size=min(num_augmentations, len(strategy_names)),
            replace=False,
            p=strategy_weights
        )

        # 執行所有選中的增強策略
        augmented_audio = audio.copy()
        for strategy in selected_strategies:
            augmented_audio = self._apply_single_strategy(
                augmented_audio, sr, strategy, adjusted_intensity
            )

        return augmented_audio

    # 調整增強強度，加上一些隨機性
    def _adjust_intensity_by_epoch(self, base_intensity: float) -> float:
        noise = np.random.normal(0, 0.05)
        return np.clip(base_intensity + noise, 0.05, 0.8)

    # 單一策略的應用實現
    def _apply_single_strategy(self, audio: np.ndarray, sr: int, 
                             strategy: str, intensity: float) -> np.ndarray:
        config = self.strategies[strategy]

        if strategy == 'add_noise':
            factor = np.random.uniform(config['min_factor'], config['max_factor']) * intensity
            noise = np.random.randn(len(audio)) * factor
            return audio + noise

        elif strategy == 'time_shift':
            shift_ratio = np.random.uniform(config['min_shift'], config['max_shift']) * intensity
            shift_samples = int(len(audio) * shift_ratio)
            if shift_samples > 0:
                shift = np.random.randint(-shift_samples, shift_samples)
                return np.roll(audio, shift)

        elif strategy == 'time_mask':
            mask_ratio = np.random.uniform(config['min_mask'], config['max_mask']) * intensity
            mask_length = int(len(audio) * mask_ratio)
            if mask_length > 0:
                start = np.random.randint(0, max(1, len(audio) - mask_length))
                audio_masked = audio.copy()
                audio_masked[start:start + mask_length] *= 0.1
                return audio_masked

        elif strategy == 'volume_change':
            factor = np.random.uniform(config['min_factor'], config['max_factor'])
            factor = 1.0 + (factor - 1.0) * intensity
            return audio * factor

        return audio

# 語音序列對序列資料集，用於訓練語音識別模型
from torch.utils.data import Dataset
import numpy as np

class SpeechSeq2SeqDataset(Dataset):

    def __init__(self, data_list: list, processor, sampling_rate: int = 16000, 
                 augment: bool = False, aug_intensity: float = 0.5):
        self.processor = processor
        self.sr = sampling_rate
        self.data = data_list  # 每筆為 dict: {'audio', 'sentence', 'file_id'}
        self.augment = augment
        self.aug_intensity = aug_intensity

        self.augmenter = AdvancedAudioAugmentation()

        # 統計資料增強狀況
        self.aug_stats = {
            'total_samples': 0,
            'augmented_samples': 0,
            'strategy_counts': {}
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio = item['audio']
        self.aug_stats['total_samples'] += 1

        if self.augment:
            original_audio = audio.copy()
            audio = self.augmenter.apply_mixed_augmentation(audio, self.sr, self.aug_intensity)

            if not np.array_equal(original_audio, audio):
                self.aug_stats['augmented_samples'] += 1

        return {
            "audio": audio,
            "sentence": item["sentence"],
            "file_id": item["file_id"]
        }

    def collate_fn(self, batch):
        audios = [item["audio"] for item in batch]
        sentences = [item["sentence"] for item in batch]

        feat = self.processor.feature_extractor(
            audios,
            sampling_rate=self.sr,
            return_tensors='pt',
            return_attention_mask=True
        )
        padded = self.processor.feature_extractor.pad(
            {
                'input_features': feat['input_features'],
                'attention_mask': feat['attention_mask']
            },
            padding=True,
            return_tensors='pt'
        )

        tok = self.processor.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        labels = tok['input_ids'].masked_fill(tok['attention_mask'].ne(1), -100)

        return {
            'input_features': padded['input_features'],
            'attention_mask': padded['attention_mask'],
            'labels': labels
        }

    def get_augmentation_stats(self):
        if self.aug_stats['total_samples'] > 0:
            aug_rate = self.aug_stats['augmented_samples'] / self.aug_stats['total_samples']
            return {
                'augmentation_rate': aug_rate,
                'total_samples_processed': self.aug_stats['total_samples'],
                'augmented_samples': self.aug_stats['augmented_samples']
            }
        return {}


# 命名實體識別資料集，用於指令式訓練語言模型
class NERDataset(Dataset):
    def __init__(self, tokenizer, texts, answers, system_prompt):
        self.texts = texts
        self.answers = answers
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer

        self.eos_id = torch.tensor([self.tokenizer.eos_token_id])

        self.input_datas = []
        for texts, answers in zip(texts, answers):
            self.input_datas.append(self.sft_format(texts, answers).values())
        
    # 格式化成訓練格式：包含 prompt、輸入與答案
    def sft_format(self, text, answer=None):
        prompt = f'### PROMPT\n{self.system_prompt}\n\n### INPUT\n{text}\n\n' 
        prompt = self.tokenizer(text=prompt, add_special_tokens=False, truncation=True, max_length=64000, return_tensors='pt')

        prompt_ids, prompt_attention_mask = prompt.input_ids[0], prompt.attention_mask[0]
        if answer is not None:
            answer = self.tokenizer(text='### OUTPUT\n' + answer, add_special_tokens=False, truncation=True, max_length=64000, return_tensors='pt')
            answer_ids = answer.input_ids[0]

            label_mask = torch.full_like(prompt_ids, -100)

            input_ids = torch.cat((prompt_ids, answer_ids, self.eos_id), dim=0)
            labels = torch.cat((label_mask, answer_ids, self.eos_id), dim=0)
            attention_mask = torch.full_like(input_ids, 1)

            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

        return {'input_ids': prompt_ids, 'attention_mask': prompt_attention_mask}
             
    def __len__(self):
        return len(self.input_datas)

    def __getitem__(self, idx):
        return self.input_datas[idx]

    # 對 batch 資料進行 padding
    def collate_fn(self, batch):
        x, mask, y = zip(*batch)

        input_ids = pad_sequence(x, padding_value=self.tokenizer.eos_token_id, batch_first=True)
        labels = pad_sequence(y, padding_value=-100, batch_first=True)
        attention_mask = pad_sequence(mask, padding_value=0, batch_first=True)

        return {
            'input_ids': input_ids,  
            'labels': labels,
            'attention_mask': attention_mask    
        }
