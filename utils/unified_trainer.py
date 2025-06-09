import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm  # 進度條顯示

class UnifiedLoRATrainer:
    def __init__(
        self,
        model,
        tokenizer=None,
        train_loader=None,
        valid_loader=None,
        epochs=12,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        scheduler_type='cosine',
        cosine_cycles=1,
        early_stopping=3,
        adapter_path='output_lora_adapter',
        enable_aug=False,
        enable_spike_detection=False
    ):
        # 初始化參數與模型
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.adapter_path = adapter_path
        self.enable_aug = enable_aug
        self.enable_spike_detection = enable_spike_detection
        self.early_stopping = early_stopping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 計算總訓練步數與 warmup 步數
        total_steps = len(train_loader) * epochs
        warmup_steps = int(warmup_ratio * total_steps)

        # 建立優化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # 選擇學習率排程策略
        if scheduler_type == 'cosine':
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps, num_cycles=cosine_cycles
            )
        elif scheduler_type == 'cosine_no_restart':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps
            )

        # 設定資料增強（augmentation）強度的排程
        self.aug_schedule = [
            0.0, 0.15, 0.25, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0, 0.0
        ]
        self.aug_reduction_factor = 0.8
        self.val_loss_history = []
        self.aug_history = []

    # 根據驗證損失調整資料增強強度
    def _adjust_aug(self, epoch):
        base_intensity = self.aug_schedule[min(epoch, len(self.aug_schedule)-1)]
        if len(self.val_loss_history) >= 2 and self.val_loss_history[-1] > self.val_loss_history[-2]:
            return base_intensity * self.aug_reduction_factor
        return base_intensity

    # 單一訓練 epoch 的流程
    def train_epoch(self, epoch, spike_threshold=0.1):
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')

        # 若啟用資料增強並支援 aug_intensity，調整增強強度
        if self.enable_aug and hasattr(self.train_loader.dataset, 'aug_intensity'):
            intensity = self._adjust_aug(epoch)
            self.train_loader.dataset.aug_intensity = intensity
            self.train_loader.dataset.augment = intensity > 0
            self.aug_history.append(intensity)

        for idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss

            # 若啟用異常偵測（spike detection）
            if self.enable_spike_detection and loss.item() > spike_threshold / (epoch + 1e-12) and epoch != 0:
                print(f"\n>>> Spike detected at epoch {epoch} batch {idx}, loss {loss.item():.3f}")
                if self.tokenizer:
                    decoded = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                    for text in decoded:
                        print(text)

            # 梯度反向傳播與優化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

        return train_loss / len(self.train_loader)

    # 驗證流程
    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        pbar = tqdm(self.valid_loader, desc=f'Valid Epoch {epoch}', leave=False)
        with torch.no_grad():
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                val_loss += outputs.loss.item()
                pbar.set_postfix({'loss': f'{outputs.loss.item():.4f}'})
        return val_loss / len(self.valid_loader)

    # 訓練整體流程
    def train(self):
        best_loss = float('inf')
        stop_count = 0
        history = {'train_loss': [], 'val_loss': [], 'aug_intensity': []}

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            if self.enable_aug:
                history['aug_intensity'].append(
                    self.aug_history[-1] if self.aug_history else 0.0
                )

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 模型保存與早停（early stopping）策略
            if val_loss < best_loss:
                best_loss = val_loss
                stop_count = 0
                self.model.save_pretrained(self.adapter_path)
                print(f"✓ Model saved (val loss {val_loss:.4f})")
            else:
                stop_count += 1
                print(f"✗ No improvement ({stop_count}/{self.early_stopping})")
                if stop_count >= self.early_stopping:
                    print("Early stopping triggered.")
                    break

            self.val_loss_history.append(val_loss)

        self.plot_history(history)
        return history

    # 畫出訓練歷程圖
    def plot_history(self, history):
        fig, ax = plt.subplots(1, 2 if self.enable_aug else 1, figsize=(12, 5))
        if not isinstance(ax, (list, np.ndarray)):
            ax = [ax]

        # 損失曲線圖
        ax[0].plot(history['train_loss'], label='Train Loss')
        ax[0].plot(history['val_loss'], label='Val Loss')
        ax[0].set_title('Loss Curve')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True)

        # 資料增強強度變化圖
        if self.enable_aug:
            ax[1].plot(history['aug_intensity'], label='Aug Intensity', color='green')
            ax[1].set_title('Augmentation Schedule')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Intensity')
            ax[1].legend()
            ax[1].grid(True)
            ax[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()
