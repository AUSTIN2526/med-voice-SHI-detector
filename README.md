# 🩺 Med-Voice-SHI-Detector

> 🏆 **Champion project of the [Codabench AI Challenge](https://www.codabench.org/competitions/4890/?secret_key=38d92718-cc4d-4907-9c65-c73419671268#/pages-tab) on medical voice de-identification**!

This project combines the power of two advanced models to create a cutting-edge system designed specifically for **de-identifying sensitive information in medical voice data**:

* 🤗 [nyrahealth/CrisperWhisper](https://huggingface.co/nyrahealth/CrisperWhisper)
* 🤗 [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base)

---

## 🔬 Project Overview

**Med-Voice-SHI-Detector** is designed for high-quality English medical voice transcription and de-identification. The system uses a dual-model architecture with **Whisper** for speech-to-text conversion and **Qwen3** for natural language understanding and entity recognition. It enables word-level timestamping and precise identification and masking of Sensitive Health Information (SHI).

### ✅ Key Features

* **Dual-model synergy**: Combines Whisper's accurate ASR with Qwen3’s advanced semantic understanding and NER capabilities.
* **Balance of privacy and usability**: De-identification protects privacy while preserving the utility of voice data.
* **Optimized for competition**: Tailored for challenging medical voice processing tasks, showcasing AI's potential in healthcare NLP.

📌 **Competition Link**: [Codabench AI Challenge](https://www.codabench.org/competitions/4890/?secret_key=38d92718-cc4d-4907-9c65-c73419671268#/pages-tab)

---

## ⚙️ Installation & Usage Guide

Using a virtual environment is recommended to avoid package dependency conflicts.

### 1. Environment Setup

#### 🔧 Install PyTorch

Select the command based on your CUDA version (example below is for CUDA 12.6):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Refer to the [official PyTorch site](https://pytorch.org/) for other CUDA versions.

#### 📦 Install Whisper

```bash
pip install -r requirement.txt
```

#### 📦 Install Qwen

```bash
pip install -r requirement_qwen.txt
```

---

### 2. Workflow

Execute the following steps in the project root directory:

#### 🗂️ Step 1: Data Format Conversion

```bash
python convert_trining_format.py
```

#### 🧠 Step 2: Fine-tune Whisper Model (ASR Task)

```bash
python finetune_moodel.py --task asr
```

#### 🧠 Step 3: Fine-tune Qwen Model (NER Task)

```bash
python finetune_moodel.py --task ner
```

#### 🔍 Step 4: Whisper Inference

```bash
python infer_whisper.py
```

#### 🔍 Step 5: Qwen Inference

```bash
python infer_qwen.py
```

#### 📤 Step 6: Output Formatting

```bash
python convert_answer.py
```

> The final output is ready for competition submission or further analysis.

---

## 🧠 Technical Highlights

### 🎙️ A. Whisper: ASR and Data Preprocessing

* **Contextual Completion**: Recursive inference improves transcription completeness and accuracy.
* **Data Augmentation**: Techniques like background noise, time scaling, and volume variation improve model generalization.

### 🧾 B. Qwen: Entity Recognition and De-identification

1. **Data Cleaning**: Removes noise to enhance data quality.
2. **Initial Semantic Tagging**: Qwen3-14B-Base provides initial SHI annotation.
3. **Advanced Fine-Tuning**: Iterative training boosts NER precision.
4. **Data Augmentation Techniques**:

   * **NEFtune**: Noise Embedding Fine-tuning injects minor noise into embedding layers to enhance semantic generalization and reduce overfitting.
   * **Sliding Window**: Long-text data is segmented using a sliding window approach to ensure the model captures rich context.
   * **Invalid Data Filtering**: Removes irrelevant, low-quality, or duplicate samples during preprocessing to enhance corpus purity and avoid misleading patterns.
5. **Inference Optimization**:

   * Combines sliding window with multi-pass prediction to improve SHI recognition coverage and stability.
