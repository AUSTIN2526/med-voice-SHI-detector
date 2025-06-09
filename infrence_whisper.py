import os
import json
import argparse
import librosa
import torch
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")


class SpeechTranscriber:
    def __init__(self, model_id="nyrahealth/CrisperWhisper", lora_path=None, language="en", device=None, dtype=None, split_threshold=0.12):
        """
        初始化語音轉換模型與處理器。
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.model_id = model_id
        self.lora_path = lora_path
        self.language = language
        self.split_threshold = split_threshold
        self._init_model_and_processor()
        self._build_pipeline()

    def _init_model_and_processor(self):
        """
        初始化模型與 Whisper 處理器，若指定 LoRA 模型則合併。
        """
        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=self.dtype,
            attn_implementation="eager",
        )
        if self.lora_path:
            peft_model = PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                device_map="auto",
                torch_dtype=self.dtype,
            )
            peft_model.merge_and_unload()
            self.model = peft_model.eval()
        else:
            self.model = base_model.eval()
        self.processor = WhisperProcessor.from_pretrained(self.model_id)

    def _build_pipeline(self):
        """
        建立 Transformers 語音辨識 Pipeline。
        """
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=1,
            return_timestamps="word",
            torch_dtype=self.dtype,
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )

    def transcribe(self, wav_path):
        """
        對指定音訊檔案進行語音轉換，並回傳轉換結果與時間區段。
        """
        audio, sr = librosa.load(wav_path, sr=self.processor.feature_extractor.sampling_rate, mono=True)
        sample = {"array": audio, "sampling_rate": sr}
        result = self.asr_pipe(sample)

        text = result.get("text", "").strip()
        print(f"[TEXT] {text}")

        chunks = result.get("chunks", [])
        adjusted_chunks = self.adjust_pauses_for_hf_pipeline_output(chunks, split_threshold=self.split_threshold)

        return {
            "text": text,
            "words": [
                {
                    "id": idx,
                    "start": float(ch["timestamp"][0]) if ch["timestamp"][0] is not None else None,
                    "end": float(ch["timestamp"][1]) if ch["timestamp"][1] is not None else None,
                    "text": ch.get("text", ""),
                }
                for idx, ch in enumerate(adjusted_chunks)
            ],
        }

    @staticmethod
    def adjust_pauses_for_hf_pipeline_output(chunks, split_threshold=0.12):
        """
        根據詞與詞之間的時間間隔進行微調，避免不自然的停頓。
        """
        if not chunks:
            return chunks
        adjusted_chunks = [c.copy() for c in chunks]
        for i in range(len(adjusted_chunks) - 1):
            cur = adjusted_chunks[i]
            nxt = adjusted_chunks[i + 1]
            cs, ce = cur["timestamp"]
            ns, ne = nxt["timestamp"]
            pause = ns - ce
            if pause > 0:
                distribute = split_threshold / 2 if pause > split_threshold else pause / 2
                cur["timestamp"] = (cs, ce + distribute)
                nxt["timestamp"] = (ns - distribute, ne)
        return adjusted_chunks


def batch_transcribe(input_dir, output_dir, transcriber, audio_exts=(".wav", ".mp3", ".flac")):
    """
    批次處理指定資料夾內的音訊檔案並儲存為 JSON。
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(audio_exts)]
    total_files = len(files)
    success_count = 0
    fail_count = 0

    print(f"[INFO] Found {total_files} audio files in '{input_dir}'\n")

    for idx, fname in enumerate(files, start=1):
        input_path = os.path.join(input_dir, fname)
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(output_dir, f"{base}.json")

        print(f"[INFO] ({idx}/{total_files}) Processing: {fname}")
        try:
            result = transcriber.transcribe(input_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Saved transcription to: {out_path}\n")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to transcribe '{fname}': {str(e)}\n")
            fail_count += 1

    print("=" * 50)
    print(f"[SUMMARY] Completed transcription.")
    print(f"[SUMMARY] Successful: {success_count} | Failed: {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch transcribe audio files using CrisperWhisper")

    parser.add_argument("--input_dir", type=str, default="aicup_data/test/audio", help="Directory containing input audio files")
    parser.add_argument("--output_dir", type=str, default="asr_result", help="Directory to save transcription results")
    parser.add_argument("--language", type=str, default="en", help="Language to force the model to transcribe")
    parser.add_argument("--model_id", type=str, default="nyrahealth/CrisperWhisper", help="HuggingFace model ID")
    parser.add_argument("--lora_path", type=str, default="all_adapter/whisper_adapter", help="Path to the LoRA adapter model")
    parser.add_argument("--split_threshold", type=float, default=0.12, help="Threshold to adjust pauses between word chunks")
    parser.add_argument("--audio_exts", type=str, nargs="+", default=[".wav", ".mp3", ".flac"], help="Allowed audio file extensions")

    args = parser.parse_args()

    transcriber = SpeechTranscriber(
        model_id=args.model_id,
        lora_path=args.lora_path,
        language=args.language,
        split_threshold=args.split_threshold
    )

    batch_transcribe(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        transcriber=transcriber,
        audio_exts=tuple(args.audio_exts)
    )
