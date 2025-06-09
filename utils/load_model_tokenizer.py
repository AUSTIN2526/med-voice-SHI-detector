# 匯入必要的套件
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers.modeling_utils import unwrap_model

# 初始化 Whisper 模型並加載 LoRA 設定
def initialize_whisper_with_lora(
    model_name='openai/whisper-large-v3',
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj'], 
    freeze_encoder_modules=None,
    language=None,
    torch_dtype=torch.float16
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 設定 4-bit 量化參數
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

    # 載入 Whisper 語音轉錄模型
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        device_map='auto' if torch.cuda.is_available() else None,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        use_cache=False
    )

    # 建立 LoRA 配置
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        target_modules=target_modules
    )

    # 準備模型支援量化訓練
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    # 加入 LoRA 模型
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    model.to(device)

    # 選擇性凍結 Encoder 的某些 LORA 層
    if freeze_encoder_modules:
        for name, param in model.named_parameters():
            if 'encoder' in name and any(key in name for key in freeze_encoder_modules):
                param.requires_grad = False

    # 載入處理器（語音輸入用）
    processor = AutoProcessor.from_pretrained(
        model_name,
        predict_timestamps=False,
        task="transcribe",
        language=language
    )

    return model, processor, device, torch_dtype

# 啟用 NeFTune
def activate_neftune(model, neftune_noise_alpha=5):
    def neftune_post_forward_hook(module, input, output):
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2), device=output.device, dtype=output.dtype)
            mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
        return output

    # 取得輸入嵌入層
    def get_embeddings_layer(model):
        unwrapped = unwrap_model(model)
        if isinstance(unwrapped, PeftModel):
            base = unwrapped.base_model.model
        elif hasattr(unwrapped, "model"):
            base = unwrapped.model
        else:
            base = unwrapped

        if hasattr(base, "get_input_embeddings"):
            return base.get_input_embeddings()
        else:
            raise AttributeError("model not found embedding layer")

    # 註冊 hook 到 Embedding layer
    embeddings = get_embeddings_layer(model)
    embeddings.neftune_noise_alpha = neftune_noise_alpha
    embeddings.register_forward_hook(neftune_post_forward_hook)

    return model

def initialize_qwen_with_lora(
    model_name="Qwen/Qwen3-8B-Base",
    enable_neftune=True,
    torch_dtype=torch.bfloat16,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    eos_token_id = 151643
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype
    )

    # 載入 tokenizer 並手動設定 EOS token（避免某些模型未自帶）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.eos_token_id = eos_token_id

    # 載入語言模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map='auto',
        use_cache=False,
        quantization_config=bnb_config
    )

    # 建立 LoRA 設定
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM"
    )

    # 準備模型支援 LoRA 訓練
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)

    # 啟用 NeFTune（若指定）
    if enable_neftune:
        model = activate_neftune(model)
        print('Enable neftune')

    return model, tokenizer, device, torch_dtype
