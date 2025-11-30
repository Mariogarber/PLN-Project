import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import T5Tokenizer, T5ForConditionalGeneration

class QLoraConfig:
    def __init__(self, model_name, r, target_modules, lora_alpha=32, lora_dropout=0.1):
        # BitsAndBytes configuration
        self.bnb_config = self.load_bnb_config()
        # Q configuration
        self.model_name = model_name
        self.model_quantized = self.load_model_quantized()
        # Lora configuration
        self.r = r
        self.target_modules = target_modules
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_config = self.load_lora_config()

    def load_bnb_config(self):
        """Load BitsAndBytesConfig for 4-bit quantization"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        return bnb_config

    def load_model_quantized(self):
        model = T5ForConditionalGeneration.from_pretrained(
                            self.model_name,
                            quantization_config=self.bnb_config,
                            device_map="auto",
                            torch_dtype=torch.bfloat16
                        )
        model_quantized = prepare_model_for_kbit_training(model)
        return model_quantized

    def load_lora_config(self):
        """Load LoRA configuration"""
        lora_config = LoraConfig(
            r=self.r,
            target_modules=self.target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM,
            bias="none",
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        return lora_config
    
    def get_peft_model(self):
        """Get PEFT model with LoRA applied"""
        model_lora = get_peft_model(self.model_quantized, self.lora_config)
        print(f"LoRA model {self.model_name} loaded with the following configuration:")
        print(f" - Rank (r): {self.r}")
        print(f" - Target Modules: {self.target_modules}")
        print(f" - LoRA Alpha: {self.lora_alpha}")
        print(f" - LoRA Dropout: {self.lora_dropout}")
        return model_lora