import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from tools import Tools
from peft import LoraConfig, get_peft_model, PeftModel
from config import LoraArguments


class LoraModel(PeftModel):
    def __init__(self, config: Config, model):
        lora_args = LoraArguments()
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            task_type="CAUSAL_LM",
        )
        super().__init__(model, lora_config)
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("The causal language model config must define hidden_size for the value head")
        self.v_head = torch.nn.Linear(hidden_size, 1, bias=False).to(config.device)
        if lora_args.is_reload_trained_params:
            super().from_pretrained(model, config.save_lora_path)
            self.v_head.load_state_dict(torch.load(config.save_v_head_path))
        for name, module in self.named_modules():
            if 'lora_' in name:
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask, tools: Tools):
        res = super().forward(input_ids, attention_mask, output_hidden_states=True)
        values = self.v_head(res.hidden_states[0]).squeeze(-1)[:, :-1]
        values = tools.filter_mask(values)
        probs = tools.probs_from_logits(res.logits[:, :-1, :], input_ids[:, 1:])
        probs = tools.filter_mask(probs)
        return probs, values


# class ActorCriticLoraModel(torch.nn.Module):
#     def __init__(self, config: Config):
#         super().__init__()
#         model = AutoModelForCausalLM.from_pretrained(config.gpt_model, attn_implementation="eager", sliding_window=None).to(config.device).eval()
#         self.model = LoraModel(config, model)
#         self.tokenizer = AutoTokenizer.from_pretrained(config.gpt_model)

#     def forward(self, input_ids, attention_mask, tools: Tools):
#         probs, values = self.model(input_ids, attention_mask, tools)
#         return probs, values

#     @torch.no_grad()
#     def actor_generate(self, input_ids, attention_mask=None):
#         generated_ids = self.model.generate(input_ids, max_new_tokens=512, top_p=1.0,
#                                             num_beams=1,
#                                             do_sample=False)
#         response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         response_id = generated_ids[:, input_ids.shape[1]:]
#         return response, generated_ids, response_id


class ActorCriticLoraModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        base_model = AutoModelForCausalLM.from_pretrained(config.gpt_model, attn_implementation="eager", sliding_window=None)
        base_model = base_model.to(config.device)  # 确保基础模型在 config.device（如cuda:0）
        base_model.eval()  # 评估模式（生成阶段无需训练）
        
        self.model = LoraModel(config, base_model)
        self.model = self.model.to(config.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.gpt_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 步骤4：保存目标设备，供后续生成时复用（避免硬编码）
        self.device = config.device
        # 验证：打印模型最终设备（确保是 cuda:0，而非 cpu）

    def forward(self, input_ids, attention_mask, tools: Tools):
        # （可选）forward中也校验输入设备，避免后续训练时数据设备错误
        input_ids = input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        probs, values = self.model(input_ids, attention_mask, tools)
        return probs, values

    @torch.no_grad()
    def actor_generate(self, input_ids, attention_mask=None):
        # 步骤1：强制将输入数据转移到模型设备（双重保险，避免外部未转移）
        input_ids = input_ids.to(self.config.device)
        # 验证：打印输入数据设备（确保与模型设备一致）
        # print(f"actor_generate 输入设备: {input_ids.device}")
        # print(f"actor_generate 模型设备: {next(self.model.parameters()).device}")
        
        # 步骤2：模型生成（此时数据和模型已同设备）
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
        }
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask.to(self.config.device)
        generated_ids = self.model.generate(input_ids, **generation_kwargs)
        
        # 步骤3：解码与返回（generated_ids 已在目标设备，不影响解码）
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response_id = generated_ids[:, input_ids.shape[1]:]  # 截取生成的response部分
        return response, generated_ids, response_id
