import os
from pathlib import Path

import torch
from typing import List
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent


class Config:
    # Set these variables to local model directories or Hugging Face repo IDs.
    # The old /data/xxx paths were placeholders and were interpreted as invalid
    # Hub repo IDs when the directories did not exist.
    Sentiment_model = os.getenv(
        "PPO_SENTIMENT_MODEL",
        "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment",
    )
    gpt_model = os.getenv("PPO_GPT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    data_path = os.getenv(
        "PPO_DATA_PATH", str(PROJECT_ROOT / "data" / "train_data.json")
    )
    save_lora_path = os.getenv(
        "PPO_SAVE_LORA_PATH", str(PROJECT_ROOT / "save_model" / "save_lora")
    )
    save_v_head_path = os.getenv(
        "PPO_SAVE_V_HEAD_PATH",
        str(PROJECT_ROOT / "save_model" / "v_head" / "pytorch_model.bin"),
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = int(os.getenv("PPO_BATCH_SIZE", "2"))
    epochs = int(os.getenv("PPO_EPOCHS", "10"))
    max_train_steps = int(os.getenv("PPO_MAX_TRAIN_STEPS", "0"))
    max_new_tokens = int(os.getenv("PPO_MAX_NEW_TOKENS", "512"))
    lr = float(os.getenv("PPO_LR", "0.001"))
    # PPO 参数 ############################
    ppo_epochs = int(os.getenv("PPO_PPO_EPOCHS", "3"))
    kl_ctl_value = 0.2
    gamma = 1.0  # 用于优势计算的折扣因子。控制未来奖励的重要性。
    lam = 0.95  # 用于优势计算的Lambda参数。它用于控制对未来奖励的考虑程度，结合时间差异方法。
    cliprange_value = 0.2  # 损失计算中值函数的裁剪范围。裁剪可以防止极端值对训练过程的负面影响。
    cliprange = 0.2  # PPO策略梯度损失中的裁剪范围。这个裁剪范围用于限制策略更新的步长，从而保持训练的稳定性。
    vf_coef = 0.1


@dataclass
class LoraArguments:
    lora_r: int = 2
    lora_alpha: int = 8
    lora_dropout: float = 0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['k_proj',  'v_proj']
    )
    # lora_target_modules = None
    lora_weight_path: str = ""
    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    is_reload_trained_params = False  # 是否接着上次训练模型继续训练
