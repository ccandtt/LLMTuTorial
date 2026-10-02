from pathlib import Path

import torch
from config import Config
from torch.utils.data import DataLoader
from torch.optim import Adam
from actor_critic_model import ActorCriticLoraModel
from reward_model import RewardModel
from data_load import CustomDataset
from reference_model import ReferenceModel
from ppo import PPO
from tools import Tools


class TrainPpo:
    def __init__(self):
        self.config = Config()
        # 演员和评论家模型
        self.actor_critic_model = ActorCriticLoraModel(self.config).to(self.config.device)
        self.tokenizer = self.actor_critic_model.tokenizer
        # 获得演员和评论家模型优化器, 这里使用的是lora, 不优化全量数据
        self.actor_critic_opt = Adam(self.actor_critic_model.parameters(), lr=self.config.lr)
        # 参考模型
        self.reference_model = ReferenceModel(self.config).to(self.config.device)
        # 奖励模型
        self.reward_model = RewardModel(self.config).to(self.config.device)
        # 训练数据
        dataset = CustomDataset(self.config.data_path, self.tokenizer)
        
        self.data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True,
                                      collate_fn=dataset.collate_fn)
        self.ppo = PPO(self.actor_critic_model, self.config, self.actor_critic_opt)

    def train_ppo(self):
        self.save_model()
        completed_steps = 0
        for epoch in range(self.config.epochs):
            for batch_data in self.data_loader:
                # 修复：如果batch_data是元组，直接使用索引访问，而不是当作字典处理
                # 假设batch_data的第一个元素是输入数据
                batch_tensor, batch_attention_mask = batch_data
                batch_tensor = batch_tensor.to(self.config.device)
                batch_attention_mask = batch_attention_mask.to(self.config.device)

                # 获得演员模型生成的结果，并显式传入 attention mask。
                prompt_generate, prompt_generate_ids, generate_ids = self.actor_critic_model.actor_generate(
                    batch_tensor, batch_attention_mask)

                # EOS is also Qwen's pad token. Keep the first EOS in the
                # response so a short generation never produces an empty mask.
                eos_mask = generate_ids.eq(self.tokenizer.eos_token_id)
                eos_count = eos_mask.cumsum(dim=1)
                first_eos = eos_mask & eos_count.eq(1)
                response_mask = (~eos_mask) | first_eos
                attention_mask = torch.cat((batch_attention_mask, response_mask), dim=1)
                generate_ids_mask = response_mask[:, :-1]
                response_shape = generate_ids.shape[1] - 1
                if response_shape < 1 or not generate_ids_mask.any(dim=1).all():
                    continue
                tools = Tools(response_shape, generate_ids_mask)
                
                pure_generate = [
                    one.split("assistant\n", 1)[1] if "assistant\n" in one else one
                    for one in prompt_generate
                ]
                reward = self.reward_model(pure_generate)
                prob_refs = self.reference_model(prompt_generate_ids, attention_mask, tools)
                self.ppo.train(prompt_generate_ids, attention_mask, prob_refs, reward, tools)
                completed_steps += 1
                if self.config.max_train_steps and completed_steps >= self.config.max_train_steps:
                    break
            if self.config.max_train_steps and completed_steps >= self.config.max_train_steps:
                break

        self.save_model()
    

    def save_model(self):
        Path(self.config.save_lora_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.save_v_head_path).parent.mkdir(parents=True, exist_ok=True)
        # 保存lora参数
        self.actor_critic_model.model.save_pretrained(self.config.save_lora_path, safe_serialization=False)
        # 保存价值模型参数
        torch.save(self.actor_critic_model.model.v_head.state_dict(), self.config.save_v_head_path)


if __name__ == '__main__':
    train_ppo = TrainPpo()
    train_ppo.train_ppo()