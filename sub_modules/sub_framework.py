import os

import numpy as np
import torch

from hyperparameters import device

from sub_modules.sub_module_LM import Image_Captioning_Model, Behavior_LLM, SkillEmbedder


class Front_Part(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.task_prompt = "In this task, you have to analye the situation and present step by step procedure to solve the problem."

        # Front Part
        self.image_cap_model = Image_Captioning_Model()
        self.behavior_LLM = Behavior_LLM()
        self.skill_embedder = SkillEmbedder()
        
        # Output size : torch.Size([1, 10, 768])
        pass


    def forward(self,initial_obs, instruction):
        
        image_cap = self.image_cap_model(initial_obs)
        # print("image cap : ", image_cap)

        initial_input = image_cap + ". In this situation, I need to " + instruction

        skill_description_seq = self.behavior_LLM(self.task_prompt, initial_input)
        # print("skill des seq : ", skill_description_seq)
        
        skill_embedding_seq = self.skill_embedder(skill_description_seq)
        # print("skill emb seq : ", skill_embedding_seq)
        
        return skill_embedding_seq




if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))
    