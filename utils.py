import os
import yaml
import json

import imageio
import numpy as np

from PIL import Image
import PIL.ImageDraw as ImageDraw

import alfworld.agents.environment as environment



'''
export ALFWORLD_DATA=/workspace/check_envs/alfworld_data_storage/
'''


def get_alfred_expert_data(datasize:int=10):
    with open("configs/test_config.yaml") as confile:
        config = yaml.load(confile, yaml.SafeLoader)
    env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
    env = getattr(environment, env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)
    
    # run episode
    print("Env is Set")

    obs, info = env.reset()
    initial_info = info.copy()
    initial_obs = obs[0]
    print("Initial Info : ", info)
    print("Initial Obs: ", obs[0])
    
    observations = []
    actions = []
    next_observations = []
    
    timestep = 0        
    images = []
    is_done=False
    while not is_done:
        # get image of each step
        img = env.get_frames()
        images.append(img[0])
        
        observations.append(obs[0])  # get current observation data
        
        expert_actions = list(info['expert_plan'][0])  # get expert action from 'info'
        obs, scores, dones, infos = env.step(expert_actions[0])
        
        actions.append(expert_actions[0])  # get expert action
        next_observations.append(obs[0])  # get observation after taking the action
        
        print("infos : ", infos)                
        print("Action: {}, Obs: {}".format(expert_actions[0], obs[0]))
        print("dones : ", dones[0])
        
        
        timestep += 1
        is_done = dones[0]
        
    episode = dict(
        initial_info = initial_info,
        initial_observation = initial_obs,
        observations = observations,
        actions=actions,
        next_observation=next_observations                
    )
    print(timestep, len(observations), len(actions), len(next_observations))
    
    assert timestep == len(observations)
    assert timestep == len(actions)
    assert timestep == len(next_observations)
    
    save_as_json(episode)
    
    name = "test"
    gif_filename = os.path.join("logs/gifs", name + '.' + "gif")
    gif_config = {
        'loop':1,
        'duration': 1/60        
    }
    imageio.mimsave(gif_filename, images, format='gif', **gif_config)  
    
    pass

    





def save_as_json(json_data, filepath:str="data/expert_data.json"):
    with open(filepath, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    return




if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))  
    get_alfred_expert_data()