import os
import yaml
import random

from typing import Any, SupportsFloat

import imageio
import numpy as np

from PIL import Image

import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

from sub_modules.sub_framework import Front_Part


# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()




'''
export ALFWORLD_DATA=/workspace/check_envs/alfworld_data_storage/
export ALFWORLD_DATA=/home/sandman/check_envs/alfworld_data_storage/


export DISPLAY=:0.0

'''

'''
python test.py alfworld/configs/base_config.yaml > test_print.txt
'''

'''
import sys
sys.path.append('D:/myspace/blog/code/python/here/')
'''



if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))  

    # load config
    # config = generic.load_config()
    
    with open("configs/test_config.yaml") as confile:
        config = yaml.load(confile, yaml.SafeLoader)
    env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
    print("config type : ", type(config))
    print("config : \n", config)
    print("env_type type : ", type(env_type))
    print("env_type : \n", env_type)
    # exit()

    # setup environment
    print("Setting up the Environment")
    env = getattr(environment, env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

    # interact
    print("Reset Environment")
    obs, info = env.reset()  # infinte loading error
    print("obs : ", obs)

    # check image of environment
    
    initial_images = env.get_frames()
    print("type: ", type(env.get_frames()))
    Image.fromarray(np.squeeze(initial_images, axis=0)).save("temp/test_alfworld_env.png")

    # Train DQN
    # model = DQN("")
    
 
    
    images = []
    iter_num = 0
    while True:
        # img = env.get_frames()
        # images.append(img[0])
        
        # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
        admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
        random_actions = [np.random.choice(admissible_commands[0])]

        # step
        obs, scores, dones, infos = env.step(random_actions)
        print("obs : ", obs)
        # print("obs space : ", env.observation_space)
        # print("act space : ", env.action_space)
        print("scores : ", scores)
        print("dones : ", dones)
        print("infos : ", infos)
        print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))
        
        images.append(np.squeeze(env.get_frames(), axis=0))
        
        iter_num += 1
        if iter_num >= 100:
            break
        
        
    name = "test"
    gif_filename = os.path.join("logs/gifs", name + '.' + "gif")
    gif_config = {
        'loop':1,
        'duration': 1/20        
    }
    imageio.mimsave(gif_filename, images, format='gif', **gif_config)  
        
    
