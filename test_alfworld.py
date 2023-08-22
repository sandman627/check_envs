import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# interact
print("starting reset")
obs, info = env.reset()
print("Infos : ", info)
# exit()
while True:
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    expert_plan = list(info['expert_plan'])
    
    random_actions = [np.random.choice(admissible_commands[0])]
    expert_action = expert_plan[0]
    print(random_actions)
    print(expert_action)
    exit()

    # step
    obs, scores, dones, infos = env.step(random_actions)
    
    
    # print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))
    print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))
    print(f"scores: {scores}, dones: {dones}")