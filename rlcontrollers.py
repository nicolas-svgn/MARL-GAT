from env import RLController, SumoEnv, CustomEnv

import random

import os
import time
import argparse
import itertools
from datetime import timedelta
import numpy as np
import torch as T
import supersuit as ss
from stable_baselines3.common.env_checker import check_env
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

def main():
    sumo_env = RLController(gui=False, log=False, rnd=(True, True))
    #for tl_id in HYPER_PARAMS['tl_ids']:
        #print(tl_id)
    print(sumo_env.tl_ids)
    print(f"action space: {sumo_env.action_space_n}")
    print(f"observation space: {sumo_env.observation_space_n}")
    sumo_env.reset()
    tls = sumo_env.tl_ids
    actions = [random.randint(0,3) for tl_id in tls]
    actions = dict(zip(tls,actions))

    for _ in range(100):
        actions = [random.randint(0,3) for tl_id in tls]
        actions = dict(zip(tls,actions))
        print(actions)
        sumo_env.step(actions)

    for tl_id in tls:
        print(tl_id)
        print("--------------------")
        print(sumo_env.obs(tl_id))
        print("--------------------")


    """for _ in range(1000):
        random_number = random.randint(0, 3)
        sumo_env.step(random_number)
    for tl_id in sumo_env.tl_ids:
        print(tl_id)
        print("--------------------")
        sumo_env.print_dtse(sumo_env.get_dtse(tl_id))
        print("--------------------")"""

    """for step in range(2):
         print("----------------------------")
         print(step)
         actions = [random.randint(0,3) for tl_id in tls]
         actions = dict(zip(tls,actions))
         sumo_env.step(actions)

         print("----------------------------")"""

"""def main2():
        env = make_env(
            env=CustomEnvWrapper(CustomEnv("train")),
            repeat=0,
            max_episode_steps=1000,
            n_env=1
        )

        env.envs[0].unwrapped.custom_env.sumo_env.set_tl_id('gneJ10')
        print(env.envs[0].unwrapped.custom_env.sumo_env.next_tl_id)"""

"""def main3():
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tls = args.tl_ids
    agents = []
    env = make_env(
            env=CustomEnvWrapper(CustomEnv("train")),
            repeat=args.repeat,
            max_episode_steps=args.max_episode_steps,
            n_env=args.n_env
        )
    
    obses = env.reset()
    print(obses)
    print(env.envs[0].unwrapped.custom_env.sumo_env.next_tl_id)
    for step in range(100):
        #print("STEP:{}".format(step))
        for tl_id in tls:
            #print("TL ID:{}".format(tl_id))
            action = random.randint(0, 3)
            env.envs[0].unwrapped.custom_env.sumo_env.set_tl_id(tl_id)
            env.step(action)
            #print(env.envs[0].unwrapped.custom_env.sumo_env.next_tl_id)

    for tl_id in tls:
         print()
        #print("TL ID:{}".format(tl_id))
        #print(env.envs[0].unwrapped.custom_env.sumo_env.get_dtse(tl_id))"""

"""def main4():
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tls = args.tl_ids
    agents = []
    n_env = args.n_env
    env = make_env(
            env=CustomEnvWrapper(CustomEnv("train")),
            repeat=args.repeat,
            max_episode_steps=args.max_episode_steps,
            n_env=args.n_env
        )
    
    for tl_id in args.tl_ids:
            print('Current TL : ',tl_id)
            new_agent = getattr(Agents, args.algo)(
                n_env=args.n_env,
                lr=args.lr,
                gamma=args.gamma,
                epsilon_start=args.eps_start,
                epsilon_min=args.eps_min,
                epsilon_decay=args.eps_dec,
                epsilon_exp_decay=args.eps_dec_exp,
                nn_conf_func=network_config2,
                input_dim=env.observation_space,
                output_dim=env.action_space.n,
                batch_size=args.bs,
                min_buffer_size=args.min_mem,
                buffer_size=args.max_mem,
                update_target_frequency=args.target_update_freq,
                target_soft_update=args.target_soft_update,
                target_soft_update_tau=args.target_soft_update_tau,
                save_frequency=args.save_freq,
                log_frequency=args.log_freq,
                save_dir=args.save_dir,
                log_dir=args.log_dir,
                load=args.load,
                algo=args.algo,
                gpu=args.gpu,
                tl_id = tl_id
            )
            agents.append(new_agent)
            print('done')

    for agent in agents:
        agent.load_model()

    min_resume_step = agents[0].resume_step
    for agent in agents:
        if agent.resume_step < min_resume_step:
            min_resume_step = agent.resume_step
    resume_step = min_resume_step

    print()
    print("TRAIN")
    print()
    print(args.algo)
    print()
    print(agent.online_network)
    print()
    [print(arg, "=", getattr(args, arg)) for arg in vars(args)]

    min_buffer_size = args.min_mem
    max_total_steps = args.max_total_steps

    print()
    print("Initialize Replay Memory Buffer")

    print(tls)
    actions = [env.action_space.sample() for agent in agents]
    print(actions)

    print(dict(zip(tls, actions))
)

    env.reset()

    print(env.envs[0].unwrapped.custom_env.sumo_env.get_dtse_shape())
    deep_q_network = Networks.DeepQNetwork(device, args.lr, network_config2, env.observation_space, 8)

    for _ in range(100):
        actions = [env.action_space.sample() for agent in agents]
        print(dict(zip(tls,actions)))
        env.envs[0].unwrapped.custom_env.sumo_env.step(dict(zip(tls, actions)))

    for tl in tls:
        obses = env.envs[0].unwrapped.custom_env.sumo_env.get_dtse_array(tl)
        input_tensor = T.tensor(obses, dtype=T.float32).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        print(tl)
        print(deep_q_network.forward(input_tensor))"""


    
"""for step in range(1000):
        actions = [env.action_space.sample() for agent in agents]
        env.envs[0].unwrapped.custom_env.sumo_env.step(dict(zip(tls, actions)))

    print("--------------------------------------------------")

    for tl in tls:
        print(tl)
        print(env.envs[0].unwrapped.custom_env.sumo_env.get_dtse(tl))"""
        

"""     tls_dtse = [env.envs[0].unwrapped.custom_env.sumo_env.get_dtse(tl_id) for tl_id in tls]
    for agent, tl_dtse in zip(agents, tls_dtse):
         print("Agent TL ID:")
         print(agent.tl_id)
         print("AGENT DTSE")
         print(tl_dtse)

         actions = [env.action_space.sample() for agent in agents]  """

    
         



"""     for t in range(min_buffer_size // n_env):
            if t >= (min_buffer_size // n_env) - resume_step:
                actions = [agent.choose_actions(tl_dtse) for agent, tl_dtse in zip(agents, tls_dtse)]
            else:
                actions = [env.action_space.sample() for agent in agents]
                print(actions) """

def main5():
    env = CustomEnv()
    tls = env.sumo_env.tl_ids
    print(tls)
    print(env.action_space)
    print(env.observation_space)
    print(env.metadata)
    actions = [random.randint(0,3) for tl_id in tls]
    actions = dict(zip(tls,actions))
    print(actions)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    #obs, _ = env.reset()
    #print(obs)
    

    """for _ in range(2):
        actions = [random.randint(0,3) for tl_id in tls]
        #actions = dict(zip(tls,actions))
        print('yo')
        print(actions)
        print('wesh')
        obs, rew, terminated, truncated, infos = env.step(actions)
        print('--------------------')
        print(obs)
        print('--------------------')
        print(rew)
        print('--------------------')
        print(terminated)
        print('--------------------')
        print(truncated)
        print('--------------------')
        print(infos)
        print('--------------------')"""


        
        

        
if __name__ == "__main__":
    """parser = argparse.ArgumentParser(description="RLCONTROLLERS")
    str2bool = (lambda v: v.lower() in ("yes", "y", "true", "t", "1"))
    parser.add_argument('-gpu', type=str, default=HYPER_PARAMS["gpu"], help='GPU #')
    parser.add_argument('-n_env', type=int, default=HYPER_PARAMS["n_env"], help='Multi-processing environments')
    parser.add_argument('-lr', type=float, default=HYPER_PARAMS["lr"], help='Learning rate')
    parser.add_argument('-gamma', type=float, default=HYPER_PARAMS["gamma"], help='Discount factor')
    parser.add_argument('-eps_start', type=float, default=HYPER_PARAMS["eps_start"], help='Epsilon start')
    parser.add_argument('-eps_min', type=float, default=HYPER_PARAMS["eps_min"], help='Epsilon min')
    parser.add_argument('-eps_dec', type=float, default=HYPER_PARAMS["eps_dec"], help='Epsilon decay')
    parser.add_argument('-eps_dec_exp', type=str2bool, default=HYPER_PARAMS["eps_dec_exp"], help='Epsilon exponential decay')
    parser.add_argument('-bs', type=int, default=HYPER_PARAMS["bs"], help='Batch size')
    parser.add_argument('-min_mem', type=int, default=HYPER_PARAMS["min_mem"], help='Replay memory buffer min size')
    parser.add_argument('-max_mem', type=int, default=HYPER_PARAMS["max_mem"], help='Replay memory buffer max size')
    parser.add_argument('-target_update_freq', type=int, default=HYPER_PARAMS["target_update_freq"], help='Target network update frequency')
    parser.add_argument('-target_soft_update', type=str2bool, default=HYPER_PARAMS["target_soft_update"], help='Target network soft update')
    parser.add_argument('-target_soft_update_tau', type=float, default=HYPER_PARAMS["target_soft_update_tau"], help='Target network soft update tau rate')
    parser.add_argument('-save_freq', type=int, default=HYPER_PARAMS["save_freq"], help='Save frequency')
    parser.add_argument('-log_freq', type=int, default=HYPER_PARAMS["log_freq"], help='Log frequency')
    parser.add_argument('-save_dir', type=str, default=HYPER_PARAMS["save_dir"], help='Save directory')
    parser.add_argument('-log_dir', type=str, default=HYPER_PARAMS["log_dir"], help='Log directory')
    parser.add_argument('-load', type=str2bool, default=HYPER_PARAMS["load"], help='Load model')
    parser.add_argument('-repeat', type=int, default=HYPER_PARAMS["repeat"], help='Steps repeat action')
    parser.add_argument('-max_episode_steps', type=int, default=HYPER_PARAMS["max_episode_steps"], help='Episode step limit')
    parser.add_argument('-max_total_steps', type=int, default=HYPER_PARAMS["max_total_steps"], help='Max total training steps')
    parser.add_argument('-algo', type=str, default=HYPER_PARAMS["algo"],
                        help='DQNAgent ' +
                             'DoubleDQNAgent ' +
                             'DuelingDoubleDQNAgent ' +
                             'PerDuelingDoubleDQNAgent'
                        )
    parser.add_argument('-tl_ids',default=HYPER_PARAMS["tl_ids"], help = 'Traffic light ids to train')"""
    #print(parser.parse_args().tl_ids)
    #main()
    #main2()
    #main3()
    #main4()
    main5()