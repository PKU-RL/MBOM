from policy.MBOM import MBOM
from baselines.PPO import PPO, PPO_Buffer
from utils.Logger import Logger
from utils.rl_utils import collect_trajectory
import os
import random
import torch
import numpy as np
import multiprocessing as mp

def tester(args, logger, env, env_model, confs, **kwargs):
    # '''set seed'''
    # random.seed(logger.seed)
    # torch.manual_seed(logger.seed)
    # np.random.seed(logger.seed)
    '''get shooter model file list'''
    shooters_path = "./logs/Football_Penalty_Kick/Shooter/layer{}".format(args.test_mode)
    shooter_model_file_list = []
    for root, dirs, files in os.walk(shooters_path):
        for f in files:
            shooter_model_file_list.append(os.path.join(root, f))
    '''get goalkeeper model file'''
    goalkeeper_ckp = "./logs/Football_Penalty_Kick/Goalkeeper/trueprob_rnn_football_ppo_vs_MBOM/MBOM_football_iter50000.ckp"

    res_l = []
    process_count = 0
    pool = mp.Pool(processes=args.test_mp)
    for shooter_id, file in enumerate(shooter_model_file_list):
        process_count += 1
        res = pool.apply_async(test_worker, (args, logger.root_dir, shooter_id, file, goalkeeper_ckp, logger.seed, env, env_model, confs[0], confs[1]))
        res_l.append(res)
        if process_count == args.test_mp:
            process_count = 0
            pool.close()
            pool.join()
            pool = mp.Pool(processes=args.test_mp)
    pool.close()
    pool.join()

def test_worker(args, root_dir, shooter_id, shooter_file, goalkeeper_ckp, seed, env, env_model, shooter_conf=None, goalkeeper_conf=None):
    '''set seed'''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    '''set logger'''
    cur_logger = Logger(root_dir, "rank", shooter_id)
    '''prepare env'''
    #env = football_env()
    #env_model = football_env_model(args.device)
    '''prepare agents'''
    if args.test_mode == 0 or args.test_mode == 1:
        agent1 = PPO.load_model(shooter_file, args, cur_logger, args.device)
    elif args.test_mode == 2:
        agent1 = MBOM.load_model(shooter_file, args, cur_logger, args.device, env_model=None)
    else:
        raise NameError
    agent2 = MBOM.load_model(goalkeeper_ckp, args, cur_logger, device=args.device, env_model=env_model)
    agent1.name = agent1.name + "_shooter"
    agent2.name = agent2.name + "_goalkeeper"
    agents = [agent1, agent2]
    buffers = [PPO_Buffer(args=args, conf=agent.conf, name=agent.name, actor_rnn=args.actor_rnn, device=args.device) for
               agent in agents]
    '''change param'''
    if shooter_conf is not None:
        agent1.conf = shooter_conf

    if goalkeeper_conf is not None:
        agent2.conf = goalkeeper_conf
    agent2.change_om_layers(args.num_om_layers)
    '''log param'''
    cur_logger.log_param(args, [agent.conf for agent in agents])

    '''test agent'''
    global_step = 0
    for epoch in range(1, args.max_epoch + 1):
        cur_logger.log("rank:{}! epoch:{} start!".format(shooter_id, epoch))

        '''collect_trajectory'''
        memory, scores, global_step = collect_trajectory(agents, env, args, global_step, is_prophetic=False)

        '''learn'''
        for i in range(2):
            cur_logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i])
            if args.test_mode == 0 and i == 0:
                pass
            else:
                if args.policy_training == False and i == 1:
                    pass
                else:
                    buffers[i].store_multi_memory(memory[i], last_val=0)
                    agents[i].learn(data=buffers[i].get_batch(), iteration=epoch, no_log=False)

        '''save'''
        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)
    cur_logger.log("test end!")
