from policy.MBOM import MBOM
from baselines.PPO import PPO, PPO_Buffer
from utils.rl_utils import collect_trajectory, collect_trajectory_reversed
from utils.Logger import Logger
import torch
import multiprocessing as mp
import os

def trainer(args, logger, env, env_model, confs):
    mp.set_start_method("spawn")
    if args.train_mode==0:
        channel_in = [mp.Queue(maxsize=1) for _ in range(args.ranks)]
        channel_out = [mp.Queue(maxsize=1) for _ in range(args.ranks)]
        global_mbom = MBOM(args=args, conf=confs[1], name="football", logger=logger, agent_idx=1, actor_rnn=args.actor_rnn, env_model=None, device=args.device)
        global_buffer = PPO_Buffer(args=args, conf=global_mbom.conf, name=global_mbom.name, actor_rnn=args.actor_rnn, device=args.device)
        processes = []
        for rank in range(args.ranks):
            p = mp.Process(target=worker, args=(args, logger.root_dir, rank, channel_in[rank], channel_out[rank], env, env_model, confs))
            p.start()
            processes.append(p)
        pid_list = [p.pid for p in processes] + [os.getpid()]
        for epoch in range(1, args.max_epoch + 1):
            for rank in range(args.ranks):
                channel_out[rank].put({"a_net": global_mbom.a_net.state_dict(), "v_net": global_mbom.v_net.state_dict()})
            logger.log("global, epoch:{} param shared!".format(epoch))
            datum = []
            for rank in range(args.ranks):
                data = channel_in[rank].get()
                datum.append(data)
            merge_data = dict()
            for key in data.keys():
                merge_data[key] = torch.cat([d[key] for d in datum])
            global_mbom.learn(data=merge_data, iteration=epoch, no_log=False)
            logger.log("global, epoch:{} param updated!".format(epoch, epoch))
            if epoch % args.save_per_epoch == 0:
                    global_mbom.save_model(epoch)
        for p in processes:
            p.join()
        pass
    else:
        print("train_model error")
        raise NameError

def worker(args, root_dir, rank, channel_out, channel_in, env, env_model, confs):
    #env = football_env()
    #env_model = football_env_model(args.device)
    logger = Logger(root_dir, "worker", rank)
    ppo = PPO(args, confs[0], name="football_rank{}".format(rank), logger=logger, actor_rnn=args.actor_rnn, device=args.device)
    MBOM = MBOM(args=args, conf=confs[1], name="football", logger=logger, agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model, device=args.device)
    agents = [ppo, MBOM]
    buffers = [PPO_Buffer(args=args, conf=agent.conf, name=agent.name, actor_rnn=args.actor_rnn, device=args.device) for agent in agents]
    logger.log_param(args, [agent.conf for agent in agents], rank=rank)
    global_step = 0
    for epoch in range(1, args.max_epoch + 1):
        logger.log("rank:{}, epoch:{} start!".format(rank, epoch))
        param = channel_in.get()
        MBOM.a_net.load_state_dict(param["a_net"])
        MBOM.v_net.load_state_dict(param["v_net"])
        memory, scores, global_step = collect_trajectory(agents, env, args, global_step, is_prophetic=True)
        for i in range(2):
            logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i])
            buffers[i].store_multi_memory(memory[i], last_val=0)
        agents[0].learn(data=buffers[0].get_batch(), iteration=epoch, no_log=False)
        channel_out.put(buffers[1].get_batch())
        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)
    print("end")
