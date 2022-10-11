import argparse
import random
import time
from utils.Logger import Logger
from trainer import trainer
from tester import tester
import os

def main(args):
    seed = random.randint(0, int(time.time())) if args.seed == -1 else args.seed
    dir = "./logs" if args.dir == "" else args.dir

    if args.env == "One-on-One":
        dir = os.path.join(dir, "One-on-One")
        #env
        #env_model
    elif args.env[:10] == "Triangel Game":
        dir = os.path.join(dir, "Triangel Game")
        # env
        # env_model
    else:
        raise NameError
    exp_name = "{}/{}{}{}{}".format(args.prefix,
                                    ("pone_" if args.prophetic_onehot else ""),
                                    ("trueprob_" if args.true_prob else ""),
                                    ("rnn_" if args.actor_rnn else ""),
                                    args.exp_name)
    logger = Logger(dir, exp_name, seed)
    if args.prefix == "train":
        trainer(args, logger)
    elif args.prefix == "test":
        tester(args, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--exp_name", type=str, default="coin_game_ppo_vs_MBOM", help="football_ppo_vs_MBOM\n " +
                                                                                     "trigame_ppo_vs_MBOM\n")
    parser.add_argument("--env", type=str, default="coin_game", help="One-on-One\n" +
                                                                   "Triangel Game\n")

    parser.add_argument("--prefix", type=str, default="test", help="train or test or search")

    parser.add_argument("--train_mode", type=int, default=1, help="0 1 2means N vs 1 de ppo vs MBOM, MBOMvsppo, continue_train")
    parser.add_argument("--alter_train", type=int, default=0, help="0 1 means no and yes")
    parser.add_argument("--alter_interval", type=int, default=100, help="epoch")
    parser.add_argument("--continue_train", type=bool, default=False, help="0 1 means no and yes")
    parser.add_argument("--batch_size", type=int, default=2, help="")

    parser.add_argument("--test_mode", type=int, default=1, help="0 1 2 means layer0, layer1, layer2")
    parser.add_argument("--test_mp", type=int, default=1, help="multi processing")

    parser.add_argument("--seed", type=int, default=-1, help="-1 means random seed")
    parser.add_argument("--ranks", type=int, default=1, help="for prefix is train")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--dir", type=str, default="", help="")

    parser.add_argument("--eps_max_step", type=int, default=30, help="")
    parser.add_argument("--eps_per_epoch", type=int, default=10, help="")
    parser.add_argument("--save_per_epoch", type=int, default=100, help="")
    parser.add_argument("--max_epoch", type=int, default=100, help="train epoch")
    parser.add_argument("--num_om_layers", type=int, default=3, help="train epoch")
    parser.add_argument("--rnn_mixer", type=bool, default=False, help="True or False")

    parser.add_argument("--actor_rnn", type=bool, default=False, help="True or False")
    parser.add_argument("--true_prob", type=bool, default=False, help="True or False, edit Actor_RNN.py line 47-48")
    parser.add_argument("--prophetic_onehot", type=bool, default=False, help="True or False")
    parser.add_argument("--policy_training", type=bool, default=False, help="True or False")

    parser.add_argument("--record_more", type=bool, default=False, help="True or False")
    parser.add_argument("--config", type=str, default="", help="extra info")
    args = parser.parse_args()
    main(args)