import json
import os
import atexit
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, dir, exp_name, seed):
        self.dir = dir
        self.exp_name = exp_name
        self.seed = seed
        try:
            idx = max(list(map(int, [d.split("_")[0] for d in os.listdir(os.path.join(dir, exp_name))]))) + 1
        except:
            idx = 0
        self.root_dir = os.path.join(dir, exp_name, "{}_{}".format(idx, seed))
        assert not os.path.exists(self.root_dir), "log dir is exist! {}".format(self.root_dir)
        os.makedirs(self.root_dir)
        self.model_dir = os.path.join(self.root_dir, "model")
        os.makedirs(self.model_dir)
        self.log_dir = os.path.join(self.root_dir, "log")
        self.log_file = open(os.path.join(self.root_dir, "log.txt"), 'a')
        atexit.register(self.log_file.close)
        self.param_file =os.path.join(self.root_dir, "param.json")
        self.writer = SummaryWriter(self.log_dir)
        atexit.register(self.writer.flush)
        atexit.register(self.writer.close)
        pass

    def log_performance(self, tag, iteration, **kwargs):
        for k, v in kwargs.items():
            main_tag = tag + "/" + k
            self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict={k: v}, global_step=iteration)
        self.writer.flush()

    def log_param(self, args, confs, **kwargs):
        with open(self.param_file, 'a') as f:
            f.write(json.dumps(kwargs))
            f.write(json.dumps({"args": vars(args), "confs": confs}))
        self.writer.flush()

    def log(self, s):
        print(s)
        self.log_file.write(s + "\n")
        self.log_file.flush()