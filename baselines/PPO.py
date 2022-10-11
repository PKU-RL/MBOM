import numpy as np
import torch
from torch.distributions.categorical import Categorical
from baselines.Base_ActorCritic import Base_ActorCritic
from utils.datatype_transform import dcn
from utils.rl_utils import discount_cumsum

class PPO(Base_ActorCritic):
    def __init__(self, args, conf, name, logger, actor_rnn, device=None):
        super(PPO, self).__init__(a_n_state=conf["n_state"],
                                  v_n_state=conf["n_state"],
                                  n_action=conf["n_action"],
                                  a_hidden_layers=conf["a_hidden_layers"],
                                  v_hidden_layers=conf["v_hidden_layers"],
                                  actor_rnn=actor_rnn,
                                  args=args,
                                  conf=conf,
                                  name="PPO_" + name,
                                  logger=logger)
        self.device = device

    def init_hidden_state(self, n_batch):
        if self.actor_rnn:
            hidden_state = torch.zeros((n_batch, self.conf["a_hidden_layers"][0]), device=self.device)
            return hidden_state
        else:
            return None

    def choose_action(self, state, greedy=False, hidden_state=None, oppo_hidden_prob=None):
        assert (type(state) is np.ndarray) or (type(state) is torch.Tensor), "choose_action input type error"
        if type(state) is np.ndarray:
            state = state.reshape(-1, self.conf["n_state"])
            a_state = torch.Tensor(state).to(device=self.device)
            v_state = torch.Tensor(state).to(device=self.device)
        else:
            a_state = state.clone()
            v_state = state
        if self.device:
            a_state = a_state.to(self.device)
            v_state = v_state.to(self.device)
        if oppo_hidden_prob is not None:
            if type(oppo_hidden_prob) is np.ndarray:
                oppo_hidden_prob = torch.Tensor(oppo_hidden_prob).to(device=self.device)
            oppo_hidden_prob = oppo_hidden_prob.view((-1, self.conf["n_opponent_action"] if self.args.true_prob else
            self.conf["opponent_model_hidden_layers"][-1]))
            if self.args.prophetic_onehot:
                oppo_hidden_prob = torch.eye(self.conf["n_opponent_action"], device=self.device)[
                    torch.argmax(oppo_hidden_prob, dim=1)]
            a_state = torch.cat([a_state, oppo_hidden_prob], dim=1)
        if hidden_state is not None:
            if type(hidden_state) is np.ndarray:
                hidden_state = torch.Tensor(hidden_state).to(device=self.device)
            hidden_state = hidden_state.view((-1, self.conf["a_hidden_layers"][0]))
        value = self.v_net(v_state)
        if self.actor_rnn:
            action_prob, hidden_prob, hidden_state = self.a_net(a_state, hidden_state)
        else:
            action_prob, hidden_prob = self.a_net(a_state)
        if greedy:
            pi = Categorical(action_prob)
            _, action = torch.max(action_prob, dim=1)
            logp_a = pi.log_prob(action)
            entropy = pi.entropy()
        else:
            pi = torch.distributions.Categorical(action_prob)
            action = pi.sample()
            logp_a = pi.log_prob(action)
            entropy = pi.entropy()
        return (dcn(action).astype(np.int32),
                dcn(logp_a),
                dcn(entropy),
                dcn(value),
                dcn(action_prob),
                dcn(hidden_prob),
                dcn(hidden_state) if self.actor_rnn else None)

    def learn(self, data, iteration, no_log=True):
        for param_group in self.a_optimizer.param_groups:
            param_group['lr'] = self.conf["a_learning_rate"]
        for param_group in self.v_optimizer.param_groups:
            param_group['lr'] = self.conf["v_learning_rate"]

        state = data["state"]
        a_state = state.clone()
        v_state = state
        if "MBOM" in self.name:
            oppo_hidden_prob = data["oppo_hidden_prob"]
            a_state = torch.cat([a_state, oppo_hidden_prob], dim=1)
        if self.actor_rnn:
            hidden_state = data["hidden_state"]
        action = data["action"]
        reward_to_go = data["reward_to_go"]
        advantage = data["advantage"]
        logp_a = data["logp_a"]

        if self.device:
            a_state = a_state.to(self.device)
            v_state = v_state.to(self.device)
            action = action.to(self.device)
            reward_to_go = reward_to_go.to(self.device)
            advantage = advantage.to(self.device)
            logp_a = logp_a.to(self.device)
            if self.actor_rnn:
                hidden_state = hidden_state.to(self.device)

        def compute_loss_a(state, action, advantage, logp_old):
            # Policy loss
            if self.conf["type_action"] == "discrete":
                if self.actor_rnn:
                    prob, _, _ = self.a_net(state, hidden_state)
                else:
                    prob, _ = self.a_net(state)
                pi = Categorical(prob)
                logp = pi.log_prob(action.squeeze())
            else:  # "continuous"
                raise NotImplementedError
            logp_old = logp_old.squeeze()
            assert logp.shape == logp_old.shape, "compute_loss_a error! logp.shape != logp_old.shape"
            ratio = torch.exp(logp - logp_old.squeeze())
            advantage = advantage.squeeze()
            assert ratio.shape == advantage.shape, "compute_loss_a error! ratio.shape != advantage.shape"
            clip_advantage = torch.clamp(ratio, 1 - self.conf["epsilon"], 1 + self.conf["epsilon"]) * advantage
            ent = pi.entropy().mean()
            loss_a = -(torch.min(ratio * advantage, clip_advantage)).mean() - self.conf["entcoeff"] * ent

            # extra info
            with torch.no_grad():
                approx_kl = (logp_old - logp).mean().item()
                ent_info = ent.item()
                clipped = ratio.gt(1 + self.conf["epsilon"]) | ratio.lt(1 - self.conf["epsilon"])
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                a_net_info = dict(kl=approx_kl, ent=ent_info, cf=clipfrac)
            return loss_a, a_net_info
        def compute_loss_v(state, reward_to_go):
            return ((self.v_net(state) - reward_to_go) ** 2).mean()

        loss_a_old, a_info_old = compute_loss_a(a_state, action, advantage, logp_a)
        loss_a_old = loss_a_old.detach().item()
        loss_v_old = compute_loss_v(v_state, reward_to_go).detach().item()

        for _ in range(self.conf["a_update_times"]):
            self.a_optimizer.zero_grad()
            loss_a, _ = compute_loss_a(a_state, action, advantage, logp_a)
            loss_a.backward()
            self.a_optimizer.step()
        for _ in range(self.conf["v_update_times"]):
            self.v_optimizer.zero_grad()
            loss_v = compute_loss_v(v_state, reward_to_go)
            loss_v.backward()
            self.v_optimizer.step()
        if not no_log:
            self.logger.log_performance(tag=self.name + "/epochs", iteration=iteration,
                                        Loss_a=loss_a_old, Loss_v=loss_v_old,
                                        KL=a_info_old['kl'], Entropy=a_info_old['ent'],
                                        )
        pass

    def save_model(self, iteration):
        import os
        filepath = os.path.join(self.logger.model_dir, self.name + "_iter" + str(iteration) + ".ckp")
        obj = {
            'v_net_state_dict': self.v_net.state_dict(),
            'a_net_state_dict': self.a_net.state_dict(),
            'v_optimizer_state_dict': self.v_optimizer.state_dict(),
            'a_optimizer_state_dict': self.a_optimizer.state_dict(),
            'args': self.args,
            'conf': self.conf,
            'name': self.name,
            'actor_rnn': self.actor_rnn,
        }
        torch.save(obj, filepath, _use_new_zipfile_serialization=False)
        self.logger.log("model saved in {}".format(filepath))

    @staticmethod
    def load_model(filepath, args, logger, device, **kwargs):
        checkpoint = torch.load(filepath, map_location='cpu')
        conf = checkpoint["conf"]
        name = checkpoint["name"].replace("PPO_", "")
        actor_rnn = checkpoint["actor_rnn"]
        ppo = PPO(args, conf, name, logger, actor_rnn, device)

        ppo.v_net.load_state_dict(checkpoint['v_net_state_dict'])
        ppo.a_net.load_state_dict(checkpoint['a_net_state_dict'])
        ppo.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
        ppo.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])

        if device:
            ppo.v_net = ppo.v_net.to(device)
            ppo.a_net = ppo.a_net.to(device)
        if logger is not None:
            logger.log("model successful load, {}".format(filepath))
        return ppo

class PPO_Buffer(object):
    def __init__(self, args, conf, name, actor_rnn, device=None):
        self.args = args
        self.conf = conf
        self.name = name
        self.actor_rnn = actor_rnn
        self.device = device
        self.gamma = conf["gamma"]
        self.lam = conf["lambda"]

        self.state = torch.zeros((conf["buffer_memory_size"], conf["n_state"]), dtype=torch.float32)
        self.action = torch.zeros((conf["buffer_memory_size"], 1), dtype=int)
        self.reward = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        self.reward_to_go = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        self.advantage = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        self.logp_a = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        self.value = torch.zeros((conf["buffer_memory_size"], 1), dtype=torch.float32)
        if self.actor_rnn:
            self.hidden_state = torch.zeros((conf["buffer_memory_size"], self.conf["a_hidden_layers"][0]), dtype=torch.float32)
        if "MBOM" in self.name:
            self.oppo_hidden_prob = torch.zeros((conf["buffer_memory_size"], self.conf["n_opponent_action"]) if self.args.true_prob else (conf["buffer_memory_size"], self.conf["opponent_model_hidden_layers"][-1]), dtype=torch.float32)

        self.next_idx, self.max_size = 0, conf["buffer_memory_size"]

    def store_memory(self, episode_memory, last_val=0):
        data = episode_memory.get_data()
        n_batch = data["state"].shape[0]
        for k in data.keys():
            assert data[k].shape[0] == n_batch, "input size error"

        reward = data["reward"]
        if type(reward) is torch.Tensor: reward = dcn(reward)
        value = data["value"]
        if type(value) is torch.Tensor: value = dcn(value)

        reward_l = np.append(reward, last_val)
        value_l = np.append(value, last_val)
        deltas = reward_l[:-1] + self.gamma * value_l[1:] - value_l[:-1]
        advantage = discount_cumsum(deltas, self.gamma * self.lam)
        reward_to_go = discount_cumsum(reward_l, self.gamma)[:-1]

        path_slice = slice(self.next_idx, self.next_idx + n_batch)
        assert self.next_idx + n_batch <= self.max_size, "Buffer {} Full!!!".format(self.name)
        if self.actor_rnn:
            hidden_state = data["hidden_state"]
            if type(hidden_state) is np.ndarray: hidden_state = torch.Tensor(hidden_state)
            self.hidden_state[path_slice] = hidden_state

        if "MBOM" in self.name:
            oppo_hidden_prob = data["oppo_hidden_prob"]
            if type(oppo_hidden_prob) is np.ndarray: oppo_hidden_prob = torch.Tensor(oppo_hidden_prob)
            self.oppo_hidden_prob[path_slice] = oppo_hidden_prob

        action = data["action"]
        if type(action) is np.ndarray: action = torch.LongTensor(action)
        self.action[path_slice] = action

        state = data["state"]
        if type(state) is np.ndarray: state = torch.Tensor(state)
        self.state[path_slice] = state

        logp_a = data["logp_a"]
        if type(logp_a) is np.ndarray: logp_a = torch.Tensor(logp_a)
        self.logp_a[path_slice] = logp_a

        advantage = advantage.copy()
        advantage = torch.Tensor(advantage).view(-1, 1)
        self.advantage[path_slice] = advantage

        reward_to_go = reward_to_go.copy()
        reward_to_go = torch.Tensor(reward_to_go).view(-1, 1)
        self.reward_to_go[path_slice] = reward_to_go

        self.next_idx = self.next_idx + n_batch
        pass

    def store_multi_memory(self, data, last_val=0):
        if type(data) == list:
            for i, d in enumerate(data):
                if (type(last_val) is not list) and (type(last_val) is not np.ndarray):
                    self.store_memory(d, last_val)
                else:
                    self.store_memory(d, last_val[i])
        else:
            self.store_memory(data, last_val)

    def clear_memory(self):
        self.next_idx = 0

    def get_batch(self, batch_size=0):
        state = self.state[:self.next_idx].to(device=self.device)
        action = self.action[:self.next_idx].to(device=self.device)
        reward_to_go = self.reward_to_go[:self.next_idx].to(device=self.device)
        advantage = self.advantage[:self.next_idx].to(device=self.device)
        adv_mean, adv_std = torch.mean(advantage), torch.std(advantage)
        if adv_std == 0.0:
            advantage = (advantage - adv_mean)
            adv_mean = adv_mean.detach().cpu()
            adv_std = adv_std.detach().cpu()
        else:
            advantage = (advantage - adv_mean) / adv_std
            adv_mean = adv_mean.detach().cpu()
            adv_std = adv_std.detach().cpu()
        logp_a = self.logp_a[:self.next_idx].to(device=self.device)
        data = dict({"state": state,
                     "action": action,
                     "reward_to_go": reward_to_go,
                     "advantage": advantage,
                     "logp_a": logp_a})
        if self.actor_rnn:
            hidden_state = self.hidden_state[:self.next_idx].to(device=self.device)
            data["hidden_state"] = hidden_state
        if "MBOM" in self.name:
            oppo_hidden_prob = self.oppo_hidden_prob[:self.next_idx].to(device=self.device)
            data["oppo_hidden_prob"] = oppo_hidden_prob
        self.next_idx = 0
        return data
