from baselines.PPO import PPO
from policy.Opponent_Model import Opponent_Model, OM_Buffer
from utils.torch_tool import soft_update
import torch
import numpy as np

class MBOM(PPO):
    def __init__(self, args, conf, name, logger, agent_idx, actor_rnn=False, env_model=None, device=None, rnn_mixer=False):
        assert conf["num_om_layers"] >= 1, "least have 1 layer opponent model"
        super(PPO, self).__init__(a_n_state=conf["n_state"] + conf["n_opponent_action"] if args.true_prob else conf["n_state"] + conf["opponent_model_hidden_layers"][-1],
                                  v_n_state=conf["n_state"],
                                  n_action=conf["n_action"],
                                  a_hidden_layers=conf["a_hidden_layers"],
                                  v_hidden_layers=conf["v_hidden_layers"],
                                  actor_rnn=actor_rnn,
                                  args=args,
                                  conf=conf,
                                  name="MBOM_" + name,
                                  logger=logger)
        self.agent_idx = agent_idx
        self.env_model = env_model
        self.oppo_model = Opponent_Model(args, conf, self.name, device)
        self.om_buffer = OM_Buffer(args, conf, device)
        self.om_phis = np.array([self.oppo_model.get_parameter()] * conf["num_om_layers"])
        if rnn_mixer:
            self.mix_ratio = torch.Tensor([1.0 / conf["num_om_layers"]] * conf["num_om_layers"]).to(device)
        else:
            self.mix_ratio = np.array([1.0 / conf["num_om_layers"]] * conf["num_om_layers"])
        self.device = device
        self.rnn_mixer = rnn_mixer
        if device is not None:
            self.change_device(device)

    def change_om_layers(self, num, rnn_mixer=False):
        self.conf["num_om_layers"] = num
        self.om_phis = np.array([self.oppo_model.get_parameter()] * self.conf["num_om_layers"])
        self.rnn_mixer = rnn_mixer
        if self.rnn_mixer:
            self.mix_ratio = torch.Tensor([1.0 / self.conf["num_om_layers"]] * self.conf["num_om_layers"]).to(self.device)
        else:
            self.mix_ratio = np.array([1.0 / self.conf["num_om_layers"]] * self.conf["num_om_layers"])

    def change_device(self, device):
        super(MBOM, self).change_device(device)
        self.oppo_model.change_device(device)

    def learn(self, data, iteration, no_log=False):
        super(MBOM, self).learn(data, iteration, no_log)

    def choose_action(self, state, greedy=False, hidden_state=None, oppo_hidden_prob=None):
        assert type(state) is np.ndarray or type(state) is torch.Tensor, "choose_action input type error"

        if type(state) is np.ndarray:
            state = torch.Tensor(state).view(-1, self.conf["n_state"]).to(device=self.device)
        if oppo_hidden_prob is None:
            if self.actor_rnn:
                oppo_hidden_prob = self._get_mixed_om_hidden_prob(state, hidden_state=hidden_state)
            else:
                oppo_hidden_prob = self._get_mixed_om_hidden_prob(state)
            self.oppo_hidden_prob = oppo_hidden_prob.clone().detach().cpu()
        if self.actor_rnn:
            return super(MBOM, self).choose_action(state, greedy, oppo_hidden_prob=oppo_hidden_prob, hidden_state=hidden_state)
        return super(MBOM, self).choose_action(state, greedy, oppo_hidden_prob=oppo_hidden_prob)

    def _get_mixed_om_hidden_prob(self, state, hidden_state=None):
        if type(state) is np.ndarray:
            state = torch.Tensor(state).to(self.device)
        if type(hidden_state) is np.ndarray:
            hidden_state = torch.Tensor(hidden_state).to(self.device)
        if self.actor_rnn:
            self._gen_om_phis(state, hidden_state=hidden_state)
        else:
            self._gen_om_phis(state)
        if self.rnn_mixer:
            actions_om_probs = []
            with torch.no_grad():
                for k in range(0, self.conf["num_om_layers"]):
                    actions_om_probs_k, _ = self.oppo_model.get_action_prob(state, self.om_phis[k])
                    actions_om_probs.append(actions_om_probs_k)
                actions_om_probs = torch.cat(actions_om_probs, dim=0)
            mixed_om_hidden_probs = torch.sum(actions_om_probs * self.mix_ratio.view(-1, 1), dim=0)
        else:
            mixed_phis = soft_update(self.om_phis, self.mix_ratio)
            with torch.no_grad():
                mixed_om_action_probs, mixed_om_hidden_probs = self.oppo_model.get_action_prob(state, mixed_phis)
        return mixed_om_hidden_probs
    def _gen_om_phis(self, state, hidden_state=None):
        for k in range(1, self.conf["num_om_layers"]):
            if self.actor_rnn:
                best_response = self._rollout(state, self.om_phis[k - 1], hidden_state=hidden_state)
            else:
                best_response = self._rollout(state, self.om_phis[k - 1])
            data = dict({"state": state, "action": best_response})
            phi, loss = self.oppo_model.learn(data, self.om_phis[k - 1],
                                               lr=self.conf["imagine_model_learning_rate"],
                                               l_times=self.conf["imagine_model_learning_times"])
            self.om_phis[k] = phi
        return None

    def _rollout(self, state, om_phi, hidden_state=None):
        assert type(state) is torch.Tensor, "_rollout input type error"
        rew_record = None
        done_record = None
        cur_batch = 1
        self.env_model.reset()
        oppo_action = torch.LongTensor([i for i in range(self.conf["n_opponent_action"])]).view(-1, 1)
        for i in range(self.conf["roll_out_length"]):
            if i != 0:
                oppo_action = oppo_action.repeat(self.conf["n_opponent_action"], 1)  # [next_batch, 1]

            cur_batch = cur_batch * self.conf["n_opponent_action"]

            state = state.repeat_interleave(self.conf["n_opponent_action"], dim=0)
            if self.actor_rnn:
                hidden_state = hidden_state.repeat_interleave(self.conf["n_opponent_action"], dim=0)

            with torch.no_grad():
                om_action_prob, om_hidden_prob = self.oppo_model.get_action_prob(state, om_phi)
                action, _, _, _, _, _, hidden_state = self.choose_action(state, greedy=True, oppo_hidden_prob=om_hidden_prob, hidden_state=hidden_state) #[cur_batch, 1]
            if self.actor_rnn:
                hidden_state = torch.Tensor(hidden_state).to(device=self.device)

            actions = [None, None]
            actions[self.agent_idx] = action
            actions[1 - self.agent_idx] = oppo_action
            state_, reward, done = self.env_model.step(state, actions)
            if rew_record is None:
                rew_record = reward[self.agent_idx].view(-1, 1).to(self.device)
                done_record = done.view(-1, 1).to(self.device)
            else:
                rew_record = torch.cat([rew_record, reward[self.agent_idx]], dim=1)
                done_record = torch.cat([done_record, done], dim=1)
            if i != self.conf["roll_out_length"] - 1:
                rew_record = rew_record.repeat(self.conf["n_opponent_action"], 1)
                done_record = done_record.repeat(self.conf["n_opponent_action"], 1)
            state = state_
        with torch.no_grad():
            v = self.v_net(state_)
        left_zero = torch.zeros((done_record.shape[0], 1), device=self.device)
        temp_1 = torch.cat([left_zero.bool(), done_record], dim=1)
        temp_11 = rew_record * (~temp_1[:, :-1])
        reward_record = torch.cat([temp_11.float(), v * (~done_record[:, -1:])], dim=1)

        if not hasattr(self, "gamma_list"):
            self.gamma_list = torch.tensor([pow(self.conf["gamma"], i) for i in range(self.conf["roll_out_length"] + 1)], device=self.device)
        discount_r = torch.sum(reward_record * self.gamma_list, axis=1)
        best_response = (torch.argmin(discount_r, dim=0) / (self.conf["n_opponent_action"] ** (self.conf["roll_out_length"] - 1))).view(1, 1).long()
        return best_response

    def observe_oppo_action(self, state, oppo_action, iteration, no_log=True):
        assert type(state) is np.ndarray, "observe_oppo_action input type error"

        if type(oppo_action) is not np.ndarray:
            oppo_action = np.array([oppo_action])
        oppo_action = oppo_action.reshape(1, 1)
        assert oppo_action.shape == (1, 1), "observe_oppo_action input shape error"
        state = state.reshape(1, -1)
        state = torch.Tensor(state).to(device=self.device)
        oppo_action = torch.LongTensor(oppo_action).to(device=self.device)
        if self.conf["num_om_layers"] > 1:
            if self.rnn_mixer:
                self._rnn_cal_mix_ratio(state, oppo_action)
                mix_ratio = self.mix_ratio.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    self._cal_mix_ratio(state, oppo_action)
                    mix_ratio = self.mix_ratio
        self.om_buffer.store_memory(state, oppo_action)

        loss = 0
        if self.om_buffer.size > 2 * self.conf["opponent_model_batch_size"]:
            data = self.om_buffer.get_batch(self.conf["opponent_model_batch_size"])
            phi, loss = self.oppo_model.learn(data, self.om_phis[0],
                                              lr=self.conf["opponent_model_learning_rate"],
                                              l_times=self.conf["opponent_model_learning_times"])
            self.om_phis[0] = phi

        if not no_log:
            self.logger.log_performance(tag=self.name+"/steps", iteration=iteration, Loss_oppo=loss)
            if self.conf["num_om_layers"] > 1:
                mix_ratio_record = {"om_layer{}_mix_ratio".format(i): mix_ratio[i] for i in range(len(mix_ratio))}
                self.logger.log_performance(tag=self.name+"/steps", iteration=iteration, **mix_ratio_record)
                self.logger.log_performance(tag=self.name+"/steps", iteration=iteration, Mix_entropy=torch.distributions.Categorical(torch.Tensor(mix_ratio)).entropy())
        pass
    def _cal_mix_ratio(self, state, oppo_action):
        if not hasattr(self, "short_term_decay_list"):
            self.short_term_decay_list = torch.tensor([pow(self.conf["short_term_decay"], i) for i in range(self.conf["short_term_horizon"])], device=self.device)
            self.all_om_action_prob = torch.full((self.conf["num_om_layers"], self.conf["short_term_horizon"]), fill_value=(1.0 / self.conf["num_om_layers"]), device=self.device)  # [num_om_layers, error_horizon]
        actions_om_probs = []
        om_prob = self.all_om_action_prob.mean(dim=1)
        with torch.no_grad():
            for k in range(0, self.conf["num_om_layers"]):
                actions_om_probs_k, _ = self.oppo_model.get_action_prob(state, self.om_phis[k])
                actions_om_probs.append(actions_om_probs_k)
        actions_om_probs = torch.cat(actions_om_probs, dim=0)
        action_om_probs = actions_om_probs[:, oppo_action.squeeze()]
        actionom_prob = action_om_probs * om_prob
        om_action_prob = actionom_prob / actionom_prob.sum(dim=0, keepdim=True).repeat(self.conf["num_om_layers"])
        self.all_om_action_prob = torch.cat([om_action_prob.unsqueeze(dim=1), self.all_om_action_prob[:, :-1]], dim=1)
        sum_om_action_prob = (self.all_om_action_prob * self.short_term_decay_list).sum(dim=1)
        self.mix_ratio = lambda_softmax(sum_om_action_prob, dim=0, factor=self.conf["mix_factor"]).cpu().numpy()
        pass
    def _rnn_cal_mix_ratio(self, state, oppo_action):
        if not hasattr(self, "mixer"):
            from base.Actor_RNN import Actor_RNN
            import itertools
            self.mixer = Actor_RNN(input=self.conf["num_om_layers"],
                                   output=self.conf["num_om_layers"],
                                   hidden_layers_features=[8, 8],
                                   output_type="prob").to(self.device)
            self.mixer_hidden = torch.zeros(size=(1, 8)).to(self.device)
            self.a_optimizer = torch.optim.Adam(itertools.chain(self.a_net.parameters(), self.mixer.parameters()), lr=self.conf["a_learning_rate"])
        actions_om_probs = []
        with torch.no_grad():
            for k in range(0, self.conf["num_om_layers"]):
                actions_om_probs_k, _ = self.oppo_model.get_action_prob(state, self.om_phis[k])
                actions_om_probs.append(actions_om_probs_k)
            actions_om_probs = torch.cat(actions_om_probs, dim=0)
            target = oppo_action.squeeze().repeat(self.conf["num_om_layers"])
            error_list = torch.nn.functional.cross_entropy(actions_om_probs, target, reduce=False)

        ratio, _, hidden = self.mixer(error_list.view(1, -1), self.mixer_hidden)
        self.mixer_hidden = hidden
        self.mix_ratio = ratio.squeeze()
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
            'agent_idx': self.agent_idx,
            'actor_rnn': self.actor_rnn,
        }
        torch.save(obj, filepath, _use_new_zipfile_serialization=False)
        self.logger.log("model saved in {}".format(filepath))

    @staticmethod
    def load_model(filepath, args, logger, device, **kwargs):
        assert "env_model" in kwargs.keys(), "must input env_model"
        checkpoint = torch.load(filepath, map_location='cpu')
        conf = checkpoint["conf"]
        name = checkpoint["name"].replace("MBOM_", "")
        agent_idx = checkpoint["agent_idx"]
        actor_rnn = checkpoint["actor_rnn"]
        MBOM = MBOM(args, conf, name, logger, agent_idx, actor_rnn, kwargs["env_model"], device, rnn_mixer=args.rnn_mixer)

        MBOM.v_net.load_state_dict(checkpoint['v_net_state_dict'])
        MBOM.a_net.load_state_dict(checkpoint['a_net_state_dict'])
        MBOM.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
        MBOM.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])

        if device:
            MBOM.v_net = MBOM.v_net.to(device)
            MBOM.a_net = MBOM.a_net.to(device)
        if logger is not None:
            logger.log("model successful load, {}".format(filepath))
        return MBOM
def lambda_softmax(y, dim, factor):
    import math
    with torch.no_grad():
        x = math.e * factor
        y = y - y.mean(dim=dim, keepdim=True)
        t = torch.pow(x, y)
        sum_t = t.sum(dim=dim, keepdim=True).repeat_interleave(repeats=y.shape[dim] ,dim=dim)
        t = t / sum_t
        return t