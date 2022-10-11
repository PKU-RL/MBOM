from base.MLP import MLP
import numpy as np
import types
import torch
class Opponent_Model(object):
    def __init__(self, args, conf, name, device=None):
        self.model = MLP(input=conf["n_state"],
                         output=conf["n_opponent_action"],
                         hidden_layers_features=conf["opponent_model_hidden_layers"],
                         output_type="prob")
        self.args = args
        self.conf = conf
        self.name = name
        self.device = device
        if device:
            self.model.to(device)

    def change_device(self, device):
        self.device = device
        self.model.to(device)

    def get_parameter(self, *args):
        return [p.clone() for p in self.model.parameters()]

    def set_parameter(self, new_parameter):
        if isinstance(new_parameter, types.GeneratorType):
            new_parameter = list(new_parameter)
        for target_param, param in zip(list(self.model.parameters()), new_parameter):
            target_param.data.copy_(param.data)

    def get_action_prob(self, state, param=None):
        assert ((type(state) is np.ndarray) or (type(state) is torch.Tensor)), "get_action_prob input type error"
        if (type(state) is np.ndarray):
            state = torch.Tensor(state)
        if self.device:
            state = state.to(self.device)
        if param is not None:
            self.set_parameter(param)
        action_prob, hidden_prob = self.model(state)
        return action_prob, hidden_prob

    def learn(self, data, param, lr, l_times):
        assert len(data["state"].shape) == 2 and len(data["action"].shape) == 2, "learn data shape error"
        if type(data["state"]) is np.ndarray:
            state = torch.Tensor(data["state"])
        else:
            state = data["state"]
        if type(data["action"]) is np.ndarray:
            action_target = torch.LongTensor(data["action"])
        else:
            action_target = data["action"]
        if self.device:
            state = state.to(self.device)
            action_target = action_target.to(self.device)
        self.set_parameter(param)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(l_times):
            optimizer.zero_grad()
            action_eval, _ = self.model(state)
            entropy = torch.distributions.categorical.Categorical(action_eval).entropy().mean()
            loss = loss_fn(action_eval, action_target.squeeze(1)) -1 * entropy
            loss.backward()
            optimizer.step()
        return self.get_parameter(), float(loss)

    def eval(self, fn_name):
        return eval("self." + fn_name)
class OM_Buffer(object):
    def __init__(self, args, conf, device=None):
        self.args = args
        self.conf = conf
        self.device = device

        self.obs = torch.zeros((conf["opponent_model_memory_size"], conf["n_state"]), dtype=torch.float32, device=device)
        self.act = torch.zeros((conf["opponent_model_memory_size"], 1), dtype=int, device=device)
        self.next_idx, self.size, self.max_size = 0, 0, conf["opponent_model_memory_size"]

    def store_memory(self, state, action):
        assert len(state.shape) == 2 and len(action.shape) == 2, "store_memory data shape error"
        assert state.shape[0] == action.shape[0], "store_memory data shape error"
        assert state.shape[1] == self.obs.shape[1], "store_memory state shape error"
        assert action.shape[1] == 1, "store_memory action shape error"
        if type(state) is np.ndarray:
            state = torch.Tensor(state).float()
        if type(action) is np.ndarray:
            action = torch.LongTensor(action)
        if self.device:
            state = state.to(self.device)
            action = action.to(self.device)
        n_batch = state.shape[0]
        if self.next_idx + n_batch < self.max_size:
            self.obs[self.next_idx:self.next_idx+n_batch] = state
            self.act[self.next_idx:self.next_idx + n_batch] = action
            self.next_idx = self.next_idx + n_batch
            self.size = max(self.size, self.next_idx)
        else:
            temp_idx = self.max_size - self.next_idx
            self.obs[self.next_idx:self.max_size] = state[:temp_idx]
            self.act[self.next_idx:self.max_size] = action[:temp_idx]
            self.next_idx = n_batch - (self.max_size - self.next_idx)
            self.obs[:self.next_idx, :] = state[temp_idx:, :]
            self.act[:self.next_idx, :] = action[temp_idx:, :]
            self.size = self.max_size

    def clear_memory(self):
        self.next_idx = 0
        self.size = 0

    def get_batch(self, batch_size):
        idxes = torch.randperm(self.size, device=self.device)[:min(batch_size, self.size)]
        state = self.obs[idxes]
        action = self.act[idxes]
        return dict({"state": state, "action": action})
