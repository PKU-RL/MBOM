import numpy as np
import torch
def discount_cumsum(x, discount):
    import scipy.signal
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Episode_Memory():
    def __init__(self):
        self.next_idx = 0
        self.state = []
        self.action = []
        self.logp_a = []
        self.entropy = []
        self.value = []
        self.oppo_hidden_prob = []
        self.hidden_state = []
        self.reward = []

        self.final_state = None
        pass

    def store_final_state(self, state, info):
        self.final_state = (state, info)
    def store_oppo_logp_a(self, oppo_logp_a):
        if not hasattr(self, "oppo_logp_a"):
            self.oppo_logp_a = []
        self.oppo_logp_a.append(oppo_logp_a)
    def store_env_info(self, state, reward):
        self.state.append(state)
        self.reward.append(reward)
    def store_oppo_hidden_prob(self, oppo_hidden_prob):
        self.oppo_hidden_prob.append(oppo_hidden_prob)
    def store_action_info(self, choose_action_return):
        self.action.append(choose_action_return[0])
        self.logp_a.append(choose_action_return[1])
        self.entropy.append(choose_action_return[2])
        self.value.append(choose_action_return[3])
        self.hidden_state.append(choose_action_return[6])
    def get_data(self, is_meta_mapg=False):
        data = dict({
            "state":np.stack(self.state),
            "action":np.stack(self.action),
            "logp_a":torch.stack(self.logp_a) if is_meta_mapg else np.stack(self.logp_a),
            "entropy":np.stack(self.entropy),
            "value":np.stack(self.value).squeeze(axis=1),
            "reward":np.stack(self.reward).reshape(-1, 1),
            "oppo_hidden_prob":np.stack(self.oppo_hidden_prob).squeeze(axis=1),
            "hidden_state":np.stack(self.hidden_state).squeeze(axis=1) if np.any(np.stack(self.hidden_state) != None) else np.stack(self.hidden_state),
        })
        if hasattr(self, "oppo_logp_a"):
            data["oppo_logp_a"] = np.stack(self.oppo_logp_a)
        return data
    def get_score(self):
        score = 0
        for i in range(len(self.reward)):
            score += self.reward[i]
        return float(score)

def collect_trajectory(agents, env, args, global_step, is_prophetic=False, greedy=False):
    memories = [[], []]
    scores = [[], []]

    for _ in range(1, args.eps_per_epoch + 1):
        hidden_state = [agent.init_hidden_state(n_batch=1) for agent in agents]
        oppo_hidden_prob = np.array([None, None])
        state = env.reset()
        temp_memory = [Episode_Memory(), Episode_Memory()]

        while True:
            global_step += 1
            actions = np.array([0, 0], dtype=int)
            for agent_idx, agent in enumerate(agents):
                if type(agent).__name__ == "MBAM":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=oppo_hidden_prob[1-agent_idx] if is_prophetic == True else None, greedy=greedy)
                elif type(agent).__name__ == "PPO":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=None, greedy=greedy)
                else:
                    raise TypeError

                if args.prophetic_onehot:
                    temp_a = action_info[0].item()
                    oppo_hidden_prob[agent_idx] = np.eye(action_info[5].shape[1])[temp_a].reshape(1, -1)
                    action_info = (action_info[0], action_info[1], action_info[2], action_info[3], action_info[4], oppo_hidden_prob[agent_idx], action_info[6])
                else:
                    oppo_hidden_prob[agent_idx] = action_info[5]
                temp_memory[agent_idx].store_action_info(action_info)
                temp_memory[1 - agent_idx].store_oppo_hidden_prob(action_info[5])
                hidden_state[agent_idx] = action_info[6]
                actions[agent_idx] = action_info[0].item()

            state_, reward, done, info = env.step(actions)

            for i in range(len(agents)):
                temp_memory[i].store_env_info(state[i], reward[i])

            if not is_prophetic:
                for agent_idx, agent in enumerate(agents):
                    if hasattr(agent, "observe_oppo_action"):
                        agent.observe_oppo_action(state=state[agent_idx], oppo_action=actions[1-agent_idx],
                                                  iteration=global_step, no_log=False)
            state = state_
            if done:
                for i in range(len(agents)):
                    temp_memory[i].store_final_state(state_[i], info)
                    memories[i].append(temp_memory[i])
                    scores[i].append(temp_memory[i].get_score())
                agents[1].logger.log_performance(tag=agents[1].name + "/steps", iteration=global_step,
                                                 Same_pick_sum=info["same_pick_sum"],
                                                 Coin_sum=info["coin_sum"],
                                                 Pick_ratio=info["same_pick_sum"]/info["coin_sum"])
                break
    scores = [sum(scores[i])/len(scores[i]) for i in range(len(agents))]
    return memories, scores, global_step

def collect_trajectory_reversed(agents, env, args, global_step, is_prophetic=False):
    memories = [[], []]
    scores = [[], []]

    for _ in range(1, args.eps_per_epoch + 1):
        hidden_state = [agent.init_hidden_state(n_batch=1) for agent in agents]
        oppo_hidden_prob = np.array([None, None])
        state = env.reset()
        temp_memory = [Episode_Memory(), Episode_Memory()]
        while True:
            global_step += 1
            actions = np.array([0, 0], dtype=int)
            for agent_idx, agent in list(enumerate(agents))[::-1]:
                if type(agent).__name__ == "MBAM":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=oppo_hidden_prob[1-agent_idx] if is_prophetic == True else None)
                elif type(agent).__name__ == "PPO":
                    action_info = agent.choose_action(state[agent_idx], hidden_state=hidden_state[agent_idx], oppo_hidden_prob=None)
                else:
                    raise TypeError
                if args.prophetic_onehot:
                    temp_a = action_info[0].item()
                    oppo_hidden_prob[agent_idx] = np.eye(action_info[5].shape[1])[temp_a].reshape(1, -1)
                    action_info = (action_info[0], action_info[1], action_info[2], action_info[3], action_info[4], oppo_hidden_prob[agent_idx], action_info[6])
                else:
                    oppo_hidden_prob[agent_idx] = action_info[5]
                temp_memory[agent_idx].store_action_info(action_info)
                temp_memory[1 - agent_idx].store_oppo_hidden_prob(action_info[5])
                hidden_state[agent_idx] = action_info[6]
                actions[agent_idx] = action_info[0].item()

            state_, reward, done, info = env.step(actions)
            for i in range(len(agents)):
                temp_memory[i].store_env_info(state[i], reward[i])

            if not is_prophetic:
                for agent_idx, agent in enumerate(agents):
                    if hasattr(agent, "observe_oppo_action"):
                        agent.observe_oppo_action(state=state[agent_idx], oppo_action=actions[1-agent_idx],
                                                  iteration=global_step, no_log=False)
            state = state_
            if done:
                for i in range(len(agents)):
                    temp_memory[i].store_final_state(state_[i], info)
                    memories[i].append(temp_memory[i])
                    scores[i].append(temp_memory[i].get_score())
                break
    scores = [sum(scores[i])/len(scores[i]) for i in range(len(agents))]
    return memories, scores, global_step