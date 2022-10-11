shooter_conf = {
    "conf_id": "shooter_conf",
    "n_state": 24,
    "n_action": 11,
    "n_opponent_action": 11,
    "action_dim": 1,
    "type_action": "discrete",
    "action_bounding": 0,  # [()]
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    "v_hidden_layers": [64, 32],
    "a_hidden_layers": [64, 32],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,  # value discount factor
    "lambda": 0.99,  # general advantage estimator
    "epsilon": 0.115,  # ppo clip param
    "entcoeff": 0.0015,
    "a_update_times": 10,
    "v_update_times": 10,
    "buffer_memory_size": 300,

    "num_om_layers": 1,
    "opponent_model_hidden_layers": [64, 32],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,
}
import math
goalkeeper_conf = {
    "conf_id": "goalkeeper_conf",
    "n_state": 24,
    "n_action": 11,
    "n_opponent_action": 11,
    "action_dim": 1,
    "type_action": "discrete",  # "discrete", "continuous"
    "action_bounding": 0,  # [()]
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    "num_om_layers": 3,
    "opponent_model_hidden_layers": [64, 32],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,

    "imagine_model_learning_rate": 0.001,
    "imagine_model_learning_times": 5,
    "roll_out_length": 5,
    "short_term_decay": 0.9,
    "short_term_horizon": 10,
    "mix_factor": 1.1/math.e,

    "v_hidden_layers": [64, 32],
    "a_hidden_layers": [64, 32],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,  # value discount factor
    "lambda": 0.99,  # general advantage estimator
    "epsilon": 0.115,  # ppo clip param
    "entcoeff": 0.0015,
    "a_update_times": 10,
    "v_update_times": 10,
    "buffer_memory_size": 300,
}