from base.MLP import MLP
from base.Actor_RNN import Actor_RNN
from torch.optim import Adam
class Base_ActorCritic(object):
    def __init__(self, a_n_state, v_n_state, n_action, a_hidden_layers, v_hidden_layers, actor_rnn, args, conf, name, logger):
        super(object, self).__init__()
        self.args = args
        self.conf = conf
        self.name = name
        self.logger = logger

        self.a_n_state = a_n_state
        self.v_n_state = v_n_state
        self.n_action = n_action
        self.a_hidden_layers = a_hidden_layers
        self.v_hidden_layers = v_hidden_layers
        self.actor_rnn = actor_rnn
        actor_net = MLP
        if self.actor_rnn:
            actor_net = Actor_RNN
        self.a_net = actor_net(input=a_n_state,
                               output=n_action,
                               hidden_layers_features=a_hidden_layers,
                               output_type="prob")
        self.v_net = MLP(input=v_n_state,
                         output=1,
                         hidden_layers_features=v_hidden_layers,
                         output_type=None)
        self.v_optimizer = Adam(self.v_net.parameters(), lr=self.conf["v_learning_rate"])
        self.a_optimizer = Adam(self.a_net.parameters(), lr=self.conf["a_learning_rate"])
    def share_memory(self):
        self.a_net.share_memory()
        self.v_net.share_memory()

    def choose_action(self, state, greedy=False, **kwargs):
        raise NotImplementedError

    def learn(self, data, iteration, no_log=False):
        raise NotImplementedError

    def change_device(self, device):
        raise NotImplementedError

    def change_conf(self, conf):
        self.conf = conf

    def save_model(self, iteration):
        raise NotImplementedError

    @staticmethod
    def load_model(filepath, args, logger, device, **kwargs):
        raise NotImplementedError
