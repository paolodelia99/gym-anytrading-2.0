class TradingAgent(object):

    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.net = None

    def act(self, state):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
