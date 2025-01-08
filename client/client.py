class Client(object):
    def __init__(self, client_id, device, model_level, training_intensity):
        self.id = client_id
        self.device = device
        self.model_level = model_level
        self.training_intensity = training_intensity
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
