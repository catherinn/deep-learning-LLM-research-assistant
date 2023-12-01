class LogicProcess():

    def __init__(self):
        self._step = 0
        self.url = ''
        self.model_vector = None

    @property
    def step(self):
        """Getter method."""
        return self._step

    @step.setter
    def step(self, value):
        """Setter method."""
        self._step = value