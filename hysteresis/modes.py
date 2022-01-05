from torch.nn import Module

FITTING = 0
REGRESSION = 1
NEXT = 2
FUTURE = 3
CURRENT = 4


class ModeModule(Module):
    _mode = FITTING

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in [REGRESSION, NEXT, FUTURE, FITTING, CURRENT]
        self._mode = value
        for ele in self.children():
            if isinstance(ele, ModeModule):
                ele.mode = value

    def fitting(self):
        self.mode = FITTING

    def regression(self):
        self.mode = REGRESSION

    def next(self):
        self.mode = NEXT

    def future(self):
        self.mode = FUTURE
        
    def current(self):
        self.mode = CURRENT
