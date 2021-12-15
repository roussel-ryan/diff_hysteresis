
FITTING = 0
REGRESSION = 1
NEXT = 2
FUTURE = 3


class ModeEvaluator:
    _mode = FITTING

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in [REGRESSION, NEXT, FUTURE, FITTING]
        self._mode = value

    def fitting(self):
        self._mode = FITTING

    def regression(self):
        self._mode = REGRESSION

    def next(self):
        self._mode = NEXT

    def future(self):
        self._mode = FUTURE
