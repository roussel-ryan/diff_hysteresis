
REGRESSION = 0
NEXT = 1
FUTURE = 2


class ModeEvaluator:
    _mode = REGRESSION

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in [REGRESSION, NEXT, FUTURE]
        self._mode = value

    def regression(self):
        self._mode = REGRESSION

    def next(self):
        self._mode = NEXT

    def future(self):
        self._mode = FUTURE
