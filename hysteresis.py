import utils
import torch
from torch.nn import Module


class Hysteresis(Module):
    states = None
    dtype = torch.float64

    def __init__(self, h_data, h_min, h_max, b_sat, n, trainable=True):
        # note h_values inside class are normalized in range [0-1]

        super(Hysteresis, self).__init__()
        self.h_min = h_min
        self.h_max = h_max
        self.b_sat = b_sat
        self.n = n
        self.h_data = self.normalize_h(h_data.to(self.dtype))

        # generate mesh in normalized space
        xx, yy = utils.generate_asym_mesh(0.0, 1.0, n)
        self._xx = xx.to(self.dtype)
        self._yy = yy.to(self.dtype)

        self.vector_shape = int(n ** 2 / 2 + n / 2)
        hyst_density_vector = torch.ones(self.vector_shape,
                                         dtype=self.dtype)

        # if we are trying to train with pytorch autograd
        self.trainable = trainable
        if self.trainable:
            self.register_parameter('_raw_hyst_density_vector',
                                    torch.nn.Parameter(hyst_density_vector))

            self.register_parameter('offset',
                                    torch.nn.Parameter(torch.zeros(1)))

            self.register_parameter('scale',
                                    torch.nn.Parameter(torch.ones(1)))

        else:
            self._raw_hyst_density_vector = hyst_density_vector
            self.offset = torch.zeros(1)
            self.scale = torch.ones(1)

        self.update_states(self.h_data)

    def normalize_h(self, h):
        return (h - self.h_min) / (self.h_max - self.h_min)

    def unnormalize_h(self, h):
        return h * (self.h_max - self.h_min) + self.h_min

    def get_density_vector(self, raw=False):
        if raw:
            return self._raw_hyst_density_vector
        else:
            return torch.nn.Softplus()(self._raw_hyst_density_vector)

    def get_density_matrix(self, raw=False):
        return utils.vector_to_tril(self.get_density_vector(raw),
                                    self.n)

    def get_mesh(self):
        return self.unnormalize_h(self._xx), self.unnormalize_h(self._yy)

    def set_density_vector(self, vector):
        assert vector.shape[0] == self.vector_shape, f'{vector.shape[0]} ' \
                                                     f'vs. {self.vector_shape} '
        assert torch.all(vector >= 0), 'density vector must be positive'
        if self.trainable:
            self._raw_hyst_density_vector = \
                torch.nn.Parameter(torch.log(
                    torch.exp(vector) - torch.tensor(1.0)))
        else:
            self._raw_hyst_density_vector = torch.log(
                    torch.exp(vector) - torch.tensor(1.0))

    def set_data(self, h_data):
        assert len(h_data.shape) == 1
        self.h_data = self.normalize_h(h_data)
        self.update_states(self.h_data)

    def get_h_data(self):
        return self.unnormalize_h(self.h_data)

    def update_states(self, h: torch.Tensor):
        self.states = self.get_states(h)

    def get_states(self, h: torch.Tensor):
        """

        Returns magnetic hysteresis state as an mxnxn tensor, where
        m is the number of distinct applied magnetic fields. The
        states are initially entirely off, and then updated per
        time step depending on both the most recent applied magnetic
        field and prior inputs (i.e. the "history" of the states tensor).

        For each time step, the state matrix is either "swept up" or
        "swept left" based on how the state matrix corresponds to like
        elements in the meshgrid; the meshgrid contains alpha, beta
        coordinates which serve as thresholds for the hysterion state to
        "flip".

        See: https://www.wolframcloud.com/objects/demonstrations/TheDiscretePreisachModelOfHysteresis-source.nb

        Parameters
        ----------
        h : array,
            The applied magnetic field H_1:t={H_1, ... ,H_t}, where
            t represents each time step.

        Raises
        ------
        ValueError
            If n is negative.
        """
        # initial_hyst_state
        hyst_state = torch.ones(self._xx.shape) * -1

        # starts off off
        hs = torch.cat((self.h_min*torch.ones(1), h))  # H_0=-t, negative

        #list of hysteresis states
        states = torch.empty((len(hs), self._xx.shape[0], self._xx.shape[1]))
        for i in range(len(hs)):
            if hs[i] > hs[i - 1]:
                hyst_state = torch.tensor([[hyst_state[j][k] if self._yy[j][k] >= hs[i]
                                            else 1
                                            for k in range(len(self._yy[j]))] for j in
                                           range(len(self._yy))])
            elif hs[i] < hs[i - 1]:
                hyst_state = torch.tensor([[hyst_state[j][k] if self._xx[j][k] <= hs[
                    i] else -1 for k in range(len(self._yy[j]))] for j in
                                           range(len(self._xx))])
            hyst_state = torch.tril(hyst_state)
            states[i] = hyst_state
        return states

    def predict_magnetization(self, h=None, h_new=None):
        assert not (h is not None and h_new is not None), 'cannot specify both h and ' \
                                                          'h_new'
        if h_new is not None:
            normed_h_new = self.normalize_h(h_new)
            h = torch.cat((self.h_data, normed_h_new))
            states = self.get_states(h)
        elif h is not None:
            h = self.normalize_h(h)
            states = self.get_states(h)
        else:
            h = self.h_data
            states = self.states

        b = torch.empty(len(h))  # b is the resulting magnetic field
        hyst_density_vector = torch.nn.Softplus()(self._raw_hyst_density_vector)
        dens = utils.vector_to_tril(hyst_density_vector, self.n)
        for i in range(len(h)):
            # print(dens * states[i+1])
            b[i] = torch.sum(dens * states[i + 1]) / \
                   utils.get_upper_trainagle_size(self.n)
        return b * self.scale + self.offset
