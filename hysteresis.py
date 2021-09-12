import utils
import torch
from torch.nn import Module


class Hysteresis(Module):
    states = None
    dtype = torch.float64

    def __init__(self, data, h_min, h_max, b_sat, n):
        super(Hysteresis, self).__init__()
        self.h_data = data.to(self.dtype)
        self.h_min = h_min
        self.h_max = h_max
        self.b_sat = b_sat
        self.n = n

        xx, yy = utils.generate_asym_mesh(h_min, h_max, n)
        self.xx = xx.to(self.dtype)
        self.yy = yy.to(self.dtype)

        hyst_density_vector = torch.ones(int(n**2 / 2 + n / 2),
                                              requires_grad=True,
                                              dtype=self.dtype)

        self.register_parameter('_raw_hyst_density_vector',
                                torch.nn.Parameter(hyst_density_vector))

        self.update_states(self.h_data)

    def get_density_vector(self):
        return torch.nn.Softplus()(self._raw_hyst_density_vector)

    def set_density_vector(self, vector):
        self._raw_hyst_density_vector = \
            torch.nn.Parameter(torch.log(
                torch.exp(vector) - 1.0))

    def update_data(self, data):
        self.h_data = data
        self.update_states(data)

    def update_states(self, h):
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
        hyst_state = torch.ones(self.xx.shape) * -1

        # starts off off
        hs = torch.cat((torch.ones(1) * -self.h_min,
                        h))  # H_0=-t, negative saturation limit
        states = torch.empty(
            (len(hs), self.xx.shape[0], self.xx.shape[1]))  # list of hysteresis states
        for i in range(len(hs)):
            if hs[i] > hs[i - 1]:
                hyst_state = torch.tensor([[hyst_state[j][k] if self.yy[j][k] >= hs[i]
                                            else 1
                                            for k in range(len(self.yy[j]))] for j in
                                           range(len(self.yy))])
            elif hs[i] < hs[i - 1]:
                hyst_state = torch.tensor([[hyst_state[j][k] if self.xx[j][k] <= hs[
                    i] else -1 for k in range(len(self.yy[j]))] for j in
                                           range(len(self.xx))])
            hyst_state = torch.tril(hyst_state)
            states[i] = hyst_state
        self.states = states

    def predict_magnetization(self, h_new=None):
        if h_new is not None:
            h = torch.cat((self.h_data, h_new))
        else:
            h = self.h_data

        b = torch.empty(len(h))  # b is the resulting magnetic field
        hyst_density_vector = torch.nn.Softplus()(self._raw_hyst_density_vector)
        dens = utils.vector_to_tril(hyst_density_vector, self.n)
        a = self.b_sat / torch.sum(dens)
        for i in range(len(h)):
            # print(dens * states[i+1])
            b[i] = torch.sum(dens * self.states[i + 1])
        return b * a


