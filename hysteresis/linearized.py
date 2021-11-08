import torch
from torch import Tensor
from torch.nn import Parameter
from hysteresis.base import TorchHysteresis
from typing import Dict, Callable


class LinearizedHysteresis(TorchHysteresis):
    def __init__(self,
                 h_train: Tensor,
                 m_train: Tensor,
                 mesh_scale: float = 1.0,
                 trainable: bool = True,
                 tkwargs: Dict = None,
                 mesh_density_function: Callable = None):
        """
        Special implementation of Preisach hysteresis optimized for ML algorithms.

        Parameters
        ----------
        h_train
        m_train
        mesh_scale
        trainable
        tkwargs
        mesh_density_function
        """

        # define object to handle linear transforms for h,m
        self.transformer = LinearizedTransform(h_train, m_train)
        self.h_train = self.transformer.transform_h(h_train)

        super(LinearizedHysteresis, self).__init__(h_train, mesh_scale,
                                                   trainable=trainable, tkwargs=tkwargs,
                                                   mesh_density_function=mesh_density_function)

        l_offset = torch.zeros(1, **self.tkwargs)
        l_slope = torch.zeros(1, **self.tkwargs)
        l_scale = torch.ones(1, **self.tkwargs)

        if trainable:
            self.register_parameter('l_offset', Parameter(l_offset))
            self.register_parameter('l_slope', Parameter(l_slope))
            self.register_parameter('l_scale', Parameter(l_scale))
        else:
            self.l_offset = l_offset
            self.l_slope = l_slope
            self.l_scale = l_scale

    def predict_magnetization(self,
                              h: Tensor = None,
                              h_new: Tensor = None,
                              density_vector: Tensor = None,
                              l_slope: Tensor = None,
                              l_scale: Tensor = None,
                              l_offset: Tensor = None,
                              real_m: bool = False) -> Tensor:

        assert not (h is not None and h_new is not None), 'cannot specify both h and ' \
                                                          'h_new'
        # get the states based on h
        if h_new is not None:
            hn_new = self.transformer.transform_h(h_new)
            hn = torch.cat((self.h_train, hn_new))
            states, _ = self.get_states(hn)
        elif h is not None:
            hn = self.transformer.transform_h(h)
            states, _ = self.get_states(hn)

        else:
            hn = self.h_train
            states = self.states

        # get the density vector from the raw version
        if density_vector is None:
            hyst_density_vector = torch.nn.Softplus()(self._raw_hyst_density)
        else:
            hyst_density_vector = torch.nn.Softplus()(density_vector)

        slope = l_slope or self.l_slope
        offset = l_offset or self.l_offset
        scale = l_scale or self.l_scale

        m = torch.sum(hyst_density_vector * states[1:], dim=-1) / len(
            hyst_density_vector)

        # calculate M in normalized-linearized space
        M = scale * m - (slope * hn - offset)

        if real_m:
            return self.transformer.transform_m(M, hn)
        else:
            return M


class LinearizedTransform:
    def __init__(self, h, m):
        """
        Object to transform between real and linearized corrdinates for fitting
        models to hysteresis data

        Parameters
        ----------
        h
        m
        """
        self.h = h
        self.m = m
        self._update_coeff()

    def transform_h(self, h):
        """ normalize h between 0-1"""
        return (h - self.A) / (self.B - self.A)

    def untransform_h(self, hn):
        return hn * (self.B - self.A) + self.A

    def transform_m(self, m, h):
        hn = self.transform_h(h)
        _m = (m - self.C) / (self.D - self.C)
        return (_m - hn) / self.E

    def untransform_m(self, mn, hn):
        return (mn * self.E + hn) * (self.D - self.C) + self.C

    def update_data(self, h=None, m=None):
        if h is not None:
            self.h = h

        if m is not None:
            self.m = m

        self._update_coeff()

    def _update_coeff(self):
        self.A = torch.min(self.h)
        self.B = torch.max(self.h)
        self.C = torch.min(self.m)
        self.D = torch.max(self.m)

        hn = (self.h - self.A) / (self.B - self.A)
        mn = (self.m - self.C) / (self.D - self.C)
        mt = mn - hn

        self.E = torch.max(mt)
