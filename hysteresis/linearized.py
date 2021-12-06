import torch
from torch import Tensor
from torch.nn import Parameter
from hysteresis.base import TorchHysteresis
from typing import Dict, Callable


class LinearizedHysteresis(TorchHysteresis):
    def __init__(
            self,
            h_train: Tensor = None,
            mesh_scale: float = 1.0,
            trainable: bool = True,
            tkwargs: Dict = None,
            mesh_density_function: Callable = None,
    ):
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

        super(LinearizedHysteresis, self).__init__(
            h_train,
            mesh_scale,
            trainable=trainable,
            tkwargs=tkwargs,
            mesh_density_function=mesh_density_function,
        )

        slope = torch.tensor(0.5)
        if trainable:
            self.register_parameter("slope", Parameter(slope))
        else:
            self.slope = slope

    def predict_magnetization_from_applied_fields(
            self,
            density_vector: Tensor = None,
            scale: Tensor = None,
            slope: Tensor = None,
            offset: Tensor = None,
    ):
        """ Predict magnetization from model regression, ie. from training data"""
        hyst_vec, scale, offset = self._get_model_parameters(
            density_vector,
            scale,
            offset
        )
        if not isinstance(slope, torch.Tensor):
            slope = self.slope

        if isinstance(self.states, torch.Tensor):
            states = self.states

            # do numerical integration
            m = torch.sum(hyst_vec * states, dim=-1) / len(
                hyst_vec
            )

            return scale * m + slope * self._applied_fields + offset

        else:
            return self.get_negative_saturation(density_vector, scale, offset)[0]


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
        """normalize h between 0-1"""
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
