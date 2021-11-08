from abc import ABC, abstractmethod
from torch.nn import Module, Parameter
import torch
from torch import Tensor
from .first_order import TorchQuad, TorchAccelerator
from typing import Dict


class HysteresisMagnet(Module, ABC):
    def __init__(self, name, L: Tensor, hysteresis_model):
        Module.__init__(self)
        self.name = name
        self.history = None
        self.register_buffer(f'length',
                             Parameter(L))
        self.register_parameter('fantasy_H',
                                Parameter(-1.0*torch.ones(1)))
        self.hysteresis_model = hysteresis_model
        self.mode = 'fantasy'

    def apply_field(self, H: Tensor):
        if self.history is None:
            self.history = H
        else:
            self.history = torch.cat((self.history, H))

    def get_magnetization(self, h_fantasy: Tensor = None) -> Tensor:
        """
        Get magnetization from the hysteresis model

        Parameters
        ----------
        h_fantasy : Tensor, optional
            If specified calculate the magnetization at a fantasy point `h_fantasy`
            given the current history state. Otherwise, return the magnetization at
            the last history.

        Returns
        -------
        m : Tensor
            Magnetization value

        """
        if h_fantasy is None:
            h = self.history
        else:
            if self.history is not None:
                h = torch.cat((self.history, h_fantasy))
            else:
                h = h_fantasy

        if h is not None:
            m = self.hysteresis_model.predict_magnetization(h=torch.atleast_1d(h))
        else:
            m = self.hysteresis_model.predict_magnetization()
        m = m[-1] if m.shape else m

        return m

    def get_transport_matrix(self, fantasy_H: Tensor = None):
        m = self.get_magnetization(fantasy_H)
        return self._calculate_beam_matrix(m)

    @property
    def M(self) -> Tensor:
        """ get current beam matrix """
        self.mode = 'current'
        matr = self.get_transport_matrix()
        self.mode = 'fantasy'
        return matr

    @property
    def state(self) -> Tensor:
        return self.history[-1]

    def forward(self) -> Tensor:
        """predict future beam matrix for optimization"""
        return self.get_transport_matrix(self.fantasy_H)

    @abstractmethod
    def _calculate_beam_matrix(self, m: Tensor):
        """calculate beam matrix given magnetization"""
        pass


class HysteresisAccelerator(TorchAccelerator):
    def __init__(self, elements):
        """
        Modifies TorchAccelerator class to include tracking of state varaibles
        relevant to hysteresis effects. By default forward calls to the model are
        fantasy evaluations of named parameters. For example, if we are optimizing or
        plotting calls to the forward method performs a calculation of what the next
        immediate step would result in.

        Parameters
        ----------
        elements
        """

        super(HysteresisAccelerator, self).__init__(elements)

    def calculate_current_transport(self):
        M_tot = torch.eye(6)
        for _, ele in self.elements.items():
            if isinstance(ele, HysteresisMagnet):
                ele.mode = 'current'
                M = ele()
                ele.mode = 'fantasy'
            else:
                M = ele()

            M_tot = torch.matmul(M, M_tot)
        return M_tot

    def apply_fields(self, fields_dict: Dict):
        for name, field in fields_dict.items():
            self.elements[name].apply_field(field)


class HysteresisQuad(HysteresisMagnet):
    def __init__(self, name, length, hysteresis_model, scale=1.0):
        super(HysteresisQuad, self).__init__(name, length, hysteresis_model)
        self.quad_model = TorchQuad('', length, torch.zeros(1))
        self.quad_model.K1.requires_grad = False
        self.quad_model.L.requires_grad = False
        self.scale = scale

    def _calculate_beam_matrix(self, m: Tensor):
        return self.quad_model.forward(m * self.scale)


