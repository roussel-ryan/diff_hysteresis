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
        self.hysteresis_model = hysteresis_model
        self.L = L

    def apply_field(self, H: Tensor):
        self.hysteresis_model.applied_fields = torch.cat((
            self.hysteresis_model.applied_fields, H.reshape(1)
        ))

    def get_transport_matrix(self):
        m = self.hysteresis_model.predict_magnetization_from_applied_fields()
        m_last = m[-1] if m.shape else m
        return self._calculate_beam_matrix(m_last)

    def get_fantasy_transport_matrix(self, h_fantasy):
        m = self.hysteresis_model.predict_magnetization_next(h_fantasy)
        return self._calculate_beam_matrix(m)

    def get_magnetization_history(self):
        return self.hysteresis_model.predict_magnetization_from_applied_fields()

    def forward(self):
        """ Returns the current transport matrix"""
        return self.get_transport_matrix()

    @abstractmethod
    def _calculate_beam_matrix(self, m: Tensor):
        """calculate beam matrix given magnetization"""
        pass


class HysteresisAccelerator(TorchAccelerator):
    def __init__(self, elements, allow_duplicates=False):
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

        super(HysteresisAccelerator, self).__init__(elements, allow_duplicates)
        self.fantasy = False

        self.hysteresis_element_names = []
        for name, ele in self.elements.items():
            if isinstance(ele, HysteresisMagnet):
                self.hysteresis_element_names += [name]

        # create parameters for optimization
        for name in self.hysteresis_element_names:
            self.register_parameter(name + '_H', Parameter(torch.zeros(1)))

    def calculate_fantasy_transport(self, h_fantasy_dict: Dict):
        M_i = torch.eye(6)
        M = [M_i]
        for name, ele in self.elements.items():
            if isinstance(ele, HysteresisMagnet):
                element_matrix = ele.get_fantasy_transport_matrix(
                    self.get_parameter(name + '.H')
                )
            else:
                element_matrix = ele.forward()

            M += [torch.matmul(element_matrix, M[-1])]

        return torch.cat([m.unsqueeze(0) for m in M], dim=0)

    def forward(self, R: Tensor, full=True, fantasy=False):
        if fantasy:
            h_fantasy_dict = {}
            M = self.calculate_fantasy_transport(h_fantasy_dict)
        else:
            M = self.calculate_transport()

        R_f = self.propagate_beam(M, R)

        if full:
            return R_f[-1]
        else:
            return R_f

    def apply_fields(self, fields_dict: Dict):
        for key in fields_dict:
            assert key in list(self.elements)

        for name, field in fields_dict.items():
            self.elements[name].apply_field(field)


class HysteresisQuad(HysteresisMagnet):
    def __init__(self, name, length, hysteresis_model, scale=1.0):
        super(HysteresisQuad, self).__init__(name, length, hysteresis_model)
        self.quad_model = TorchQuad("", length, None)
        self.quad_model.K1.requires_grad = False
        self.quad_model.L.requires_grad = False
        self.scale = scale

    def _calculate_beam_matrix(self, m: Tensor):
        return self.quad_model.get_matrix(m * self.scale)
