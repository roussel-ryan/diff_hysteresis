from abc import ABC, abstractmethod
from torch.nn import Module, Parameter
import torch
from torch import Tensor
from .first_order import TorchQuad, TorchAccelerator
from typing import Dict
from hysteresis.modes import ModeModule, CURRENT
from hysteresis.base import BaseHysteresis


class HysteresisMagnet(ModeModule, ABC):
    def __init__(self, name, L: Tensor, hysteresis_model: BaseHysteresis):
        Module.__init__(self)
        self.name = name
        self.hysteresis_model = hysteresis_model
        self.L = L

    def apply_field(self, H: Tensor):
        self.hysteresis_model.apply_field(H)

    def get_transport_matrix(self, X: Tensor):
        if isinstance(X, Tensor):
            self.hysteresis_model.mode = self.mode
            m = self.hysteresis_model(X)
            return self._calculate_beam_matrix(m)

        else:
            assert self.hysteresis_model.mode == CURRENT
            self.hysteresis_model.mode = self.mode
            m = self.hysteresis_model()
            return self._calculate_beam_matrix(m).squeeze()

    def forward(self, X: Tensor = None):
        """Returns the current transport matrix"""
        return self.get_transport_matrix(X)

    @abstractmethod
    def _calculate_beam_matrix(self, m: Tensor):
        """calculate beam matrix given magnetization"""
        pass


class HysteresisAccelerator(TorchAccelerator, ModeModule):
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
            self.register_parameter(name + "_H", Parameter(torch.zeros(1)))

    def set_element_modes(self):
        for name, ele in self.elements.items():
            if isinstance(ele, HysteresisMagnet):
                ele.mode = self.mode

    def forward(self, R: Tensor, full=True):
        self.set_element_modes()
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

    def set_histories(self, history_h):
        h_elements = [self.elements[ele] for ele in self.hysteresis_element_names]
        assert len(history_h.shape) == 2
        assert history_h.shape[-1] == len(h_elements)
        for idx, ele in enumerate(h_elements):
            ele.hysteresis_model.set_history(history_h[:, idx])



class HysteresisQuad(HysteresisMagnet):
    def __init__(self, name, length, hysteresis_model, scale=1.0):
        super(HysteresisQuad, self).__init__(name, length, hysteresis_model)
        self.quad_model = TorchQuad("", length, None)
        self.quad_model.K1.requires_grad = False
        self.quad_model.L.requires_grad = False
        self.scale = scale

    def _calculate_beam_matrix(self, m: Tensor):
        return self.quad_model.get_matrix(m * self.scale)
