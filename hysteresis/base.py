import torch
from torch.nn import Module, Parameter
from torch import Tensor
from typing import Dict, Callable
from .meshing import create_triangle_mesh, default_mesh_size
from .states import get_states, predict_batched_state
from .transform import HysteresisTransform
from .modes import ModeEvaluator, REGRESSION, NEXT, FUTURE

import logging

logger = logging.getLogger(__name__)


class BaseHysteresis(Module, ModeEvaluator):

    def __init__(
            self,
            train_h: Tensor = None,
            train_m: Tensor = None,
            trainable: bool = True,
            tkwargs: Dict = None,
            mesh_scale: float = 1.0,
            mesh_density_function: Callable = None,
            polynomial_degree: int = 1,
            polynomial_fit_iterations: int = 3000,
            temp: float = 1e-2,
    ):
        super(BaseHysteresis, self).__init__()

        self.tkwargs = tkwargs or {}
        self.tkwargs.update({"dtype": torch.double, "device": "cpu"})

        # generate mesh grid on 2D normalized domain [[0,1],[0,1]]
        self.temp = temp
        self.mesh_scale = mesh_scale
        self.mesh_points = torch.tensor(
            create_triangle_mesh(mesh_scale, mesh_density_function), **self.tkwargs
        )

        # initialize trainable parameters
        density = torch.zeros(len(self.mesh_points))
        offset = torch.zeros(1)
        scale = torch.ones(1)
        slope = torch.zeros(1)
        param_vals = [density, offset, scale, slope]
        param_names = [
            "_raw_hysterion_density",
            "offset",
            "scale",
            "slope"
        ]

        self.trainable = trainable
        for param_name, param_val in zip(param_names, param_vals):
            if self.trainable:
                self.register_parameter(param_name, Parameter(param_val))

            else:
                self.register_buffer(param_name, Parameter(param_val))

        # if provided create normalization transform and set magnetization history
        self.polynomial_degree = polynomial_degree
        self.polynomial_fit_iterations = polynomial_fit_iterations

        # if training info is specified then set the history vars and create the fit
        # object - otherwise initialize with default values
        if isinstance(train_h, Tensor):
            self.set_history(train_h, train_m)

    def set_history(self, history_h, history_m, retrain_normalization=True):
        """ set historical state values and recalculate hysterion states"""

        if isinstance(history_h, Tensor):
            history_h = history_h.to(**self.tkwargs)

        if isinstance(history_m, Tensor):
            history_m = history_m.to(**self.tkwargs)

            if torch.equal(history_h, history_m):
                raise RuntimeError('train h and train m cannot be equal')

        if retrain_normalization:
            self.transformer = HysteresisTransform(
                history_h, history_m,
                self.polynomial_degree,
            )

        _history_h, _history_m = self.transformer.transform(history_h, history_m)
        self.register_buffer('_history_h', _history_h.detach())

        if isinstance(_history_m, Tensor):
            self.register_buffer('_history_m', _history_m.detach())

        # recalculate states
        _states = get_states(self._history_h, self.mesh_points, temp=self.temp)
        self.register_buffer('_states', _states)

    @property
    def hysterion_density(self):
        return torch.nn.Softplus()(self._raw_hysterion_density)

    @hysterion_density.setter
    def hysterion_density(self, value: Tensor):
        self._raw_hysterion_density = Parameter(
            torch.log(torch.exp(value.clone()) - 1).to(**self.tkwargs)
        )

    def _predict_normalized_magnetization(self, states, h):
        m = torch.sum(self.hysterion_density * states, dim=-1) / torch.sum(
            self.hysterion_density
        )
        return self.scale * m + self.offset + h * self.slope

    def get_negative_saturation(self):
        return self.transformer.untransform(torch.zeros(1), -self.scale +
                                            self.offset)[1].to(**self.tkwargs)

    def forward(self, x: Tensor, return_real=False):
        x = x.to(**self.tkwargs)
        if self._mode == REGRESSION:
            assert torch.all(torch.isclose(
                x,
                self.transformer.untransform(self._history_h)[0]
            )), "must do regression on history fields if in REGRESSION mode"
            states = self._states
            norm_h = self._history_h

        elif self.mode == FUTURE:
            if len(x.shape) != 1:
                raise ValueError('input must be 1D for FUTURE mode')

            norm_h, _ = self.transformer.transform(x)
            states = get_states(
                norm_h,
                self.mesh_points,
                current_state=self._states[-1],
                current_field=self._history_h[-1],
                tkwargs=self.tkwargs
            )

        elif self.mode == NEXT:
            if x.shape[1:] != torch.Size([1, 1]) and len(x.shape) == 3:
                raise ValueError(f'shape of x must be [-1, 1, 1] for NEXT mode, '
                                 'current shape is {x.shape}')
            norm_h, _ = self.transformer.transform(x)

            states = predict_batched_state(
                norm_h,
                self.mesh_points,
                current_state=self._states[-1],
                current_field=self._history_h[-1],
            )

            norm_h = norm_h.reshape(-1, states.shape[-2])

        else:
            raise ValueError(f'mode:`{self.mode}` not accepted')

        # return values w/or w/o normalization
        if return_real:
            result = self.transformer.untransform(
                norm_h,
                self._predict_normalized_magnetization(
                    states, norm_h
                )
            )[1]

        else:
            result = self._predict_normalized_magnetization(
                states, norm_h
            )

        if self.mode == NEXT:
            return result.reshape(-1, 1, 1)
        else:
            return result

    def load_from_state_dict(self, state_dict):
        for ele in ['_history_h', '_history_m', '_states']:
            self.register_buffer(ele, state_dict.pop(ele))

        # load polynomial fit
        self.transformer = HysteresisTransform(
            None, None,
            self.polynomial_degree,
        )
        self.load_state_dict(state_dict, strict=False)

    @property
    def valid_domain(self):
        return self.transformer.get_valid_domain()

    @property
    def n_mesh_points(self):
        return len(self.mesh_points)

    @property
    def history_h(self):
        return self.transformer.untransform(self._history_h)[0].detach()

    @property
    def history_m(self):
        return self.transformer.untransform(
            self._history_h, self._history_m
        )[1].detach()
