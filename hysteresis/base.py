import gpytorch.constraints
import torch
from torch.nn import Parameter
from gpytorch import Module
from torch import Tensor
from typing import Dict, Callable
from .meshing import create_triangle_mesh, default_mesh_size
from .states import get_states, predict_batched_state
from .transform import HysteresisTransform
from .modes import ModeModule, REGRESSION, NEXT, FUTURE, FITTING, CURRENT

import logging

logger = logging.getLogger(__name__)


class HysteresisError(Exception):
    pass


class BaseHysteresis(Module, ModeModule):
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
            fixed_domain: Tensor = None,

    ):
        super(BaseHysteresis, self).__init__()

        self.tkwargs = tkwargs or {}
        self.tkwargs.update({"dtype": torch.double, "device": "cpu"})

        # initialize with empty transformer
        self.transformer = HysteresisTransform(fixed_domain=fixed_domain)

        self.trainable = trainable

        # generate mesh grid on 2D normalized domain [[0,1],[0,1]]
        self.temp = temp
        self.mesh_scale = mesh_scale
        self.mesh_points = torch.tensor(
            create_triangle_mesh(mesh_scale, mesh_density_function), **self.tkwargs
        )

        # initialize trainable parameters
        density = torch.zeros(len(self.mesh_points))
        param_vals = [density, torch.zeros(1), torch.zeros(1), torch.zeros(1)]
        param_names = ["raw_hysterion_density", "raw_offset", "raw_scale", "raw_slope"]
        param_constraints = [
            gpytorch.constraints.Interval(0.0, 1.0),
            gpytorch.constraints.Interval(-2000.0, 2000.0),
            gpytorch.constraints.Interval(-2000.0, 2000.0),
            gpytorch.constraints.Interval(-2000.0, 2000.0),
        ]

        for param_name, param_val, param_constraint in zip(
                param_names, param_vals, param_constraints
        ):
            self.register_parameter(param_name, Parameter(param_val))
            self.register_constraint(param_name, param_constraint)

            if not self.trainable:
                getattr(self, param_name).requires_grad = False

        # set initial values for linear parameters
        self.offset = torch.zeros(1)
        self.scale = torch.zeros(1)
        self.slope = torch.ones(1)

        # create initial transformer object
        self.polynomial_degree = polynomial_degree
        self.polynomial_fit_iterations = polynomial_fit_iterations
        self._fixed_domain = fixed_domain

        # if data is specified then set the history data and train transformer
        if isinstance(train_h, Tensor):
            self.set_history(train_h, train_m)

        # freeze transformer if not trainable model
        if not self.trainable:
            self.transformer.freeze()

    def set_history(
            self,
            history_h,
            history_m=None
    ):
        """set historical state values and recalculate hysterion states"""
        if self.trainable:
            self.transformer = HysteresisTransform(
                history_h,
                history_m,
                self._fixed_domain,
                self.polynomial_degree,
                self.polynomial_fit_iterations
            )

        if isinstance(history_h, Tensor):
            history_h = history_h.to(**self.tkwargs)
            if len(history_h.shape) != 1:
                raise ValueError('history_h must be a 1D tensor')

        if isinstance(history_m, Tensor):
            history_m = history_m.to(**self.tkwargs)
            if len(history_h.shape) != 1:
                raise ValueError('history_m must be a 1D tensor')

            if torch.equal(history_h, history_m):
                raise RuntimeError("train h and train m cannot be equal")

        _history_h, _history_m = self.transformer.transform(history_h, history_m)
        self._update_h_history_buffer(_history_h)

        if not isinstance(_history_m, Tensor):
            old_mode = self.mode
            self.regression()
            _history_m = self.forward(history_h.detach())
            self.mode = old_mode
        self.register_buffer("_history_m", _history_m.detach())

    def _update_h_history_buffer(self, norm_history_h):
        self.register_buffer("_history_h", norm_history_h.detach())

        # recalculate states
        _states = get_states(self._history_h, self.mesh_points, temp=self.temp)
        self.register_buffer("_states", _states)

    def apply_field(self, h):
        """
        updates magnet history and recalculates 
        """
        h = torch.atleast_1d(h)
        self._check_inside_valid_domain(h)
        if hasattr(self, '_history_h'):
            _history_h = torch.cat(
                (self._history_h, self.transformer.transform(h)[0])
            ).detach()
        else:
            _history_h = self.transformer.transform(h)[0].to(**self.tkwargs)
        self._update_h_history_buffer(_history_h)

    def _predict_normalized_magnetization(self, states, h):
        m = torch.sum(self.hysterion_density * states, dim=-1) / torch.sum(
            self.hysterion_density
        )
        return self.scale * m.reshape(h.shape) + h * self.slope + self.offset

    def get_negative_saturation(self):
        return self.transformer.untransform(torch.zeros(1), -self.scale + self.offset)[
            1
        ].to(**self.tkwargs)

    def forward(self, x: Tensor = None, return_real=False):
        if isinstance(x, Tensor):
            x = x.to(**self.tkwargs)
            self._check_inside_valid_domain(x)
        else:
            if self.mode != CURRENT:
                raise HysteresisError('must specify field when not using CURRENT mode')

        # get current state/field if available
        if hasattr(self, "history_h"):
            current_fld = self._history_h[-1]
            current_state = self._states[-1]
        else:
            current_state = None
            current_fld = None

        if self.mode == FITTING:
            if not hasattr(self, "history_h"):
                raise RuntimeError(
                    "no training data supplied to do fitting! Try "
                    "using FUTURE mode instead OR set data using "
                    "set_history()"
                )

            if self._history_h.shape != self._history_m.shape:
                raise HysteresisError("history datasets must match shape for fitting")

            if not torch.allclose(x,
                                  self.transformer.untransform(self._history_h)[0].to(
                                      x)):
                raise HysteresisError(
                    "must do regression on history fields if in FITTING mode")
            states = self._states
            norm_h = self._history_h

        elif self._mode == REGRESSION:
            norm_h, _ = self.transformer.transform(x)
            states = get_states(
                norm_h, self.mesh_points, tkwargs=self.tkwargs, temp=self.temp
            )

        elif self.mode == CURRENT:
            if not hasattr(self, "history_h"):
                raise HysteresisError(
                    "no history data to determine current state! Try "
                    "using FUTURE mode instead OR set data using "
                    "set_history()/apply_field()"
                )
            states = current_state.unsqueeze(0)
            norm_h = current_fld.unsqueeze(0)

        elif self.mode == FUTURE:
            if len(x.shape) != 1:
                raise ValueError("input must be 1D for FUTURE mode")

            norm_h, _ = self.transformer.transform(x)
            states = get_states(
                norm_h,
                self.mesh_points,
                current_state=current_state,
                current_field=current_fld,
                tkwargs=self.tkwargs,
                temp=self.temp,
            )

        elif self.mode == NEXT:
            norm_h, _ = self.transformer.transform(x)

            states = predict_batched_state(
                norm_h,
                self.mesh_points,
                current_state=current_state,
                current_field=current_fld,
                tkwargs=self.tkwargs,
                temp=self.temp,
            )

        else:
            raise ValueError(f"mode:`{self.mode}` not accepted")

        # return values w/or w/o normalization
        if return_real:
            result = self.transformer.untransform(
                norm_h, self._predict_normalized_magnetization(states, norm_h)
            )[1]

        else:
            result = self._predict_normalized_magnetization(states, norm_h)
        return result

    def _check_inside_valid_domain(self, values):
        machine_error = 1e-4
        if torch.any(values < self.valid_domain[0] - machine_error) or torch.any(
                values > self.valid_domain[1] + machine_error
        ):
            raise HysteresisError(
                f"Argument values are not inside valid domain ("
                f"{list(self.valid_domain)}) for this model! Offending tensor is {values}"
            )

    def reconstruct_hysterion_density(self):
        return self.hysterion_density

    def reset_history(self):
        del self._history_h
        del self._history_m

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        for param in self.parameters(recurse=True):
            param.requires_grad = value
        if not value:
            self.transformer.freeze()

    @property
    def fixed_domain(self):
        return self._fixed_domain is not None

    @property
    def valid_domain(self):
        return self.transformer.domain

    @property
    def n_mesh_points(self):
        return len(self.mesh_points)

    @property
    def history_h(self):
        return self.transformer.untransform(self._history_h)[0].detach()

    @property
    def history_m(self):
        return self.transformer.untransform(self._history_h, self._history_m)[
            1
        ].detach()

    @property
    def hysterion_density(self):
        return self.raw_hysterion_density_constraint.transform(
            self.raw_hysterion_density
        )

    @hysterion_density.setter
    def hysterion_density(self, value: Tensor):
        self.initialize(
            raw_hysterion_density=self.raw_hysterion_density_constraint.inverse_transform(
                value
            )
        )

    @property
    def offset(self):
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: Tensor):
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    @property
    def scale(self):
        return self.raw_scale_constraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value: Tensor):
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))

    @property
    def slope(self):
        return self.raw_slope_constraint.transform(self.raw_slope)

    @slope.setter
    def slope(self, value: Tensor):
        self.initialize(raw_slope=self.raw_slope_constraint.inverse_transform(value))
