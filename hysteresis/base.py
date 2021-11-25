import torch
from torch.nn import Module, Parameter
from torch import Tensor
from typing import Dict, Callable
from .meshing import create_triangle_mesh, default_mesh_size
from .states import get_states, predict_batched_state


class TorchHysteresis(Module):
    states = None
    _old_h = None
    _old_states = None

    def __init__(
            self,
            h_train: Tensor = None,
            mesh_scale: float = 1.0,
            temp: float = 1e-3,
            trainable: bool = True,
            tkwargs: Dict = None,
            mesh_density_function: Callable = None,
            h_min: float = 0.0,
            h_max: float = 1.0,
    ):
        """
        Torch module used to numerically calculate hysteresis using a
        Preisach hysteresis model while preserving gradient information.

        Parameters
        ----------
        h_train : torch.Tensor
            A 1 x d-dim tensor that denotes a sequence of applied fields at arbitrary
            time steps.

        mesh_scale : float, default = 1.0
            Scaling of triangular mesh used to discretize beta-alpha hysterion
            density space. Default value results in approximately 110 grid points.

        trainable : bool, default = True
            If True, register the hysterion density values, bulk offset and bulk
            scale at trainable torch vectors.

        tkwargs : Dict, optional
            Modifies the data type and device of all tensors used internally.
            Default data type is torch.double and default device is 'cpu'.

        """

        super(TorchHysteresis, self).__init__()
        self.temp = temp
        self.tkwargs = tkwargs or {}
        self.tkwargs.update({"dtype": torch.double, "device": "cpu"})

        self.trainable = trainable

        # generate mesh grid on 2D normalized domain [[0,1],[0,1]]
        self.mesh_scale = mesh_scale
        self.mesh_points = torch.tensor(
            create_triangle_mesh(mesh_scale, mesh_density_function), **self.tkwargs
        )

        # initialize protected applied fields tensor
        self._applied_fields = torch.zeros(1, **self.tkwargs)
        self.states = torch.ones((1, self.mesh_points.shape[0]), **self.tkwargs) * -1.0

        # generate hysterion density vector, offset and scale parameters
        hyst_density = torch.ones(self.mesh_points.shape[0], **self.tkwargs)
        offset = torch.zeros(1, **self.tkwargs)
        scale = torch.ones(1, **self.tkwargs)

        # if trainable register the parameters
        if self.trainable:
            self.register_parameter("_raw_hyst_density", Parameter(hyst_density))
            self.register_parameter("offset", Parameter(offset))
            self.register_parameter("scale", Parameter(scale))

        else:
            # self.register_buffer("_raw_hyst_density", Parameter(hyst_density))
            # self.register_buffer("offset", Parameter(offset))
            # self.register_buffer("scale", Parameter(scale))
            self._raw_hyst_density = hyst_density
            self.offset = offset
            self.scale = scale

        if h_train is not None:
            # normalize training data
            self.h_min = torch.min(h_train)
            self.h_max = torch.max(h_train)
            self.applied_fields = h_train

        else:
            self.h_min = h_min
            self.h_max = h_max
            self.applied_fields = torch.zeros(1)

    @property
    def hysterion_density(self):
        return torch.nn.Softplus()(self._raw_hyst_density)

    @hysterion_density.setter
    def hysterion_density(self, value: Tensor):
        if self.trainable:
            self._raw_hyst_density = Parameter(
                torch.log(torch.exp(value.clone()) - 1).to(**self.tkwargs)
            )
        else:
            self._raw_hyst_density = torch.log(torch.exp(value.clone()) - 1).to(
                **self.tkwargs
            )

    @property
    def applied_fields(self):
        return self.unnormalize_h(self._applied_fields)

    @applied_fields.setter
    def applied_fields(self, value: Tensor):
        new_applied_fields = self.normalize_h(value.to(**self.tkwargs))
        if not torch.equal(new_applied_fields, self._applied_fields):
            self._applied_fields = new_applied_fields.clone()
            self.update_states(self._applied_fields)

    def get_mesh_size(self, x, y):
        return default_mesh_size(x, y, self.mesh_scale)

    def normalize_h(self, h):
        return (h - self.h_min) / (self.h_max - self.h_min)

    def unnormalize_h(self, h):
        return h * (self.h_max - self.h_min) + self.h_min

    def update_states(self, h: torch.Tensor):
        self.states = get_states(
            h, self.mesh_points, tkwargs=self.tkwargs, temp=self.temp
        )

    def _get_model_parameters(self, density_vector, scale, offset):
        # get the density vector from the raw version
        if not isinstance(density_vector, torch.Tensor):
            hyst_density_vector = torch.nn.Softplus()(self._raw_hyst_density)
        else:
            hyst_density_vector = torch.nn.Softplus()(density_vector)

        s = scale if isinstance(scale, torch.Tensor) else self.scale
        o = offset if isinstance(offset, torch.Tensor) else self.offset
        return hyst_density_vector, s, o

    def get_negative_saturation(
            self, density_vector: Tensor = None, scale: Tensor = None,
            offset: Tensor = None
    ):
        """get the nagetive stautration magnetization given a set of parameters
        internal or specified via argument"""
        # get the density vector from the raw version
        if density_vector is None:
            hyst_density_vector = torch.nn.Softplus()(self._raw_hyst_density)
        else:
            hyst_density_vector = torch.nn.Softplus()(density_vector)

        s = scale if scale is not None else self.scale
        o = offset if offset is not None else self.offset

        neg_sat_state = (
                torch.ones((1, self.mesh_points.shape[0]), **self.tkwargs) * -1.0
        )
        m = torch.sum(hyst_density_vector * neg_sat_state, dim=-1) / len(
            hyst_density_vector
        )
        return s * m + o

    def predict_magnetization_from_applied_fields(
            self,
            density_vector: Tensor = None,
            scale: Tensor = None,
            offset: Tensor = None,
    ):
        hyst_vec, s, o = self._get_model_parameters(density_vector, scale, offset)
        if isinstance(self.states, torch.Tensor):
            states = self.states

            # do numerical integration
            m = torch.sum(hyst_vec * states, dim=-1) / len(
                hyst_vec
            )
            return s * m + o

        else:
            return self.get_negative_saturation(density_vector, scale, offset)[0]

    def predict_magnetization_future(
            self,
            h: Tensor,
            density_vector: Tensor = None,
            scale: Tensor = None,
            offset: Tensor = None,
    ):
        hyst_vec, s, o = self._get_model_parameters(density_vector, scale, offset)

        # return predicted magnetization for each element of h_new
        normed_h_new = self.normalize_h(h.reshape(-1, 1, 1)).to(**self.tkwargs)

        states = predict_batched_state(
            normed_h_new,
            self.mesh_points,
            self.states[-1],
            self._applied_fields[-1]
        )

        # do numerical integration
        m = torch.sum(hyst_vec * states, dim=-1) / len(hyst_vec)
        return s * m + o
