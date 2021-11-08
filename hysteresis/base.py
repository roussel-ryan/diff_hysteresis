import torch
from torch.nn import Module, Parameter
from torch import Tensor
from typing import Dict, Callable
from .meshing import create_triangle_mesh, default_mesh_size


def switch(x, s, T=1e-5):
    """
    switch function  to change between negative and positive states
    For x > s the function rapidly takes a value of 1.0
    for x < s the function rapidly takes a value of -1.0

    Temperature parameter (T) determines how quickly this happens, set T << 1 to get
    heavyside behavior.
    """
    return 2.0 / (1.0 + torch.exp(-(x - s) / T)) - 1


class TorchHysteresis(Module):
    states = None
    _old_h = None
    _old_states = None

    def __init__(self,
                 h_train: Tensor = None,
                 mesh_scale: float = 1.0,
                 trainable: bool = True,
                 tkwargs: Dict = None,
                 mesh_density_function: Callable = None):
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
        self.tkwargs = tkwargs or {}
        self.tkwargs.update({'dtype': torch.double, 'device': 'cpu'})

        self.trainable = trainable

        # generate mesh grid on 2D normalized domain [[0,1],[0,1]]
        self.mesh_scale = mesh_scale
        self.mesh_points = torch.tensor(
            create_triangle_mesh(
                mesh_scale,
                mesh_density_function
            ),
            **self.tkwargs
        )

        # generate hysterion density vector, offset and scale parameters
        hyst_density = torch.ones(self.mesh_points.shape[0], **self.tkwargs)
        offset = torch.zeros(1, **self.tkwargs)
        scale = torch.ones(1, **self.tkwargs)

        # if trainable register the parameters
        if self.trainable:
            self.register_parameter('_raw_hyst_density', Parameter(hyst_density))
            self.register_parameter('offset', Parameter(offset))
            self.register_parameter('scale', Parameter(scale))

        else:
            self.register_buffer('_raw_hyst_density', Parameter(hyst_density))
            self.register_buffer('offset', Parameter(offset))
            self.register_buffer('scale', Parameter(scale))

        if h_train is not None:
            # normalize training data
            self.h_min = torch.min(h_train)
            self.h_max = torch.max(h_train)
            self.h_train = self.normalize_h(h_train.to(**self.tkwargs))

            self.update_states(self.h_train)
        else:
            self.h_min = 0.0
            self.h_max = 1.0
            self.h_train = None

    @property
    def hysterion_density(self):
        return torch.nn.Softplus()(self._raw_hyst_density)

    @hysterion_density.setter
    def hysterion_density(self, value):
        self._raw_hyst_density = Parameter(torch.log(torch.exp(value) - 1).to(
            **self.tkwargs))
        if not self.trainable:
            self._raw_hyst_density.requires_grad = False

    def get_mesh_size(self, x, y):
        return default_mesh_size(x, y, self.mesh_scale)

    def normalize_h(self, h):
        return (h - self.h_min) / (self.h_max - self.h_min)

    def unnormalize_h(self, h):
        return h * (self.h_max - self.h_min) + self.h_min

    def update_states(self, h: torch.Tensor):
        self.states, _ = self.get_states(h)

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

        This calculation can be expensive, so we skip recalcuation until if h !=
        current_h

        See: https://www.wolframcloud.com/objects/demonstrations
        /TheDiscretePreisachModelOfHysteresis-source.nb

        Parameters
        ----------
        h : torch.Tensor,
            The applied magnetic field H_1:t={H_1, ... ,H_t}, where
            t represents each time step.

        Raises
        ------
        ValueError
            If n is negative.
        """

        n_mesh_points = self.mesh_points.shape[0]

        # note running into machine precision issues when normalizing
        if not (torch.all(h <= 1.0 + 1e-7) and torch.all(h >= 0.0 - 1e-7)):
            raise ValueError('model not valid for applied fields outside unit domain')

        # starts off off
        hs = torch.cat((torch.zeros(1, **self.tkwargs), h))  # H_0=-t, negative

        # list of hysteresis states with initial state set
        states = torch.empty((len(hs), n_mesh_points), **self.tkwargs)
        states[0] = torch.ones(n_mesh_points, **self.tkwargs) * -1.0

        # flag to track if we need to recalculate states
        # if there are no saved states then we need to calculate_states
        if self._old_h is None and self._old_states is None:
            calculate_states = True
        else:
            calculate_states = False

        # loop through the states
        n_calcs = 0
        for i in range(1, len(hs)):
            # check if state should be recalculated
            if not calculate_states:
                # if the two values of h do NOT match then we need to recalc the rest
                # of the states
                try:
                    if not torch.isclose(hs[i], self._old_h[i]):
                        calculate_states = True
                except IndexError:
                    calculate_states = True

            # if we need to calculate the state do so - if not grab the old state
            if calculate_states:
                n_calcs += 1
                if hs[i] > hs[i - 1]:
                    # if the new applied field is greater than the old one, sweep up to
                    # new applied field
                    states[i] = torch.where(self.mesh_points[:, 1] <= hs[i],
                                            torch.tensor(1.0, **self.tkwargs),
                                            states[i - 1])

                elif hs[i] < hs[i - 1]:
                    # if the new applied field is less than the old one, sweep left to
                    # new applied field
                    states[i] = torch.where(self.mesh_points[:, 0] >= hs[i],
                                            torch.tensor(-1.0, **self.tkwargs),
                                            states[i - 1])
                else:
                    states[i] = states[i - 1]
            else:
                states[i] = self._old_states[i]

        # preserve calculation for future use
        self._old_h = hs
        self._old_states = states
        if n_calcs:
            print(f'calculated {n_calcs} states')

        return states, n_calcs

    def get_negative_saturation(self,
                                density_vector: Tensor = None,
                                scale: Tensor = None,
                                offset: Tensor = None
                                ):
        """ get the nagetive stautration magnetization given a set of parameters
        internal or specified via argument"""
        # get the density vector from the raw version
        if density_vector is None:
            hyst_density_vector = torch.nn.Softplus()(self._raw_hyst_density)
        else:
            hyst_density_vector = torch.nn.Softplus()(density_vector)

        s = scale if scale is not None else self.scale
        o = offset if offset is not None else self.offset

        neg_sat_state = torch.ones((1, self.mesh_points.shape[0]), **self.tkwargs) * \
                        -1.0
        m = torch.sum(hyst_density_vector * neg_sat_state, dim=-1) \
            / len(hyst_density_vector)
        return s * m + o

    def predict_magnetization(self,
                              h: Tensor = None,
                              h_new: Tensor = None,
                              density_vector: Tensor = None,
                              scale: Tensor = None,
                              offset: Tensor = None) -> Tensor:
        """
        Predict the magnetization using the Preisach model

        Parameters
        ----------
        h : torch.Tensor, optional
            Sequence of applied fields that the model will calculate the
            magnetization at. If None or h == h_train then hysteresis states are not
            recalculated. Cannot be specified at the same time as `h_new`. Fields
            should not be normalized.

        h_new : torch.Tensor, optional
            Sequence of applied fields that will be appended to the training data to
            reduce recalculation of hysterion states. Cannot be specified at the same
            time as `h`. Fields should not be normalized.

        density_vector : torch.Tensor, optional
            Fixed density_vector to be used in magnetization prediction. By default
            the predicition uses the density vector stored internally in the model.

        scale : torch.Tensor, optional
            Fixed scale to be used in magnetization prediction. By default
            the predicition uses the scale stored internally in the model.

        offset : torch.Tensor, optional
            Fixed offset to be used in magnetization prediction. By default
            the predicition uses the offset stored internally in the model.

        Returns
        -------
        magnetization : torch.Tensor
            Magnetization as a function of applied field
            - M(h_train) if h, h_new are not specified
            - M(h) if h is specified
            - M({h_train, h_new}) if h_new is specified

        """

        assert not (h is not None and h_new is not None), 'cannot specify both h and ' \
                                                          'h_new'

        # get the density vector from the raw version
        if density_vector is None:
            hyst_density_vector = torch.nn.Softplus()(self._raw_hyst_density)
        else:
            hyst_density_vector = torch.nn.Softplus()(density_vector)

        s = scale if scale is not None else self.scale
        o = offset if offset is not None else self.offset

        # get the states based on h
        if h_new is not None:
            normed_h_new = self.normalize_h(h_new)
            if self.h_train is None:
                normed_h = normed_h_new
            else:
                normed_h = torch.cat((self.h_train, normed_h_new))
            states, _ = self.get_states(normed_h)
        elif h is not None:
            normed_h = self.normalize_h(h)
            states, _ = self.get_states(normed_h)

        else:
            # if no states have been calculated
            if self.states is not None:
                states = self.states
            else:
                return self.get_negative_saturation(density_vector, scale, offset)[0]

        m = torch.sum(hyst_density_vector * states[1:], dim=-1) \
            / len(hyst_density_vector)
        return s * m + o
