import torch
from torch.nn import Module
from hysteresis.polynomial import Polynomial
from hysteresis.training import train_MSE


class HysteresisTransform(Module):
    _min_m = None
    _max_m = None
    offset_m = torch.zeros(1)
    scale_m = torch.ones(1)
    _fixed_domain = False
    _domain = torch.tensor((0.0, 1.0))
    _mrange = torch.tensor((0.0, 1.0))

    def __init__(
        self,
        train_h=None,
        train_m=None,
        fixed_domain=None,
        polynomial_degree=5,
        polynomial_fit_iterations=5000,
    ):
        super(HysteresisTransform, self).__init__()
        self.polynomial_degree = polynomial_degree
        self.polynomial_fit_iterations = polynomial_fit_iterations

        # set fixed domain if specified
        if isinstance(fixed_domain, torch.Tensor):
            self.set_fixed_domain(fixed_domain)

        if isinstance(train_m, torch.Tensor) and isinstance(train_h, torch.Tensor):
            self.update_all(train_h, train_m)
        elif isinstance(train_h, torch.Tensor):
            self.update_h_transform(train_h)
            self._poly_fit = Polynomial(self.polynomial_degree)
        else:
            self._poly_fit = Polynomial(self.polynomial_degree)

    def set_fixed_domain(self, domain):
        self._domain = domain
        self._fixed_domain = True

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        if self._fixed_domain:
            raise RuntimeError("cannot set new domain when fixed domain is specified!")
        else:
            if value.shape != torch.Size([2]) or (value[1] < value[0]):
                raise RuntimeError("domain value misspecified")
        self._domain = value

    @property
    def domain_width(self):
        return self.domain[1] - self.domain[0]

    @property
    def mrange(self):
        return self._mrange

    @mrange.setter
    def mrange(self, value):
        if value.shape != torch.Size([2]) or (value[1] < value[0]):
            raise RuntimeError("domain value misspecified")
        self._mrange = value

    @property
    def mrange_width(self):
        return self.mrange[1] - self.mrange[0]

    def freeze(self):
        self._poly_fit.requires_grad_(False)

    def update_all(self, train_h, train_m):
        self.update_h_transform(train_h)
        self.update_m_transform(train_h, train_m)

    def get_fit(self, h):
        return self._unnorm_m(self._poly_fit(self._norm_h(h)))

    def get_fit_grad(self, h):
        h_copy = h.clone()
        h_copy.requires_grad = True

        out = self._unnorm_m(self._poly_fit(self._norm_h(h_copy)))
        out.backward(torch.ones_like(h_copy))
        return h_copy.grad

    def update_fit(self, hn, mn):
        """do polynomial fitting on normalized train_h and train_m"""
        self._poly_fit = Polynomial(self.polynomial_degree)
        train_MSE(self._poly_fit, hn, mn, self.polynomial_fit_iterations)
        self._poly_fit.requires_grad_(False)

    def update_h_transform(self, train_h):
        if not self._fixed_domain:
            self.domain = torch.tensor((torch.min(train_h), torch.max(train_h)))

    def _norm_h(self, h):
        return (h - self.domain[0]) / self.domain_width

    def _unnorm_h(self, hn):
        return hn * self.domain_width + self.domain[0]

    def _norm_m(self, m):
        return (m - self.mrange[0]) / self.mrange_width

    def _unnorm_m(self, mn):
        return mn * self.mrange_width + self.mrange[0]

    def update_m_transform(self, train_h, train_m):
        self.mrange = torch.tensor((min(train_m), max(train_h)))
        self.update_fit(self._norm_h(train_h), self._norm_m(train_m))

        fit = self._unnorm_m(self._poly_fit(self._norm_h(train_h)))
        m_subtracted = train_m - fit
        self.offset_m = torch.mean(m_subtracted)
        self.scale_m = torch.std(m_subtracted - self.offset_m)

    def _transform_h(self, h):
        return self._norm_h(h)

    def _transform_m(self, h, m):
        fit = self._unnorm_m(self._poly_fit(self._norm_h(h)))
        return (m - fit - self.offset_m) / self.scale_m

    def transform(self, h, m=None):
        hn = self._transform_h(h)
        if isinstance(m, torch.Tensor):
            mn = self._transform_m(h, m)
        else:
            mn = None

        return hn, mn

    def _untransform_h(self, hn):
        return self._unnorm_h(hn)

    def _untransform_m(self, hn, mn):
        fit = self._unnorm_m(self._poly_fit(hn))
        return self.scale_m * mn + fit.reshape(hn.shape) + self.offset_m

    def untransform(self, hn, mn=None):
        # verify the inputs are in the normalized region within some machine epsilon
        epsilon = 1e-6
        if torch.min(hn) + epsilon < 0.0 or torch.max(hn) - epsilon > 1.0:
            raise RuntimeWarning(
                "input bounds of hn are outside normalization "
                "region, are you sure h is normalized?"
            )

        h = self._untransform_h(hn)
        if isinstance(mn, torch.Tensor):
            m = self._untransform_m(hn, mn)
        else:
            m = None

        return h, m
