import torch
from torch.nn import Module
from hysteresis.polynomial import Polynomial
from hysteresis.training import train_MSE


class HysteresisTransform(Module):
    _min_h = None
    _max_h = None
    _min_m = None
    _max_m = None
    _offset_m = torch.zeros(1)
    _scale_m = torch.ones(1)

    def __init__(
        self, train_h, train_m=None, polynomial_degree=5, polynomial_fit_iterations=5000
    ):
        super(HysteresisTransform, self).__init__()
        self.polynomial_degree = polynomial_degree
        self.polynomial_fit_iterations = polynomial_fit_iterations

        if isinstance(train_m, torch.Tensor) and isinstance(train_h, torch.Tensor):
            self.update_all(train_h, train_m)
        elif isinstance(train_h, torch.Tensor):
            self.update_h_transform(train_h)
            self._poly_fit = Polynomial(self.polynomial_degree)
        else:
            self._poly_fit = Polynomial(self.polynomial_degree)

    def freeze(self):
        self._poly_fit.requires_grad_(False)

    def update_all(self, train_h, train_m):
        self.update_h_transform(train_h)
        self.update_m_transform(train_h, train_m)

    def get_fit(self, h):
        return self._unnorm_m(self._poly_fit(self._norm_h(h)))

    def update_fit(self, hn, mn):
        """do polynomial fitting on normalized train_h and train_m"""
        self._poly_fit = Polynomial(self.polynomial_degree)
        train_MSE(self._poly_fit, hn, mn, self.polynomial_fit_iterations)
        self._poly_fit.requires_grad_(False)

    def update_h_transform(self, train_h):
        self._min_h = torch.min(train_h)
        self._max_h = torch.max(train_h)

    def _norm_h(self, h):
        if self._max_h is None and self._min_h is None:
            return h
        else:
            return (h - self._min_h) / (self._max_h - self._min_h)

    def _unnorm_h(self, hn):
        if self._max_h is None and self._min_h is None:
            return hn
        else:
            return hn * (self._max_h - self._min_h) + self._min_h

    def _norm_m(self, m):
        if self._max_m is None and self._min_m is None:
            return m
        else:
            return (m - self._min_m) / (self._max_m - self._min_m)

    def _unnorm_m(self, mn):
        if self._max_m is None and self._min_m is None:
            return mn
        else:
            return mn * (self._max_m - self._min_m) + self._min_m

    def get_valid_domain(self):
        if self._max_h is None and self._min_h is None:
            return torch.tensor((0.0, 1.0))
        else:
            return torch.tensor((self._min_h, self._max_h))

    def update_m_transform(self, train_h, train_m):
        self._min_m = torch.min(train_m)
        self._max_m = torch.max(train_m)

        self.update_fit(self._norm_h(train_h), self._norm_m(train_m))

        fit = self._unnorm_m(self._poly_fit(self._norm_h(train_h)))
        m_subtracted = train_m - fit
        self._offset_m = torch.mean(m_subtracted)
        self._scale_m = torch.std(m_subtracted - self._offset_m)

    def _transform_h(self, h):
        return self._norm_h(h)

    def _transform_m(self, h, m):
        fit = self._unnorm_m(self._poly_fit(self._norm_h(h)))
        return (m - fit - self._offset_m) / self._scale_m

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
        return self._scale_m * mn + fit.reshape(hn.shape) + self._offset_m

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
