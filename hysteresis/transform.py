import torch
from torch.nn import Module
from hysteresis.polynomial import Polynomial
from hysteresis.training import train_MSE


class HysteresisTransform(Module):
    _min_h = None
    _max_h = None
    _offset_m = None
    _scale_m = None
    _trained_m = False
    _trained_h = False

    def __init__(self, train_h, train_m=None, polynomial_degree=5,
                 polynomial_fit_iterations=3000):
        super(HysteresisTransform, self).__init__()
        self.polynomial_degree = polynomial_degree
        self.polynomial_fit_iterations = polynomial_fit_iterations
        if isinstance(train_m, torch.Tensor):
            self.update_all(train_h, train_m)
        else:
            self.update_h_normalize(train_h)

    def update_all(self, train_h, train_m):
        self.update_h_normalize(train_h)
        self.update_m_normalize(train_h, train_m)

    def update_fit(self, train_h, train_m):
        """ do polynomial fitting without normalizing train_h"""
        self.poly_fit = Polynomial(self.polynomial_degree)
        train_MSE(self.poly_fit, train_h, train_m, self.polynomial_fit_iterations)
        self.poly_fit.requires_grad_(False)

    def update_h_normalize(self, train_h):
        self._min_h = torch.min(train_h)
        self._max_h = torch.max(train_h)
        self._trained_h = True

    def update_m_normalize(self, train_h, train_m):
        self.update_fit(train_h, train_m)

        fit = self.poly_fit(train_h)
        m_subtracted = train_m - fit
        self._offset_m = torch.mean(m_subtracted)
        self._scale_m = torch.std(m_subtracted - self._offset_m)
        self._trained_m = True

    def _transform_h(self, h):
        if self._trained_h:
            return (h - self._min_h) / (self._max_h - self._min_h)
        else:
            raise RuntimeError('h transformation not trained yet')

    def _transform_m(self, h, m):
        if self._trained_m:
            fit = self.poly_fit(h)
            return (m - fit - self._offset_m) / self._scale_m
        else:
            raise RuntimeError('m transformation not trained yet')

    def transform(self, h, m=None):
        hn = self._transform_h(h)
        if isinstance(m, torch.Tensor):
            mn = self._transform_m(h, m)
        else:
            mn = None

        return hn, mn

    def _untransform_h(self, hn):
        if self._trained_h:
            return hn * (self._max_h - self._min_h) + self._min_h
        else:
            raise RuntimeError('h transformation not trained yet')

    def _untransform_m(self, hn, mn):
        if self._trained_m:
            fit = self.poly_fit(self._untransform_h(hn))
            return self._scale_m * mn + fit.reshape(hn.shape) + self._offset_m
        else:
            raise RuntimeError('m transformation not trained yet')

    def untransform(self, hn, mn=None):
        if torch.min(hn) < 0.0 or torch.max(hn) > 1.0:
            raise RuntimeWarning('input bounds of hn are outside normalization '
                                 'region, are you sure h is normalized?')

        h = self._untransform_h(hn)
        if isinstance(mn, torch.Tensor):
            m = self._untransform_m(hn, mn)
        else:
            m = None

        return h, m
