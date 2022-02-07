import matplotlib.pyplot as plt
import torch
import pytest

from hysteresis.transform import HysteresisTransform


class TestHysteresisTransform:
    def test_init(self):
        train_h = torch.linspace(-1.0, 5.0, 20)
        t = HysteresisTransform(train_h)
        assert torch.equal(t.domain, torch.tensor((-1.0, 5.0)))
        hn = t.transform(train_h)[0]
        assert torch.allclose(hn, torch.linspace(0, 1, 20))
        assert torch.allclose(train_h, t.untransform(hn)[0], rtol=0.001)

        train_m = 0.01 * train_h ** 3 - 0.5 * train_h ** 2 + train_h
        t2 = HysteresisTransform(train_h, train_m)
        hn, mn = t2.transform(train_h, train_m)
        assert torch.allclose(hn, torch.linspace(0, 1, 20))
        assert torch.allclose(train_h, t2.untransform(hn)[0], rtol=0.001)

        assert torch.allclose(train_m, t2.untransform(hn, mn)[1], rtol=0.01)
        assert torch.allclose(train_m, t2.get_fit(train_h), rtol=0.5)

    def test_fixed(self):
        t = HysteresisTransform(fixed_domain=torch.tensor((-10.0, 5.0)))
        assert torch.equal(t.domain, torch.tensor((-10.0, 5.0)))

        with pytest.raises(RuntimeError):
            t.domain = torch.tensor((0.0, 1.0))
