import pytest
import torch

from torchAccelerator.first_order import TorchDrift, TorchAccelerator


class TestAccelerator:
    def test_accelerator_calculate_transport(self):
        d1 = TorchDrift("d1", torch.tensor(1.0))
        d2 = TorchDrift("d2", torch.tensor(2.0))
        accel = TorchAccelerator([d1, d2])
        assert accel.calculate_transport()[-1, 0, 1] == torch.tensor(3.0)

        d3 = TorchDrift("d2", torch.tensor(2.0))
        with pytest.raises(RuntimeError):
            accel = TorchAccelerator([d1, d2, d3])

    def test_accelerator_forward(self):
        d1 = TorchDrift("d1", torch.tensor(1.0))
        d2 = TorchDrift("d2", torch.tensor(2.0))
        accel = TorchAccelerator([d1, d2])

        # test full beamline prop
        R = torch.eye(6)
        M = torch.eye(6)
        M[0, 1] = 3.0
        M[2, 3] = 3.0
        correct_result = M @ R @ M.T
        assert torch.equal(accel.forward(R), correct_result)

        # test partial beamline prop
        M = torch.eye(6)
        M[0, 1] = 1.0
        M[2, 3] = 1.0
        correct_result = M @ R @ M.T
        assert torch.equal(accel.forward(R, full=False)[1], correct_result)





