import torch

from hysteresis.torch_accelerator.first_order import TorchQuad


class TestQuad:
    def test_get_matrix(self):
        q = TorchQuad("Q1", torch.ones(1), torch.zeros(1))

        # test using forward
        res = q()
        assert res.shape == torch.Size([1, 6, 6])

        # test in single value mode
        res = q.get_matrix(torch.randn(1))
        assert res.shape == torch.Size([1, 6, 6])

        res = q.get_matrix(torch.tensor(1.0))
        print(res)
        assert res.shape == torch.Size([6, 6])

        # test in batched mode
        test_k = torch.randn(4, 5)

        # test to make sure rotation matrix is working
        pos_matrix = q.get_matrix(test_k)
        neg_matrix = q.get_matrix(-test_k)

        for p, n in zip(pos_matrix.reshape(-1, 6, 6), neg_matrix.reshape(-1, 6, 6)):
            for ii in range(2):
                for jj in range(2):
                    assert p[ii, jj] == n[ii + 2, jj + 2]
