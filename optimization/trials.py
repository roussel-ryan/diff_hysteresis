from copy import deepcopy
from hysteresis.base import BaseHysteresis
from torchAccelerator.hysteresis import HysteresisAccelerator, HysteresisQuad
from torchAccelerator.first_order import TorchDrift

import matplotlib.pyplot as plt
import torch
from multiprocessing import Pool
import numpy as np
import time

from optim import optimize
import logging

logging.basicConfig(level=logging.INFO)


def run_trial(initial_X, scale, verbose=False, use_hybrid=False, path='', fname=''):
    def objective(R):
        return torch.abs(torch.sqrt(R[0, 0]) - 8e-3) + torch.abs(
            torch.sqrt(R[2, 2]) - 8e-3)

    objective_func = objective
    # hysteresis model
    H = BaseHysteresis(
        trainable=False,
        fixed_domain=torch.tensor((-1.0, 1.0))
    )

    m_sat = 400.0
    H.scale = scale
    H.slope = (m_sat - H.scale) * 2.0
    H.offset = -H.slope / 2.0

    # accelerator model
    hmodels = [deepcopy(H), deepcopy(H), deepcopy(H)]

    # define quadrupoles
    q1 = HysteresisQuad("q1", torch.tensor(0.01), hmodels[0])
    d1 = TorchDrift("d1", torch.tensor(1.0))
    d2 = TorchDrift("d2", torch.tensor(1.0))
    q2 = HysteresisQuad("q2", torch.tensor(0.01), hmodels[1])
    d3 = TorchDrift("d3", torch.tensor(1.0))
    q3 = HysteresisQuad("q3", torch.tensor(0.01), hmodels[2])

    HA = HysteresisAccelerator([q1, d1, q2, d2, q3, d3])
    HA.current()

    init_beam_matrix = torch.eye(6) * 1.0e-8

    # set x_rms beam size to 1 mm and rms divergence to 0.1 mrad
    init_beam_matrix[0, 0] = 5.0e-3 ** 2
    init_beam_matrix[1, 1] = 1.0e-4 ** 2
    init_beam_matrix[2, 2] = 5.0e-3 ** 2
    init_beam_matrix[3, 3] = 1.0e-4 ** 2
    R = init_beam_matrix

    logging.info('running optimization')
    points, objective_vals, accelerator_model, gp_hybrid = optimize(
        deepcopy(HA),
        R,
        [deepcopy(H), deepcopy(H), deepcopy(H)],
        objective_func,
        initial_X=initial_X,
        steps=10,
        use_hybrid=use_hybrid,
        verbose=verbose
    )

    # plot HA hysteresis
    test_applied_fields = torch.cat(
        (torch.linspace(-1, 1, 11), torch.flipud(torch.linspace(-1, 1, 11)))
    )

    HA.elements['q1'].hysteresis_model.regression()
    M = HA.elements['q1'].hysteresis_model(test_applied_fields, return_real=True).detach()

    fig, ax = plt.subplots()
    ax.plot(test_applied_fields, M)
    plt.show()

    results = torch.cat((points, objective_vals), dim=1)
    #torch.save(results, f'{path}{fname}.pt')


if __name__ == '__main__':
    logging.info('starting')
    path = '/global/cfs/cdirs/m669/rroussel/archive/hysteresis/'
    n_samples = 3
    inputs = torch.load('random_inputs.pt')[:n_samples]


    def run_h_on(idx, x):
        return run_trial(x, 400., False, fname=f'normal_hyst_on_{idx}', path=path)


    def run_h_off(idx, x):
        return run_trial(x, 0., False, fname=f'normal_hyst_off_{idx}', path=path)


    def run_h_on_hybrid(idx, x):
        return run_trial(x, 0.5, use_hybrid=True, fname=f'hybrid_hyst_on_{idx}',
                         path=path)


    def run_h_off_hybrid(idx, x):
        return run_trial(x, 0.005, use_hybrid=True, fname=f'hybrid_hyst_off_{idx}',
                         path=path)


    run_h_on(0, inputs[0])
    run_h_off(0, inputs[0])
