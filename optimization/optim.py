import torch

from hysteresis.hybrid import ExactHybridGP

from gpytorch.utils.errors import NotPSDError
from botorch import fit_gpytorch_model
from gpytorch import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from tqdm.notebook import trange


from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

def get_model(train_X, train_Y, use_hybrid = False, h_models = None):
    if use_hybrid:
        gpmodel = ExactHybridGP(
            train_X.clone().detach().double(),
            train_Y.clone().detach().flatten().double(), 
            h_models,
        )
        
        mll = ExactMarginalLogLikelihood(gpmodel.gp.likelihood, gpmodel)
        fit_gpytorch_model(mll)
        
    else:
        gpmodel = SingleTaskGP(
            train_X.clone().detach(),
            train_Y.clone().detach(),
            input_transform=Normalize(train_X.shape[-1]),
            outcome_transform=Standardize(1)
        )
        mll = ExactMarginalLogLikelihood(gpmodel.likelihood, gpmodel)
        fit_gpytorch_model(mll)
    return gpmodel

def optimize(
        accelerator_model,
        initial_beam_matrix,
        h_models,
        objective,
        initial_X=None,
        steps=50,
        use_hybrid = True,
        verbose=False
):
    iterations = steps

    if initial_X is None:
        # initialize with a couple of points
        train_X = torch.ones((3, 3)) * 0.25
        train_X[0] = train_X[0] * 0.0
        train_X[2] = torch.tensor((0.15,-0.36, 0.36))
    else:
        train_X = initial_X
        
    train_Y = torch.empty((len(train_X),1))

    for j in range(len(train_X)):
        accelerator_model.apply_fields({'q1': train_X[j, 0],
                                        'q2': train_X[j, 1],
                                        'q3': train_X[j, 2],})

        train_Y[j] = objective(
            accelerator_model.forward(initial_beam_matrix)
        )

    if verbose:
        print(train_X)
        print(train_Y)

    gpmodel = get_model(train_X, train_Y, use_hybrid, h_models)

    for i in range(iterations):
        UCB = UpperConfidenceBound(gpmodel, beta=2.0, maximize=False)

        if use_hybrid:
            gpmodel.next()
        
        bounds = torch.stack([-1.0 * torch.ones(3), torch.ones(3)])
        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        train_X = torch.cat((train_X, candidate))

        # apply candidate
        accelerator_model.apply_fields({'q1': candidate[0,0],
                                        'q2': candidate[0,1],
                                        'q3': candidate[0,2]})

        # make next measurement
        bs = objective(
            accelerator_model.forward(initial_beam_matrix)
        ).reshape(1, 1)
        train_Y = torch.cat((train_Y.clone(), bs.clone()))

        if verbose:
            print(torch.cat((candidate, bs), dim=1))


        # train new model
        try:
            gpmodel = get_model(train_X, train_Y, use_hybrid, h_models)
        except NotPSDError:
            print('handling training issues')

    return train_X, train_Y, accelerator_model, gpmodel