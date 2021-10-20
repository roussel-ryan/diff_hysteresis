import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
from pyro.nn import PyroModule, PyroSample

import utils


class BayesianHysteresis(PyroModule):
    def __init__(self, model, n):
        super(BayesianHysteresis, self).__init__()

        self.hysteresis_model = model
        # vector length
        vector_length = utils.get_upper_trainagle_size(n)

        # represent the hysterion density
        self.density = PyroSample(dist.Normal(0.0, 1.).expand(torch.Size([
            vector_length])).to_event(1))

        # represent the scale and offset with Normal distributions - priors assume
        # normalized output
        self.scale = PyroSample(dist.Normal(1.0, 1.0))
        self.offset = PyroSample(dist.Normal(0.0, 1.0))
        #self.noise = PyroSample(dist.Normal(-15.0, 10.0))

        self.train = True

    def forward(self, x, y=None):
        # set hysteresis model parameters to do calculation
        raw_vector = self.density.to(self.hysteresis_model.h_data)
        scale = torch.nn.Softplus()(self.scale)
        #noise = torch.nn.Softplus()(self.noise)

        # do prediction
        mean = \
            self.hysteresis_model.predict_magnetization(h=x,
                                                        raw_dens_vector=raw_vector,
                                                        scale=scale,
                                                        offset=self.offset)

        # condition on observations
        with pyro.plate('data', len(x)):
            pyro.sample('obs', dist.Normal(mean, 0.01), obs=y)
        return mean

    def posterior_predictive(self, x, n_samples=1):
        results = torch.empty(n_samples, (len(x)))
        param_loc = pyro.param('AutoMultivariateNormal.loc')
        param_scale_tril = pyro.param('AutoMultivariateNormal.scale_tril')
        samples = dist.MultivariateNormal(param_loc,
                                          scale_tril=param_scale_tril).sample([n_samples])
        samples = samples.to(self.hysteresis_model.h_data)
        for i in range(n_samples):
            # set hysteresis model parameters to do calculation
            raw_vector = samples[i, :-2]
            scale = torch.nn.Softplus()(samples[i, -2])
            offset = samples[i, -1]

            # do prediction
            mean = \
                self.hysteresis_model.predict_magnetization(h=x,
                                                            raw_dens_vector=raw_vector,
                                                            scale=scale,
                                                            offset=offset)
            results[i] = dist.Normal(mean, 0.01).sample()
        return results

