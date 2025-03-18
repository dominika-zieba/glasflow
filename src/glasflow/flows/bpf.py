# -*- coding: utf-8 -*-
"""
Implementation of Neural Spline Flows.

See: https://arxiv.org/abs/1906.04032
"""

#from glasflow.nflows.transforms.coupling import PiecewiseBernsteinCouplingTransform
from glasflow.nflows.transforms.coupling import PiecewiseCouplingTransform

import numpy as np
import torch
import torch.nn.functional as F
from .coupling import CouplingFlow

from ..transforms.bernstein import bernstein_transform

class PiecewiseBernsteinCouplingTransform(PiecewiseCouplingTransform):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        bernstein_degree=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        log = False
    ):

        self.bernstein_degree = bernstein_degree 
        self.tails = tails
        self.tail_bound = tail_bound
        self.log = log


        if apply_unconditional_transform:
            raise NotImplementedError()
        
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        return self.bernstein_degree - 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unconstrained_alphas = transform_params

        if hasattr(self.transform_net, "hidden_features"):
            unconstrained_alphas /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unconstrained_alphas /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        
        if self.tails is None:
             bijector_fn = bernstein_transform
        else:
            raise NotImplementedError()

        return bijector_fn(
            inputs=inputs,
            unconstrained_alphas=unconstrained_alphas,
            inverse=inverse, log=self.log)


class CouplingBPF(CouplingFlow):
    """Implementation of Bernstein Polynomial Flows using a coupling transform.

    Supports use of a uniform distribution for the latent space, so far only. This
    automatically disables the tails and sets the bounds to [0, 1).

    Parameters
    ----------
    n_inputs : int
        Number of inputs
    n_transforms : int
        Number of transforms
    n_conditional_inputs: int
        Number of conditionals inputs
    n_neurons : int
        Number of neurons per residual block in each transform 
    n_blocks_per_transform : int
        Number of residual blocks per transform 
    batch_norm_within_blocks : bool
        Enable batch normalisation within each residual block
    batch_norm_between_transforms : bool
        Enable batch norm between transforms
    activation : function
        Activation function to use. Defaults to ReLU
    dropout_probability : float
        Amount of dropout to apply. 0 being no dropout and 1 being drop
        all connections
    linear_transform : str, {'permutation', 'lu', 'svd', None}
        Not implemented. Linear transform to apply before each coupling transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent space. Default is multivariate uniform. 
    mask : Union[torch.Tensor, list, numpy.ndarray]
        Mask or array of masks to use to construct the flow. If not specified,
        an alternating binary mask will be used.
    bernstein_degree : int
        Degree of Bernstein polynomial transform in each dimension.
    tail_type : {None, 'linear'}
        Not implemented. Type of tails to use outside the bounds on which the bernstein polynomials are
        defined.
    tail_bound : float
        Not implemented. Bound that defines the region over which the polynomials are defined.
        I.e. [-tail_bound, tail_bound]
    kwargs :
        Keyword arguments passed to the transform when is it initialised.
    """

    def __init__(
        self,
        n_inputs,
        n_transforms,
        n_conditional_inputs=None,
        n_neurons=32,
        n_blocks_per_transform=2,
        batch_norm_within_blocks=False,
        batch_norm_between_transforms=False,
        activation=F.relu,
        dropout_probability=0.0,
        linear_transform=None,
        distribution="uniform",
        mask = None,
        bernstein_degree=10,
        log = False,
        tail_type=None,
        tail_bound=1.0,
        **kwargs,
    ):
        
        transform_class = PiecewiseBernsteinCouplingTransform

        if distribution == "uniform":
            from ..distributions import MultivariateUniform

            tail_bound = 1.0
            tail_type = None
            distribution = MultivariateUniform(
                low=torch.Tensor(n_inputs * [0.0]),
                high=torch.Tensor(n_inputs * [1.0]),
            )
            batch_norm_between_transforms = False

        else:
            raise NotImplementedError()

        super().__init__(
            transform_class,
            n_inputs,
            n_transforms,
            n_conditional_inputs=n_conditional_inputs,
            n_neurons=n_neurons,
            n_blocks_per_transform=n_blocks_per_transform,
            batch_norm_within_blocks=batch_norm_within_blocks,
            batch_norm_between_transforms=batch_norm_between_transforms,
            activation=activation,
            dropout_probability=dropout_probability,
            linear_transform=linear_transform,
            distribution=distribution,
            mask=mask,
            bernstein_degree=bernstein_degree,
            log = log,
            tails=tail_type,
            tail_bound=tail_bound,
            **kwargs,
        )


#todo: initialise to identity transform..