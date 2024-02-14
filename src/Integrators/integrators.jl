""" 
    Integrator

    A struct of abstract type Integrator computes the expectation of a function h(x, θ) with respect to an invariant measure p(x, θ).
"""
abstract type Integrator end

include("quadrature.jl")
include("monte_carlo.jl")
include("importance_sampling.jl")

export Integrator
export MCIntegrator, MonteCarlo, MCMC, MCSamples
export ISIntegrator, ISMC, ISMCMC, ISSamples, ISMixSamples
export QuadIntegrator, GaussQuadrature, gausslegendre