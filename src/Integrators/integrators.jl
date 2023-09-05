abstract type Integrator end

include("quadrature.jl")
include("monte_carlo.jl")
include("importance_sampling.jl")

export Integrator
export MCIntegrator, MonteCarlo, MCMC, MCSamples
export ISIntegrator, ISMC, ISMCMC, ISSamples, ISMixSamples
export QuadIntegrator, GaussQuadrature, gausslegendre