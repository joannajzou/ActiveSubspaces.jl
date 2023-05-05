abstract type ISIntegrator <: Integrator end

"""
struct ImportanceSampMC <: ISIntegrator

Defines the struct containing parameters for Importance Sampling (IS) integration using MC samples from a biasing distribution.
This method is implemented when the biasing distribution g can be analytically sampled using rand().

# Arguments
- `g :: Distribution`      : biasing distribution
- `n :: Int`               : number of samples
"""
struct ImportanceSampMC <: ISIntegrator
    g :: Distribution
    n :: Int
end

"""
struct ImportanceSampMCMC <: ISIntegrator

Defines the struct containing parameters for Importance Sampling (IS) integration using MCMC samples from a biasing distribution.
This method is implemented when the biasing distribution g cannot be analytically sampled.

# Arguments
- `g :: Distribution`      : biasing distribution
- `n :: Int`               : number of samples
- `sampler :: Sampler`     : type of sampler (see `Sampler`)
- `ρ0 :: Distribution`     : prior distribution of the state
"""
struct ImportanceSampMC <: ISIntegrator
    g :: Distribution
    n :: Int
    sampler :: Sampler
    ρ0 :: Distribution
end


"""
struct ImportanceSampSamples <: ISIntegrator

Defines the struct containing containing pre-defined Monte Carlo samples from the biasing distribution to evaluate in integration.
This method is implemented with the user providing samples from the biasing distribution. 

# Arguments
- `g :: Distribution`        : biasing distribution
- `xsamp :: Vector{<:Real}`  : fixed set of samples
"""
struct ImportanceSampSamples <: ISIntegrator
    g :: Distribution
    xsamp :: Vector{<:Real}
end