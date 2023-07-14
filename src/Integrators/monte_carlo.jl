abstract type MCIntegrator <: Integrator end

"""
struct MonteCarlo <: MCIntegrator

Contains parameters for Monte Carlo (MC) integration using MC samples.
This method may only be implemented when the distribution can be analytically sampled using rand().

# Arguments
- `n :: Int`   : number of samples
"""
struct MonteCarlo <: MCIntegrator
    n :: Int
end


"""
struct MCMC <: MCIntegrator
    
Contains parameters for Monte Carlo (MC) integration using MCMC samples.
This method is implemented when the distribution cannot be analytically sampled.

# Arguments
- `n :: Int`               : number of samples
- `sampler :: Sampler`     : type of sampler (see `Sampler`)
- `ρ0 :: Distribution`     : prior distribution of the state
"""
struct MCMC <: MCIntegrator
    n :: Int
    sampler :: Sampler
    ρ0 :: Distribution
end


"""
struct MCSamples <: MCIntegrator

Contains pre-defined Monte Carlo samples to evaluate in integration.
This method is implemented with the user providing samples from the distribution.

# Arguments
- `xsamp :: Vector`    : fixed set of samples
"""
struct MCSamples <: MCIntegrator
    xsamp :: Vector
end
