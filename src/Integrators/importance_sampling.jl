abstract type ISIntegrator <: Integrator end

"""
struct ISMC <: ISIntegrator

Contains parameters for Importance Sampling (IS) integration using MC samples from a biasing distribution.
This method is implemented when the biasing distribution g can be analytically sampled using rand().

# Arguments
- `g :: Distribution`      : biasing distribution
- `n :: Int`               : number of samples
"""
struct ISMC <: ISIntegrator
    g :: Distribution
    n :: Int
end

"""
struct ISMCMC <: ISIntegrator

Contains parameters for Importance Sampling (IS) integration using MCMC samples from a biasing distribution.
This method is implemented when the biasing distribution g cannot be analytically sampled.

# Arguments
- `g :: Distribution`      : biasing distribution
- `n :: Int`               : number of samples
- `sampler :: Sampler`     : type of sampler (see `Sampler`)
- `ρ0 :: Distribution`     : prior distribution of the state
"""
struct ISMCMC <: ISIntegrator
    g :: Distribution
    n :: Int 
    sampler :: Sampler
    ρ0 :: Distribution
end


"""
struct ISSamples <: ISIntegrator

Contains pre-defined Monte Carlo samples from the biasing distribution to evaluate in integration.
This method is implemented with the user providing samples from the biasing distribution. 

# Arguments
- `g :: Distribution`                               : biasing distribution
- `xsamp :: Vector`                                 : fixed set of samples
- `normint :: Union{Integrator, Nothing}`           : integrator for computing normalizing constant of biasing distribution (required for mixture models)

"""
mutable struct ISSamples <: ISIntegrator
    g :: Distribution
    xsamp :: Vector
    normint :: Union{Integrator, Nothing}

    function ISSamples(g::MixtureModel, xsamp::Vector, normint::Integrator)
        return new(g, xsamp, normint)
    end

    function ISSamples(g::Distribution, xsamp::Vector)
        return new(g, xsamp, nothing)
    end

end



"""
struct ISMixSamples <: ISIntegrator

Contains parameters for Importance Sampling (IS) integration using samples from a mixture biasing distribution.
The mixture weights are computed based on a kernel distance metric of the parameter with respect to parameters of the mixture component distributions.
This method is implemented with the user providing samples from each component mixture distribution. 

# Arguments
- `g :: MixtureModel`        : mixture biasing distribution
- `n :: Int`                 : number of samples
- `knl :: Kernel`            : kernel function to compute mixture weights
- `xsamp :: Vector`          : Vector of sample sets from each component distribution
- `normint :: Integrator`    : Integrator for the approximating the normalizing constant of each component distribution

"""
mutable struct ISMixSamples <: ISIntegrator
    g :: MixtureModel
    refs :: Vector
    n :: Int
    knl :: Kernel
    xsamp :: Vector
    normint :: Integrator


    function ISMixSamples(g::MixtureModel, refs::Vector, n::Int, knl::Kernel, xsamp::Vector, normint::Integrator)
        return new(g, refs, n, knl, xsamp, normint)
    end

    function ISMixSamples(g::MixtureModel, n::Int, knl::Kernel, xsamp::Vector, normint::Integrator)
        d = new()
        d.g = g
        d.refs = [πg.θ for πg in g.components]
        d.n = n
        d.knl = knl
        d.xsamp = xsamp
        d.normint = normint
        return d
    end
end
