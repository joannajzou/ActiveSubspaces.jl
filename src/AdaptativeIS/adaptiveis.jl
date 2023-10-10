using Optim
"""
function adapt_mixture_IS(θsamp::Vector,
    q::GibbsQoI,
    sampler::Sampler,
    ρx0::Distribution,
    N0=5000,
    Nk=N0,
    niter=10)

Solves for the optimal components of a mixture biasing distribution by adaptive importance sampling.

# Arguments
- `θsamp :: Vector`             : parameters of target distributions
- `q :: GibbsQoI`               : expectation over a target distribution
- `sampler :: Sampler`          : Sampler struct specifying sampling algorithm
- `ρx0 :: Distribution`         : distribution of initial state for sampling
- `N0 = 5000`                   : number of samples from initial biasing dist.
- `Nk = N0`                     : number of samples from subsequent 
- `niter=10`                    : number of iterations of adaptation

# Outputs
- `λset::Vector{Vector}`        : set of parameters from adaptation (niter elements)

"""
function adapt_mixture_IS(θsamp::Vector,
                q::GibbsQoI,
                sampler::Sampler,
                ρx0::Distribution,
                N0=5000,
                Nk=N0,
                niter=10)

    # define set of target expectations
    nλ = length(θsamp)
    qois = [assign_param(q, θ) for θ in θsamp]

    # initialize
    λset = Vector{Vector}(undef, niter)
    xset = []
    Neval = N0

    # compute initial biasing dist
    λset[1] = θsamp 
    g0 = MixtureModel(λset[1], q.p)

    # draw samples
    xsamp = rand(g0, N0, sampler, ρx0)
    ISint = ISSamples(g0, xsamp)
    append!(xset, xsamp)

    # start adaptation
    for k = 2:niter
        println("------- iteration $k -------")

        # update parameters
        println("updating params")
        λk = update_dist_params(λset[k-1], qois, ISint)
        # draw samples
        gk = MixtureModel(λk, πg)
        xsamp = rand(gk, Nk, sampler, ρx0)
        append!(xset, xsamp)

        # update biasing distribution
        Neval += Nk
        hk = MixtureModel((h0, gk), ((Neval - Nk)/Neval, Nk/Neval))

        # save values
        λset[k] = λk
        h0 = hk
    end

    return λset
end
    

## update distribution parameters
function update_dist_params(λsamp::Vector,
                qois::Vector{GibbsQoI},
                integrator::ISSamples,
                )

    # load data
    xsamp = integrator.xsamp
    N = length(xsamp) # number of samples
    J = length(λsamp) # number of mixture components

    # initialize
    λopt = Vector{Vector{Float64}}(undef, J)

    # adapt each component
    for j = 1:J
        println("adapt component $j")
        entlog = λ -> lossfunc(λ, qois[j], integrator)

        @time res = optimize(entlog, λsamp[j], NelderMead(),
        Optim.Options(f_tol=1e-4, g_tol=1e-6, f_calls_limit=100)) # ; autodiff = :f

        λopt[j] = Optim.minimizer(res)
    end
    
    return λopt

end


## integrand of the QoI evaluated at samples from biasing distribution
function integrand(θ::Vector, qoi::GibbsQoI, integrator::ISSamples)
    qoim = assign_param(qoi, θ)
    hsamp = abs.(qoim.h.(integrator.xsamp))
    gsamp = updf.((qoim.p,), integrator.xsamp) ./ normconst(qoim.q, integrator)
    return hsamp .* log.(gsamp)
end


## log importance sampling weights
function logwt(qoi::GibbsQoI, integrator::ISSamples)
    return logupdf.((qoi.p,), integrator.xsamp) - logpdf(integrator.g, integrator.xsamp, integrator)
end 


## cross entropy loss function
function lossfunc(θ::Vector, qoi::GibbsQoI, integrator::ISSamples)
    qoim = assign_param(qoi, θ)
    logwts = logwt(qoim, integrator)
    M = maximum(logwts)
    # stable importance sampling estimate of cross entropy loss function
    ent = sum(integrand(θ, qoim) .* exp.(logwts .- M)) / sum( exp.(logwts .- M))
    if length(ent) > 1 # vector-valued entropy 
        return sum(ent) # sum of entropic vector
    else # scalar-valued entropy
        return ent
    end
end



