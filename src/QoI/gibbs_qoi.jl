import Base: @kwdef

"""
@kwdef mutable struct GibbsQoI <: QoI

Defines the struct for computing the quantity of interest (QoI) as the expected value of a function h over Gibbs measure p given input parameters θ, e. g. E_p[h(θ)]

# Arguments
- `h :: Function`   : function to evaluate expectation, receiving inputs (x,θ)
- `p :: Gibbs`      : Gibbs (invariant) distribution with θ == nothing (undefined parameters)

"""
@kwdef mutable struct GibbsQoI <: QoI
    h :: Function                   # function to evaluate expectation, receiving inputs (x,θ)
    p :: Gibbs                      # Gibbs (invariant) distribution, receiving inputs (x,θ)
end


"""
function expectation(θ:: Union{Real,Vector{<:Real}}, qoi::QoI, integrator::Integrator)

Evaluates the expectation E_p[h(θ)], where qoi.h(θ) is the random variable and qoi.p is the probability measure 

# Arguments
- `θ :: Union{Real, Vector{<:Real}}`    : parameters to evaluate expectation
- `qoi :: QoI`                          : QoI object containing h and p, where θ==nothing
- `integrator :: Integrator`            : struct specifying method of integration

# Outputs 
- `expec_estimator :: Real`             : estimate of scalar-valued expectation

# Optional Outputs (from Importance Sampling)
- `h(x) :: Vector`                      : integrand evaluations from importance sampling 
- `wt(x) :: Vector`                     : weights from importance sampling 

""" 
# quadrature integration
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::GaussQuadrature)
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))

    Z = normconst(qoim.p, integrator.ξ, integrator.w)
    h̃(x) = qoim.h(x) * updf(qoim.p, x) / Z
    return sum(integrator.w .* h̃.(integrator.ξ))
end


# integration with MCMC samples
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::MCMC)
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))

    # x samples
    x = rand(qoim.p, integrator.n, integrator.sampler, integrator.ρ0) 

    return sum(qoim.h.(x)) / length(x)
end


# integration with MC samples provided
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::MCSamples)
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))

    # x samples
    x = integrator.xsamp

    return sum(qoim.h.(x)) / length(x)
end


# importance sampling (MCMC sampling)
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::ISMCMC)
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))

    # x samples
    x = rand(integrator.g, integrator.n, integrator.sampler, integrator.ρ0) 

    # compute log IS weights
    logwt(x) =if hasupdf(integrator.g) # g has an unnormalized pdf
        logupdf(qoim.p, x) - logupdf(integrator.g, x)
    else # g does not have an unnormalized pdf
        logupdf(qoim.p, x) - logpdf(integrator.g, x)
    end

    M = maximum(logwt.(x))
    return sum( qoim.h.(x) .* exp.(logwt.(x) .- M) ) / sum( exp.(logwt.(x) .- M) ), qoim.h.(x), exp.(logwt.(x))
end


# importance sampling (MC sampling)
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::ISMC) 
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))

    # x samples
    x = rand(integrator.g, integrator.n)

    # compute log IS weights
    logwt(x) = logupdf(qoim.p, x)  - logpdf(integrator.g, x)
    M = maximum(logwt.(x))
    return sum( qoim.h.(x) .* exp.(logwt.(x) .- M) ) / sum( exp.(logwt.(x) .- M) ), qoim.h.(x), exp.(logwt.(x))
end


# importance sampling (samples provided)
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::ISSamples) 
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))
        
    # x samples
    x = integrator.xsamp

    # compute log IS weights
    logwt(x) = if hasupdf(integrator.g) # g has an unnormalized pdf
        logupdf(qoim.p, x) - logupdf(integrator.g, x)
    else # g does not have an unnormalized pdf
        logupdf(qoim.p, x) - logpdf(integrator.g, x)
    end

    M = maximum(logwt.(x))
    return sum( qoim.h.(x) .* exp.(logwt.(x) .- M) ) / sum( exp.(logwt.(x) .- M) ), qoim.h.(x), exp.(logwt.(x))
end


# importance sampling from mixture distribution (samples provided)
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::ISMixSamples) 
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))
        
    # draw samples from mixture model
    _, centers = params(integrator.g)
    wts = [compute_kernel(θ, c, integrator.knl) for c in centers]
    wts = wts ./ sum(wts)
    mm = MixtureModel(integrator.g, wts)
    x, _ = rand(mm, integrator.n, integrator.xsamp)

    # compute log IS weights
    logwt(x) = if hasupdf(integrator.g) # g has an unnormalized pdf
        logupdf(qoim.p, x) - logupdf(integrator.g, x)
    else # g does not have an unnormalized pdf
        logupdf(qoim.p, x) - logpdf(integrator.g, x)
    end

    M = maximum(logwt.(x))
    return sum( qoim.h.(x) .* exp.(logwt.(x) .- M) ) / sum( exp.(logwt.(x) .- M) ), qoim.h.(x), exp.(logwt.(x))
end


"""
function grad_expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::Integrator; gradh::Union{Function, Nothing}=nothing)

Computes the gradient of the QoI with respect to the parameters θ, e. g. ∇θ E_p[h(θ)].
See `expectation` for more information.

# Arguments
- `θ :: Union{Real, Vector{<:Real}}`    : parameters to evaluate expectation
- `qoi :: QoI`                          : QoI object containing h and p
- `integrator :: Integrator`            : struct specifying method of integration
- `gradh :: Union{Function, Nothing}`   : gradient of qoi.h; if nothing, compute with ForwardDiff

# Outputs
- `expec_estimator :: Vector{<:Real}`   : estimate of vector-valued gradient of the expectation


"""
function grad_expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::ISIntegrator; gradh::Union{Function, Nothing}=nothing)
    # compute gradient of h
    if gradh === nothing
        ∇θh = (x, γ) -> ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    else
        ∇θh = (x, γ) -> gradh(x, γ)
    end

    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV, _, _ = expectation(θ, E_qoi, integrator) # for ISIntegrator

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) - qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return expectation(θ, hh_qoi, integrator)

end


function grad_expectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI, integrator::Integrator; gradh::Union{Function, Nothing}=nothing)
    # compute gradient of h
    if gradh === nothing
        ∇θh = (x, γ) -> ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    else
        ∇θh = (x, γ) -> gradh(x, γ)
    end

    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV = expectation(θ, E_qoi, integrator)

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) - qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return expectation(θ, hh_qoi, integrator)

end