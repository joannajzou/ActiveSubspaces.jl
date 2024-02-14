import Base: @kwdef

"""
@kwdef mutable struct GibbsQoI <: QoI

Defines the struct for computing the quantity of interest (QoI) as the expected value of a function h over Gibbs measure p given input parameters θ, e. g. E_p[h(θ)]

# Arguments
- `h :: Function`   : function to evaluate expectation, receiving inputs (x,θ)
- `p :: Gibbs`      : Gibbs (invariant) distribution with θ == nothing (undefined parameters)
- `∇h :: Union{Function, Nothing} = nothing`   : gradient of h (if nothing, gradient is computed using AutoDiff)


"""
@kwdef mutable struct GibbsQoI <: QoI
    h :: Function                               # function to evaluate expectation, receiving inputs (x,θ)
    p :: Gibbs                                  # Gibbs (invariant) distribution, receiving inputs (x,θ)
    ∇h :: Union{Function, Nothing} = nothing    # gradient of h (if nothing, gradient is computed using AutoDiff)
    
    function GibbsQoI(h::Function, p::Gibbs, ∇h::Function)
        return new(h, p, ∇h)
    end

    function GibbsQoI(h::Function, p::Gibbs, ∇h::Nothing)
        return new(h, p, ∇h)
    end

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
function expectation(θ::Union{Real, Vector{<:Real}}, 
                     qoi::GibbsQoI,
                     integrator::GaussQuadrature)
    # fix parameters
    qoim = assign_param(qoi, θ)
    # norm const
    Z = normconst(qoim.p, integrator)

    h̃(x) = qoim.h(x) .* updf(qoim.p, x) ./ Z
    return sum(integrator.w .* h̃.(integrator.ξ))
end


# integration with MCMC samples
function expectation(θ::Union{Real, Vector{<:Real}},
                     qoi::GibbsQoI,
                     integrator::MCMC)
    # fix parameters
    qoim = assign_param(qoi, θ)
    # x samples
    xsamp = rand(qoim.p, integrator.n, integrator.sampler, integrator.ρ0) 

    return sum(qoim.h.(xsamp)) / length(xsamp)
end


# integration with MC samples provided
function expectation(θ::Union{Real, Vector{<:Real}},
                     qoi::GibbsQoI,
                     integrator::MCSamples)
    # fix parameters
    qoim = assign_param(qoi, θ)

    return sum(qoim.h.(integrator.xsamp)) / length(integrator.xsamp)
end


# importance sampling (MCMC sampling from biasing dist.)
function expectation(θ::Union{Real, Vector{<:Real}},
                     qoi::GibbsQoI,
                     integrator::ISMCMC)
    # fix parameters
    qoim = assign_param(qoi, θ)
    # x samples
    xsamp = rand(integrator.g, integrator.n, integrator.sampler, integrator.ρ0) 

    # compute log IS weights
    return expectation_is_stable(xsamp, qoim.h, qoim.p, integrator.g)

end


# importance sampling (MC sampling from biasing dist.)
function expectation(θ::Union{Real, Vector{<:Real}},
                     qoi::GibbsQoI,
                     integrator::ISMC) 
    # fix parameters
    qoim = assign_param(qoi, θ)
    # x samples
    xsamp = rand(integrator.g, integrator.n)

    # compute log IS weights
    return expectation_is_stable(xsamp, qoim.h, qoim.p, integrator.g)

end


# importance sampling (samples provided)
function expectation(θ::Union{Real, Vector{<:Real}},
                     qoi::GibbsQoI,
                     integrator::ISSamples) 
    # fix parameters
    qoim = assign_param(qoi, θ)        
    # compute log IS weights
    return expectation_is_stable(integrator.xsamp, qoim.h, qoim.p, integrator.g; normint = integrator.normint)

end


# importance sampling from mixture distribution (samples provided)
function expectation(θ::Union{Real, Vector{<:Real}},
                     qoi::GibbsQoI,
                     integrator::ISMixSamples) 
    # fix parameters
    qoim = assign_param(qoi, θ)        

    # compute mixture weights
    wts = [compute_kernel(θ, c, integrator.knl) for c in integrator.refs]
    wts = wts ./ sum(wts)

    # sample from mixture model
    mm = MixtureModel(integrator.g, wts)
    xmix, rat = rand(mm, integrator.n, integrator.xsamp)

    # compute log IS weights
    expec, hsamp, iswts = expectation_is_stable(xmix, qoim.h, qoim.p, integrator.g, normint=integrator.normint)
    return expec, hsamp, iswts, wts
    
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
function grad_expectation(θ::Union{Real, Vector{<:Real}},
                          qoi::GibbsQoI,
                          integrator::Integrator)

    # compute gradient of h
    if qoi.∇h === nothing
        ∇θh = (x, γ) -> ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    else
        ∇θh = (x, γ) -> qoi.∇h(x, γ)
    end

    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV = expectation(θ, E_qoi, integrator)

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) - qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return expectation(θ, hh_qoi, integrator)

end

function grad_expectation(θ::Union{Real, Vector{<:Real}},
                          qoi::GibbsQoI,
                          integrator::ISIntegrator)

    # compute gradient of h
    if qoi.∇h === nothing
        ∇θh = (x, γ) -> ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    else
        ∇θh = (x, γ) -> qoi.∇h(x, γ)
    end

    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV, _, _ = expectation(θ, E_qoi, integrator) # for ISIntegrator

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) - qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return expectation(θ, hh_qoi, integrator)

end


function grad_expectation(θ::Union{Real, Vector{<:Real}},
                          qoi::GibbsQoI,
                          integrator::ISMixSamples)

    # compute gradient of h
    if qoi.∇h === nothing
        ∇θh = (x, γ) -> ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    else
        ∇θh = (x, γ) -> qoi.∇h(x, γ)
    end

    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV, _, _, _ = expectation(θ, E_qoi, integrator) # for ISMixSamples

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) - qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return expectation(θ, hh_qoi, integrator)

end


"""
function expectation_is_stable(xsamp::Vector, ϕ::Function, f::Gibbs, g::Distribution; normint=nothing)

Helper function for computing stable importance sampling weights (using log formulation).

"""
function expectation_is_stable(xsamp::Vector, ϕ::Function, f::Gibbs, g::Distribution; normint=nothing)
    logwt(xsamp) = if hasupdf(g) # Gibbs biasing dist
        logupdf.((f,), xsamp) .- logupdf.((g,), xsamp)
    elseif hasapproxnormconst(g) # mixture biasing dist
        logupdf.((f,), xsamp) .- logpdf(g, xsamp, normint)
    else # other biasing dist from Distributions.jl
        logupdf.((f,), xsamp) .- logpdf.((g,), xsamp)
    end

    M = maximum(logwt(xsamp))
    return sum( ϕ.(xsamp) .* exp.(logwt(xsamp) .- M) ) / sum( exp.(logwt(xsamp) .- M) ), ϕ.(xsamp), exp.(logwt(xsamp))
end



