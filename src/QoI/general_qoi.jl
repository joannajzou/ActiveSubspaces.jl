import Base: @kwdef

"""
@kwdef mutable struct GeneralQoI <: QoI

Defines the struct for computing the quantity of interest (QoI) as the expected value of a function h over a probability measure p given input parameters θ, e. g. E_p[h(θ)]

# Arguments
- `h :: Function`     : function to evaluate expectation, receiving inputs (x,θ)
- `p :: Distribution` : probability distribution with θ == nothing (undefined parameters)

"""
@kwdef mutable struct GeneralQoI <: QoI
    h :: Function                   # function to evaluate expectation, receiving inputs (x,θ)
    p :: Distribution               # probability distribution, receiving inputs (x)
end


function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI; kwargs...) # for GeneralQoI
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    A = Dict(kwargs) # arguments

    if haskey(A, :ξ) & haskey(A, :w)                                            # quadrature integration
        return expectation_(qoim, A[:ξ], A[:w])

    elseif haskey(A, :g) & haskey(A, :xsamp)                                    # importance sampling (samples provided)
        return expectation_(qoim, A[:g], A[:xsamp])

    elseif haskey(A, :g) & haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0) # importance sampling (by MCMC)
        return expectation_(qoim, A[:g], A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :n)                                        # importance sampling (by MC)                
        return expectation_(qoim, A[:g], A[:n]) 

    elseif haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)                 # MCMC sampling
        return expectation_(qoim, A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :n)                                                        # Monte Carlo sampling
        return expectation_(qoim, A[:n])

    elseif haskey(A, :xsamp)                                                    # samples provided
        return expectation_(qoim, A[:xsamp])

    else
        println("ERROR: key word arguments missing or invalid")

    end
end


# quadrature integration
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI, integrator::QuadIntegrator)
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    Z = normconst(qoim.p, integrator.ξ, integrator.w)
    h̃(x) = qoim.h(x) * updf(qoim.p, x) / Z
    return sum(integrator.w .* h̃.(ξ))
end


# integration with MCMC samples
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI, integrator::MCMC)
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    # x samples
    x = rand(qoim.p, integrator.n, integrator.sampler, integrator.ρ0) 

    return sum(qoim.h.(x)) / length(x)
end


# integration with MC samples provided
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI, integrator::MCSamples)
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    # x samples
    x = integrator.xsamp

    return sum(qoim.h.(x)) / length(x)
end


# importance sampling (samples provided)
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI, integrator::ISSamples) 
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)
        
    # x samples
    x = integrator.xsamp

    logwt(x) = if hasupdf(integrator.g) # g has an unnormalized pdf
        logpdf(qoim.p, x) - logupdf(integrator.g, x)
    else # g does not have an unnormalized pdf
        logpdf(qoim.p, x) - logpdf(integrator.g, x)
    end

    M = maximum(logwt.(x))
    return sum( qoim.h.(x) .* exp.(logwt.(x) .- M) ) / sum( exp.(logwt.(x) .- M) ), qoim.h.(x), exp.(logwt.(x))
end


# importance sampling (MCMC sampling)
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI, integrator::ISMCMC)
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    # x samples
    x = rand(integrator.g, integrator.n, integrator.sampler, integrator.ρ0) 

    logwt(x) = if hasupdf(integrator.g) # g has an unnormalized pdf
        logpdf(qoim.p, x) - logupdf(integrator.g, x)
    else # g does not have an unnormalized pdf]
        logpdf(qoim.p, x) - logpdf(integrator.g, x)
    end

    M = maximum(logwt.(x))
    return sum( qoim.h.(x) .* exp.(logwt.(x) .- M) ) / sum( exp.(logwt.(x) .- M) ), qoim.h.(x), exp.(logwt.(x))
end


# importance sampling (MC sampling)
function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI, integrator::ISMC) 
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    # x samples
    x = rand(integrator.g, integrator.n)

    logwt(x) = logpdf(qoim.p, x)  - logpdf(integrator.g, x)
    M = maximum(logwt.(x))
    return sum( qoim.h.(x) .* exp.(logwt.(x) .- M) ) / sum( exp.(logwt.(x) .- M) ), qoim.h.(x), exp.(logwt.(x))
end


# compute gradient of expectation
function grad_expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI, integrator::Integrator; gradh::Union{Function, Nothing}=nothing)
    # compute gradient of h
    if gradh === nothing
        ∇θh = (x, γ) -> ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    else
        ∇θh = (x, γ) -> gradh(x, γ)
    end
    
    # compute inner expectation E_p[∇θV]
    E_qoi = GeneralQoI(h=qoi.p.∇θV, p=qoi.p)
    if typeof(integrator) <: ISIntegrator
        E_∇θV, _, _ = expectation(θ, E_qoi, integrator)
    else
        E_∇θV = expectation(θ, E_qoi, integrator)
    end

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) + qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GeneralQoI(h=hh, p=qoi.p)
    return expectation(θ, hh_qoi, integrator)

end
