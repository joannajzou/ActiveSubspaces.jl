abstract type QoI end

"""
struct Expectation{T<:Real} <: QoI
    h :: Function                   # function to evaluate expectation 
    p :: Gibbs{Float64}             # Gibbs (invariant) distribution
    θ :: Union{T, Vector{T}}        # value of parameters to evaluate
        
Computes the quantity of interest (QoI) as the expected value of a function h over measure p given input parameters θ, e. g. E_p[h(θ)]
"""
struct Expectation <: QoI
    h :: Function                   # function to evaluate expectation 
    p :: Gibbs{Float64}             # Gibbs (invariant) distribution
end


# compute QoI with MCMC sampler
function Q(θ, qoi::QoI; n=10000, kwargs...)
    A = Dict(kwargs)
    if haskey(A, :mcmc) & haskey(A, :x0) # sample with MCMC
        x = rand(qoi.p, n, A[:mcmc], A[:x0]) 
    else
        x = rand(qoi.p, n) # sample analytically
    end
    return sum(qoi.h.(x)) / length(x)
end


# compute QoI with quadrature points
function Q(θ, qoi::QoI, ξ::Vector{T<:Real}, w::Vector{T<:Real})
    Z = normconst(qoi.p)
    h̃(x) = qoi.h(x) * updf(qoi.p, x) / Z
    return sum(w * h̃.(ξ))
end


# compute QoI with importance sampling distribution
function Q(θ, qoi::QoI, g::Distribution; kwargs...)
    A = Dict(kwargs)
    if haskey(A, :mcmc) & haskey(A, :x0) # sample with MCMC
        x = rand(g, n, A[:mcmc], A[:x0]) 
    else
        x = rand(g, n) # sample analytically
    end
    xsamp = sampler(g, )
    xsamp
