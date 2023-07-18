### define mutable Mixture distribution in Distribution.jl
import Distributions: MixtureModel
import StatsBase: params
using StatsFuns
# import Base: minimum, maximum, rand


# outer constructor: creates "mutable" MixtureModel with updated probabilities
function MixtureModel(MM :: MixtureModel, prob :: Vector{<:Real})
    return MixtureModel(MM.components, prob)
end

# 0 - parameters
# function components(d::MixtureModel) already defined
# function probs(d::MixtureModel) already defined
function params(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs})
    β = [params(component(d, i))[1] for i = 1:ncomponents(d)]
    θ = [params(component(d, i))[2] for i = 1:ncomponents(d)]
    return (β, θ)
end


# 1 - random sampler
function rand(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, n::Int, sampler::Sampler, ρ0::Distribution; burn=1000)
    categories = StatsBase.sample(1:ncomponents(d), Weights(probs(d)), n)
    xsamp = Float64[]
    for i = 1:ncomponents(d)
        ni = length(findall(x -> x == i, categories))
        nsim = burn + ni
        xi = rand(component(d, i), nsim, sampler, ρ0; burn=0.0) 
        append!(xsamp, xi[burn+1:end])
    end
    return xsamp
end 

# samples provided
function rand(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, n::Int, samples::Vector; wts::Vector=ones(ncomponents(d))/ncomponents(d))
    d = MixtureModel(d, wts)
    ntot = length(samples[1])
    categories = StatsBase.sample(1:ncomponents(d), Weights(probs(d)), n)
    xsamp = Vector{Float64}[]
    catratio = Float64[]
    for i = 1:ncomponents(d)
        ni = length(findall(x -> x == i, categories))
        append!(catratio, ni/n)
        if ni != 0
            randid = StatsBase.sample(1:ntot, ni; replace=false)
            xi = samples[i][randid]
            append!(xsamp, xi)
        end
    end
    return xsamp, catratio
end 

# 2 - unnormalized pdf
function updf(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, x)
    p = probs(d)
    return sum(pi * updf(component(d, i), x) for (i, pi) in enumerate(p) if !iszero(pi))
end 

# 3 - log unnormalized pdf
function logupdf(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, x)
    p = probs(d)
    lp = logsumexp(log(pi) + logupdf(component(d, i), x) for (i, pi) in enumerate(p) if !iszero(pi))
    return lp
end

# 4 - normalization constant (partition function)
function normconst(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, ξ::Vector, w::Vector)
    return sum(w' * updf.((d,), ξ))
end

