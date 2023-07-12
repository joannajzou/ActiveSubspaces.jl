### define mutable Mixture distribution in Distribution.jl
import Distributions: MixtureModel
using StatsFuns
# import Base: minimum, maximum, rand


# outer constructor: creates "mutable" MixtureModel with updated probabilities
function MixtureModel(MM :: MixtureModel, prob :: Vector{<:Real})
    return MixtureModel(MM.components, prob)
end

# 0 - parameters
# function components(d::MixtureModel) already defined
# function probs(d::MixtureModel) already defined

# 1 - random sampler
function rand(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, n::Int, sampler::Sampler, ρ0::Distribution; burn=0.1)
    categories = StatsBase.sample(1:ncomponents(d), Weights(probs(d)), n)
    xsamp = Float64[]
    for i = 1:ncomponents(d)
        ni = length(findall(x -> x == i, categories))
        nsim = Int(ceil((1+burn) * ni))
        xi = rand(component(d, i), nsim, sampler, ρ0; burn=0.0) 
        append!(xsamp, xi[1:ni])
    end
    return xsamp
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

