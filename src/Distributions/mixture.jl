### define mutable Mixture distribution in Distribution.jl
import Distributions: MixtureModel
import StatsBase: params
using StatsFuns
# import Base: minimum, maximum, rand


# outer constructor: creates "mutable" MixtureModel with updated probabilities
function MixtureModel(MM::MixtureModel,prob::Vector{<:Real})
    return MixtureModel(MM.components, prob)
end

function MixtureModel(params::Vector, π0::Gibbs) # equal weights
    πg = [Gibbs(π0, θ=p) for p in params]
    return MixtureModel(πg)
end

function MixtureModel(params::Vector, π0::Gibbs, prob::Vector{<:Real})
    πg = [Gibbs(π0, θ=p) for p in params]
    return MixtureModel(πg, prob)
end

function MixtureModel(dists::Tuple{MixtureModel, MixtureModel}, wts::Tuple{Real, Real})
    n = length(dists)
    comps = reduce(vcat, [dists[i].components for i = 1:n])
    α = reduce(vcat, [wts[i] .* probs(dists[i]) for i = 1:n])
    return MixtureModel(comps, α)
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
function rand(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs},
            n::Int,
            sampler::Sampler,
            ρ0::Distribution;
            burn=1000)

    categories = StatsBase.sample(1:ncomponents(d), Weights(probs(d)), n)
    xsamp = Float64[]
    for i = 1:ncomponents(d)
        ni = length(findall(x -> x == i, categories))
        if ni != 0
            nsim = burn + ni
            xi = rand(component(d, i), nsim, sampler, ρ0; burn=0) 
            append!(xsamp, rand(xi[(burn+1):end], ni))
        end
    end
    return xsamp
end 

# samples provided
function rand(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, 
            n::Int,
            samples::Vector)
    categories = StatsBase.sample(1:ncomponents(d), Weights(probs(d)), n)
    xsamp = Float64[]
    catratio = Float64[]
    for i = 1:ncomponents(d)
        ntot = length(samples[i])
        ni = length(findall(x -> x == i, categories))
        append!(catratio, ni/n)
        if ni != 0
            randid = StatsBase.sample(1:ntot, ni; replace=true)
            xi = samples[i][randid]
            append!(xsamp, xi)
        end
    end
    return xsamp, catratio
end 

# 2 - pdf
function pdf(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, x::Float64, integrator::Integrator)
    p = probs(d)
    return sum(p_i * pdf(component(d, i), x, integrator) for (i, p_i) in enumerate(p) if !iszero(p_i))
end 

function pdf(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, xsamp::Vector{Float64}, integrator::Integrator)
    p = probs(d)
    return sum(p_i * updf.((component(d, i),), xsamp) ./ normconst(component(d, i), integrator) for (i, p_i) in enumerate(p) if !iszero(p_i))
end 


# 3 - log unnormalized pdf
function logpdf(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, x::Float64, integrator::Integrator)
    p = probs(d)
    lp = logsumexp(log(p_i) + logpdf(component(d, i), x, integrator) for (i, p_i) in enumerate(p) if !iszero(p_i))
    return lp
end

function logpdf(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, xsamp::Vector{Float64}, integrator::Integrator)
    p = probs(d)
    data = reduce(hcat, [log(p_i) .+ logupdf.((component(d, i),), xsamp) .- log(normconst(component(d, i), integrator)) for (i, p_i) in enumerate(p) if !iszero(p_i)])
    lp = logsumexp.([data[i,:] for i = 1:size(data,1)])
    return lp
end



