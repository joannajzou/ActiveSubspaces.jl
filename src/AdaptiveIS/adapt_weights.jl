function gradlogint(q::GibbsQoI, x)
    logϕ(x) = log.(abs.(q.h(x))+ 1e-12)
    ∇logϕ = x -> ForwardDiff.derivative(x -> logϕ(x), x)
    ∇logp = x -> gradlogpdf(q.p, x)

    return ∇logϕ(x) - ∇logp(x)
end

function normalize(α::Vector)
    α = abs.(α)
    return α ./ sum(α)
end


function adapt_mixture_weights(
    θ::Vector,
    qoi::GibbsQoI,
    mm::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs},
    samples::Vector{Vector{T}},
    d::Discrepancy
    ) where T <: Union{Real, Vector{<:Real}}
    
    # target
    qoi = assign_param(qoi, θ)
    starget = x -> gradlogint(qoi, x)
    # components
    comps = components(mm)
    # compute discrepancy
    discr = [compute_discrepancy(starget, comp, samp, d) for (comp, samp) in zip(comps, samples)]
    # compute normalized weights
    return normalize(discr.^-1)
end


function adapt_mixture_weights(
    θ::Vector,
    λsamp::Vector, 
    k::Kernel
    ) 
    
    # # component parameters
    # _, λ = params(mm)
    # compute kernel distanace
    dist = [compute_kernel(θ, λi, k) for λi in λsamp]
    # compute normalized weights
    return ActiveSubspaces.normalize(dist)
end