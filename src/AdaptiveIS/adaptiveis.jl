import PotentialLearning: compute_distance, compute_kernel
using ForwardDiff

# include("distances.jl")
# include("kernels.jl")
include("discrepancies.jl")
include("adapt_components.jl")
include("adapt_weights.jl")

function compute_distance(
    B1::T,
    B2::T,
    e::Euclidean) where {T<:Real}
    return abs.(B1 - B2)^2
end

function compute_grad_distance(
    A::T,
    B::T,
    e::Euclidean
    ) where {T<:Vector{<:Real}}

    return 2 * e.Cinv * (A - B)
end

function compute_kernel(
    A::T,
    B::T,
    r::RBF,
    ) where {T<:Real}

    d2 = compute_distance(A, B, r.d)
    r.β * exp(-0.5 * d2 / r.ℓ)
end

function compute_kernel(
    A::T,
    B::T,
    r::RBF,
) where {T<:Union{Vector{<:Real},Symmetric{<:Real,<:Matrix{<:Real}}}}
    d2 = compute_distance(A, B, r.d)
    r.β * exp(-0.5 * d2 / r.ℓ)
end


function compute_grad_kernel(
    A::T,
    B::T,
    knl::RBF,
    ) where {T<:Vector{<:Real}}

    k = compute_kernel(A, B, knl)
    𝝯d = compute_grad_distance(A, B, knl.d)
    return -k * 𝝯d / (2*knl.ℓ^2)
end


export adapt_mixture_IS
export adapt_mixture_weights, normalize
export compute_grad_distance
export compute_grad_kernel
export KernelSteinDiscrepancy, compute_discrepancy