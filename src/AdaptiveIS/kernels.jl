"""
    Kernel

    A struct of abstract type Kernel is function that takes in two features and produces a semi-definite scalar representing the similarity between the two features.
"""
abstract type Kernel end

"""
    DotProduct <: Kernel 
        α :: Power of DotProduct kernel 


    Computes the dot product kernel between two features, i.e.,

    cos(θ) = ( A ⋅ B / (||A||^2||B||^2) )^α
"""
struct DotProduct <: Kernel
    α::Int
end
function DotProduct(; α = 2)
    DotProduct(α)
end

get_parameters(k::DotProduct) = (k.α,)

## TODO: rename 'd' to 'k' for kernel
function compute_kernel(
    A::T,
    B::T,
    d::DotProduct,
) where {T<:Union{Vector{<:Real},Symmetric{<:Real,<:Matrix{<:Real}}}}
    (dot(A, B) / sqrt(dot(A, A) * dot(B, B)))^d.α
end

## TODO: edit to be in terms of d.α, not 2
function compute_grad_kernel(
    A::T,
    B::T,
    d::DotProduct,
    ) where {T<:Vector{<:Real}}

    normA = norm(A)
    normB = norm(B)
    dotAB = dot(A, B)
    return 2 * dotAB * (normA^2 * B - dotAB * A) / (normA^4 * normB^2)
end


""" 
    RBF <: Kernel 
        d :: Distance function 
        α :: Reguarlization parameter 
        ℓ :: Length-scale parameter
        β :: Scale parameter
    

    Computes the squared exponential kernel, i.e.,

     k(A, B) = β \exp( -\frac{1}{2} d(A,B)/ℓ^2 ) + α δ(A, B) 
"""
struct RBF <: Kernel
    d::Distance
    α::Real
    ℓ::Real
    β::Real
end
function RBF(d; α = 1e-8, ℓ = 1.0, β = 1.0)
    RBF(d, α, ℓ, β)
end

get_parameters(k::RBF) = (k.α, k.ℓ, k.β)

"""
    compute_kernel(A, B, k)

Compute similarity kernel between features A and B using kernel k. 
"""
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

