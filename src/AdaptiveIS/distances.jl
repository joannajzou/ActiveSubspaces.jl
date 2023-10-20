""" 
    Distance

    A struct of abstract type Distance produces the distance between two feature vectors.
"""
abstract type Distance end

"""
    Euclidean <: Distance 
        Cinv :: Covariance Matrix 

    Computes the squared euclidean distance with weight matrix Cinv, the inverse of some covariance matrix.
"""
struct Euclidean{T} <: Distance where {T}
    Cinv::Union{Symmetric{T,Matrix{T}},Diagonal{T,Vector{T}}}
    Csqrt::Union{Symmetric{T,Matrix{T}},Diagonal{T,Vector{T}}}
end
function Euclidean(dim::Int)
    Euclidean(1.0 * I(dim), 1.0 * I(dim))
end
function Euclidean(
    Cinv::Union{Symmetric{T,Matrix{T}},Diagonal{T,Vector{T}}},
) where {T<:Real}
    Csqrt = sqrt(Cinv)
    Euclidean(Cinv, Csqrt)
end

"""
function compute_distance(B1::Vector{T}, B2::Vector{T}, e::Euclidean) where {T<:Real}

    Computes the squared euclidean distance with weight matrix Cinv, the inverse of some covariance matrix.
"""
function compute_distance(B1::Vector{T}, B2::Vector{T}, e::Euclidean) where {T<:Real}
    (B1 - B2)' * e.Cinv * (B1 - B2)
end

function compute_grad_distance(
    A::T,
    B::T,
    e::Euclidean
    ) where {T<:Vector{<:Real}}

    return 2 * e.Cinv * (A - B)
end