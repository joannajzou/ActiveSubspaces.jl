abstract type Subspace end

include("pca.jl")
include("as.jl")


function compute_covmatrix(dθ::Vector{T}) where T <: Vector{<:Real}
    return Hermitian(mean(di*di' for di in dθ))
end


function select_eigendirections(dθ::Vector{T}, tol::Float64) where T <: Vector{<:Real}
    C = compute_covmatrix(dθ)
    λ, ϕ = eigen(C)
    λ, ϕ = λ[end:-1:1], ϕ[:, end:-1:1] # reorder
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    W = ϕ[:, Σ .> tol]
    return λ, W
end


function select_eigendirections(dθ::Vector{T}, tol::Int) where T <: Vector{<:Real}
    C = compute_covmatrix(dθ)
    λ, ϕ = eigen(C)
    λ, ϕ = λ[end:-1:1], ϕ[:, end:-1:1] # reorder
    W = ϕ[:, 1:tol]
    return λ, W
end


function select_eigendirections(C::Matrix{T}, tol::Int) where T <: Real
    λ, ϕ = eigen(C)
    λ, ϕ = λ[end:-1:1], ϕ[:, end:-1:1] # reorder
    W = ϕ[:, 1:tol]
    return λ, W
end


export AS, PCA, find_subspace, compute_covmatrix, select_eigendirections

