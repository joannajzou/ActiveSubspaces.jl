abstract type Subspace end

include("pca.jl")
include("as.jl")

export AS, PCA, find_subspace


function compute_eigen(dθ::Vector{T}) where T <: Vector{<:Real}
    C = mean(di*di' for di in dθ)
    return eigen(Symmetric(C))
end


function select_eigendirections(dθ::Vector{T}, tol::Float64) where T <: Vector{<:Real}
    λ, ϕ = compute_eigen(dθ)
    λ, ϕ = λ[end:-1:1], ϕ[end:-1:1, :] # reorder
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    W = ϕ[Σ .> tol, :]
    return λ, W
end


function select_eigendirections(d::Vector{T}, tol::Int) where T <: Vector{<:Real}
    λ, ϕ = compute_eigen(d)
    λ, ϕ = λ[end:-1:1], ϕ[end:-1:1, :] # reorder
    W = ϕ[1:tol, :]
    return λ, W
end


