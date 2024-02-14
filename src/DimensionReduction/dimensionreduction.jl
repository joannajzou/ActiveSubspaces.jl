"""
struct Subspace where T <: Real
    
Defines the struct containing variables of the active subspace.

# Arguments
- `C :: Matrix{T}`          : covariance matrix
- `λ :: Vector{T}`          : eigenvalues of covariance matrix
- `W1 :: Matrix{T}`         : active subspace
- `W2 :: Matrix{T}`         : inactive subspace
- `π_y :: Distribution`     : marginal density of active variable
- `π_z :: Distribution`     : marginal density of inactive variable

"""
struct Subspace
    C :: Matrix{<:Real}           
    λ :: Vector{<:Real}          
    W1 :: Matrix{<:Real}         
    W2 :: Matrix{<:Real}         
    π_y :: Distribution     
    π_z :: Distribution  
end


include("covariances.jl")
include("subspaces.jl")

export compute_covmatrix
export Subspace, compute_eigenbasis, find_subspaces, compute_marginal, compute_as, sample_as
export transf_to_paramspace_fix, transf_to_subspace
