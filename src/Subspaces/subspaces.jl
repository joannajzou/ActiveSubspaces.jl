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


"""
function compute_covmatrix(dθ::Vector{T}) where T <: Vector{<:Real}

Computes covariance matrix from samples of feature vector dθ.

# Arguments
- `dθ :: Vector{Vector{<:Real}}`    : samples of feature vector

# Outputs 
- `C :: Matrix{Real}`            : empirical covariance matrix

""" 
function compute_covmatrix(dθ::Vector{T}) where T <: Vector{<:Real}
    return Matrix(Hermitian(mean(di*di' for di in dθ)))
end


"""
function compute_covmatrix(dθ::Vector{T}, nsamp_arr::Vector{Int64}) where T <: Vector{<:Real}

Computes array of covariance matrices using varying sample sizes of feature vector dθ. All elements of nsamp_arr <= length(dθ).

# Arguments
- `dθ :: Vector{T}`                         : samples of feature vector
- `nsamp_arr :: Vector{Int64}`              : subsets of samples

# Outputs 
- `C_arr :: Dict{Int64, Matrix{Float64}}`   : dictionary with keys being nsamp, values being covariance matrices

""" 
function compute_covmatrix(dθ::Vector{T}, nsamp_arr::Vector{Int64}) where T <: Vector{<:Real}
    C_arr = Dict{Int64, Matrix{Float64}}()

    for nsamp in nsamp_arr
        C_arr[nsamp] = compute_covmatrix(dθ[1:nsamp])
    end

    return C_arr
end


"""
function compute_eigenbasis(dθ::Vector{T}) where T <: Vector{<:Real}

Computes eigendecomposition of empirical covariance matrix of feature vector dθ.

# Arguments
- `dθ :: Vector{T}`                      : samples of feature vector

# Outputs 
- `C :: Matrix{Real}`                    : empirical covariance matrix
- `λ :: Vector{Real}`                    : eigenvalues of C
- `W :: Matrix{Real}`                    : eigenvectors of C

""" 
function compute_eigenbasis(dθ::Vector{T}) where T <: Vector{<:Real}
    C = compute_covmatrix(dθ)
    λ, W = eigen(C)
    idx = sortperm(λ, rev=true)
    λ, W = λ[idx], W[:,idx] # reorder
    for i = 1:length(λ)
        if W[1,i] < 0; W[:,i] = -W[:,i]; end
    end
    return C, λ, W
end


"""
function compute_eigenbasis(C::Matrix{T}) where T <: Real

Computes eigendecomposition of provided covariance matrix.

# Arguments
- `C :: Matrix{T}`                       : covariance matrix

# Outputs 
- `C :: Matrix{Real}`                    : covariance matrix (same as input)
- `λ :: Vector{Real}`                    : eigenvalues of C
- `W :: Matrix{Real}`                    : eigenvectors of C

""" 
function compute_eigenbasis(C::Matrix{T}) where T <: Real
    λ, W = eigen(C)
    idx = sortperm(λ, rev=true)
    λ, W = λ[idx], W[:,idx] # reorder
    for i = 1:length(λ)
        if W[1,i] < 0; W[:,i] = -W[:,i]; end
    end
    return C, λ, W
end


function compute_eigenbasis(C::Dict, nsamp_arr::Vector)
    nrepl = length(C[nsamp_arr[1]])
    λ = Dict{Int64, Vector}()
    W = Dict{Int64, Vector}()
    for nsamp in nsamp_arr
        λ[nsamp] = Vector{Vector{Float64}}(undef, nrepl)
        W[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)
        for j = 1:nrepl
            _, λ[nsamp][j], W[nsamp][j] = compute_eigenbasis(C[nsamp][j])
        end
    end

    return λ, W
end



"""
function find_subspaces(λ::Vector{T}, W::Matrix{T}, tol::Float64) where T <: Real

Partitions active and inactive subspaces. The partition point is determined by the tol parameter, which is the percent of residual variance.
Example: set tol=0.05 to compute subspace explaining 95% of variance in the QoI. 

# Arguments
- `λ :: Vector{T}`                       : eigenvalues of d-by-d matrix
- `W :: Matrix{T}`                       : eigenvectors of d-by-d matrix
- `tol :: Float64`                       : tolerance representing cutoff percent of residual variance

# Outputs 
- `W1 :: Matrix{T}`                      : active subspace (d-by-r matrix)
- `W2 :: Matrix{T}`                      : inactive subspace (d-by-(d-r) matrix)

""" 
function find_subspaces(λ::Vector{T}, W::Matrix{T}, tol::Float64) where T <: Real
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    id = findall(x -> x .> tol, Σ)[end]
    W1 = W[:, 1:id] # active subspace
    W2 = W[:, id+1:end]  # inactive subspace

    return W1, W2
end


"""
function find_subspaces(W::Matrix{T}, ρθ::Distribution, tol::Int64) where T <: Real

Partitions active and inactive subspaces. The dimension of the active subspace is specified by the tol parameter. 

# Arguments
- `λ :: Vector{T}`                       : eigenvalues of d-by-d matrix
- `W :: Matrix{T}`                       : eigenvectors of d-by-d matrix
- `ρθ :: Distribution`                   : sampling density for the parameter θ
- `tol :: Int64`                         : cutoff dimension of active subspace (r)

# Outputs 
- `W1 :: Matrix{T}`                      : active subspace (d-by-r matrix)
- `W2 :: Matrix{T}`                      : inactive subspace (d-by-(d-r) matrix)

""" 
function find_subspaces(λ::Vector{T}, W::Matrix{T}, tol::Int64) where T <: Real
    W1 = W[:, 1:tol] # active subspace
    W2 = W[:, tol+1:end] # inactive subspace

    return W1, W2
end


"""
function compute_marginal(Wsub::Matrix{T}, ρθ::MvNormal) where T <: Real

Computes marginal density of the active/inactive variable, given a multivariate normal parameter density.

# Arguments
- `Wsub :: Matrix{T}`                    : active/inactive subspace
- `ρθ :: MvNormal`                       : sampling density for the parameter θ

# Outputs 
- `πsub :: MvNormal`                     : marginal density of active/inactive variable

""" 
function compute_marginal(Wsub::Matrix{T}, ρθ::MvNormal) where T <: Real
    μsub = Wsub' * ρθ.μ
    Σsub = Hermitian(Wsub' * ρθ.Σ * Wsub)
    πsub = MvNormal(μsub, Σsub)
    return πsub
end


"""
function compute_as(dθ::Vector{T}, ρθ::Distribution, tol::Real) where T <: Vector{<:Real}

Computes the active subspace from a set of feature vectors, their sampling density, and tolerance parameter. 

# Arguments
- `dθ :: Vector{T}`                      : samples of feature vector
- `ρθ :: Distribution`                   : sampling density for the parameter θ
- `tol :: Real`                          : tolerance to determine active dimension r

# Outputs 
- `as :: Subspace`                       : struct containing W1, W2, π_y, π_z  

""" 
function compute_as(dθ::Vector{T}, ρθ::Distribution, tol::Real) where T <: Vector{<:Real}
    C, λ, W = compute_eigenbasis(dθ)
    W1, W2 = find_subspaces(λ, W, tol)

    # sampling density of active variable y
    π_y = compute_marginal(W1, ρθ)

    # sampling density of inactive variable z
    π_z = compute_marginal(W2, ρθ)

    # save in struct
    as = Subspace(C, λ, W1, W2, π_y, π_z)
    return as
end


"""
function compute_as(C::Matrix{T}, ρθ::Distribution, tol::Real) where T <: Real

Computes the active subspace from a set of feature vectors, their sampling density, and tolerance parameter. 

# Arguments
- `C :: Matrix{T}`                       : covariance matrix
- `ρθ :: Distribution`                   : sampling density for the parameter θ
- `tol :: Real`                          : tolerance to determine active dimension r

# Outputs 
- `as :: Subspace`                       : struct containing W1, W2, π_y, π_z  

""" 
function compute_as(C::Matrix{T}, ρθ::Distribution, tol::Real) where T <: Real
    _, λ, W = compute_eigenbasis(C)
    W1, W2 = find_subspaces(λ, W, tol)

    # sampling density of active variable y
    π_y = compute_marginal(W1, ρθ)

    # sampling density of inactive variable z
    π_z = compute_marginal(W2, ρθ)

    as = Subspace(C, λ, W1, W2, π_y, π_z)
    return as
end


"""
function sample_as(n::Int64, as::Subspace)

Draws random samples from the active subspace.

# Arguments
- `n :: Int64`                           : number of samples
- `as :: Subspace`                       : struct containing W1, W2, π_y, π_z  

# Outputs 
- `ysamp :: Vector{Vector{Float64}}`     : samples of active variable
- `θsamp :: Vector{Vector{Float64}}`     : samples transformed into original parameter space

""" 
function sample_as(n::Int64, as::Subspace)
    ysamp = [rand(as.π_y) for i = 1:n] # sample active variable
    θsamp = transf_to_paramspace_fix.(ysamp, (as,))
    return ysamp, θsamp
end


"""
function transf_to_paramspace_fix(y::Vector, as::Subspace)

Transforms a sample from the active subspace (y) to the original parameter space (θ), fixing the inactive variable at the nominal (mean) value.

# Arguments
- `y :: Float or Vector`                 : sample from the active subspace
- `as :: Subspace`                       : struct containing W1, W2, π_y, π_z  

# Outputs 
- `θ :: Vector`                          : sample transformed to the original parameter space 

""" 
function transf_to_paramspace_fix(y::Vector, as::Subspace)
    θ = as.W1*y + as.W2*as.π_z.μ
    return θ
end

function transf_to_paramspace_fix(y::Float64, as::Subspace)
    y = [y]
    return transf_to_paramspace_fix(y, as)
end


"""
function transf_to_subspace(θ::Vector, as::Subspace)

Transforms a sample from the original parameter space (θ) to the active subspace (y).

# Arguments
- `θ :: Vector`                          : sample from the original parameter space 
- `as :: Subspace`                       : struct containing W1, W2, π_y, π_z  

# Outputs 
- `y :: Float or Vector`                 : sample transformed to the active subspace

""" 
function transf_to_subspace(θ::Vector, as::Subspace)
    y = as.W1'*θ
    return y
end


"""
function sample_as(n::Int64, as::Subspace)

Samples from the active subspace by marginalization over inactive variables.

# Arguments
- `n :: Int64`                           : number of samples
- `as :: Subspace`                       : struct containing W1, W2, π_y, π_z  

# Outputs 
- `ysamp :: Vector{Vector{Float64}}`     : samples of active variable
- `θsamp :: Vector{Vector{Float64}}`     : samples transformed into original parameter space

""" 
# function sample_as(n::Int64, as::Subspace)
#     # TO DO 
# end


export Subspace, compute_covmatrix, compute_eigenbasis, find_subspaces, compute_marginal, compute_as, sample_as, transf_to_paramspace_fix, transf_to_subspace
