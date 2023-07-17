
"""
function ForstnerDistance(A::Matrix{<:Real}, B::Matrix{<:Real}; α=1e-8) 

Computes Forstner distance between two covariance matrices. 

# Arguments
- `A :: Matrix{<:Real}`   : first n-by-n covariance matrix
- `B :: Matrix{<:Real}`   : second n-by-n covariance matrix
- `α :: Float64`          : regularization parameter


# Outputs
- `d :: Float64`    : Forstner distance between A and B
"""
function ForstnerDistance(A::Matrix{<:Real}, B::Matrix{<:Real}; α=1e-8) 
    A = Symmetric(A); B = Symmetric(B)
    dist = Forstner(α=α)
    return compute_distance(A, B, dist)
end


"""
function EuclideanDistance(A::Matrix{<:Real}, B::Matrix{<:Real})

Computes (isotropic) Euclidean distance between two covariance matrices. 

# Arguments
- `A :: Matrix{<:Real}`   : first n-by-n covariance matrix
- `B :: Matrix{<:Real}`   : second n-by-n covariance matrix

# Outputs
- `d :: Float64`    : Euclidean distance between A and B
"""
function EuclideanDistance(A::Matrix{<:Real}, B::Matrix{<:Real})
    A = Symmetric(A); B = Symmetric(B)
    n = size(A,1)
    dist = Euclidean(n)
    return compute_distance(A, B, dist)
end


"""
function EuclideanDistance(A::Vector{<:Real}, B::Vector{<:Real})

Computes (isotropic) Euclidean distance between two vectors. 

# Arguments
- `A :: Vector{<:Real}`   : first n-vector
- `B :: Vector{<:Real}`   : second n-vector

# Outputs
- `d :: Float64`    : Euclidean distance between A and B
"""
function EuclideanDistance(A::Vector{<:Real}, B::Vector{<:Real})
    n = length(A)
    dist = Euclidean(n)
    return compute_distance(A, B, dist)
end


"""
function WeightedSubspaceDistance(A::Matrix, B::Matrix, rA::Int64, rB::Int64=rA)

Computes weighted subspace distance between r-dimensional subspaces of covariance matrices A and B

# Arguments
- `A :: Matrix`   : first n-by-n covariance matrix
- `B :: Matrix`   : second n-by-n covariance matrix
- `rA :: Int64`   : dimension of subspace of A (rA < n)
- `rB :: Int64`   : dimension of subspace of B (rB < n)

# Outputs
- `d :: Float64`    : weighted subspace distance between A and B
"""
function WeightedSubspaceDistance(A::Matrix, B::Matrix, rA::Int64; rB::Int64=rA)
    
    # compute subspace of A
    _, γA, WA = compute_eigenbasis(A)
    γA = γA[1:rA]; ψA = WA[:, 1:rA]
    DA = Diagonal(γA ./ sum(γA))^(1/4) # diagonal matrix of weights

    # compute subspace of B
    _, γB, WB = compute_eigenbasis(B)
    γB = γB[1:rB]; ψB = WB[:, 1:rB]
    DB = Diagonal(γB ./ sum(γB))^(1/4) # diagonal matrix of weights

    return sqrt( 1 - norm( (ψA * DA)' * (ψB * DB) )^2 )

end
