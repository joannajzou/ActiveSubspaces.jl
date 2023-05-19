
"""
function ForstnerDistance(A::Matrix, B::Matrix) 

Computes Forstner distance between two covariance matrices. 

# Arguments
- `A :: Matrix`   : first n-by-n covariance matrix
- `B :: Matrix`   : second n-by-n covariance matrix

# Outputs
- `d :: Float64`    : Forstner distance between A and B
"""
function ForstnerDistance(A::Matrix, B::Matrix) 
    A = Hermitian(A); B = Hermitian(B)
    Ap = sqrt(inv(A))
    λ = eigvals( Ap * B * Ap )
    return sqrt(sum( log.(λ).^2 ))
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
    γA, ψA = compute_subspace(A, rA)
    DA = Diagonal(γA ./ sum(γA)) # diagonal matrix of weights

    # compute subspace of B
    γB, ψB = compute_subspace(B, rB)
    DB = Diagonal(γB ./ sum(γB)) # diagonal matrix of weights

    return sqrt( 1 - norm( (ψA * DA)' * (ψB * DB) )^2 )

end


"""
function compute_subspace(A::Matrix, r::Int64)

Computes weighted subspace distance between r-dimensional subspaces of covariance matrices A and B

# Arguments
- `A :: Matrix`   : n-by-n covariance matrix
- `rA :: Int64`   : dimension of subspace of A (rA < n)

# Outputs
- `λA :: Vector{Float64}`   : first rA eigenvalues of A
- `ΦA :: Matrix{Float64}`   : first rA eigenvectors of A
"""
function compute_subspace(A::Matrix, r::Int64)
    
    # compute eigendecomposition of A
    A = Hermitian(A) 
    λA = eigvals(A)
    ΦA = eigvecs(A)

    return λA[1:r], ΦA[:,1:r]
end