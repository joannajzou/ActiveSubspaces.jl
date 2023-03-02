struct PCA{T <: Real} <: Subspace
    tol :: T
end

function PCA(; tol = 0.01)
    PCA(tol)
end

function find_subspace(ds::Vector, pca::PCA)
    λ, W = select_eigendirections(ds, pca.tol)
    return λ, W
end