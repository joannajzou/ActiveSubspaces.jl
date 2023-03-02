struct AS{T<:Real} <: Subspace
    ∇Q :: Function 
    tol :: T 
end

function AS(∇Q::Function; tol = 0.01)
    ActiveSubspace(∇Q, tol)
end

function find_subspace(ds::Vector, as::AS)
    ∇Qθ = [as.∇Q(θi) for θi in ds]
    λ, W = select_eigendirections(∇Qθ, as.tol)
    return λ, W
end