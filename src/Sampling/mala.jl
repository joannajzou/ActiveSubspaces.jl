"""
struct MALA <: Sampler

Defines the struct with parameters of the Metropolis-Adjusted Langevin Algorithm (MALA) for sampling.

# Arguments
- `L :: Int`        : time length between evaluating accept/reject criterion
- `ϵ :: Real`       : step size

"""
struct MALA <: Sampler
    L :: Int                        # length of integrating Hamiltonian's equations
    ϵ :: Real                       # step size
end


# 1-D state
function propose(x0::Real, lπ::Function, gradlπ::Function, mala::MALA)
    x = x0
    for i = 1:mala.L
        x = x + gradlπ(x) * mala.ϵ + sqrt(2.0*mala.ϵ)*randn() # MALA; gradient-guided random walk
    end

    ΔV = lπ(x) - lπ(x0)    # lπ = log of prob distribution
    flag = accept_or_reject(ΔV)
    return flag ? x : x0, flag                                                         ### ternary operator
end

# N-D state
function propose(x0::Vector, lπ::Function, gradlπ::Function, mala::MALA)
    m = length(x0)
    x = x0
    for i = 1:mala.L
        x = x + gradlπ(x) * mala.ϵ + sqrt(2.0*mala.ϵ)*randn(m) # MALA; gradient-guided random walk
    end

    ΔV = lπ(x) - lπ(x0)    # lπ = log of prob distribution
    flag = accept_or_reject(ΔV)
    return flag ? x : x0, flag                                                         ### ternary operator
end

