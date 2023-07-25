"""
struct HMC <: Sampler

Defines the struct with parameters of the Hamiltonian Monte Carlo algorithm for sampling.

# Arguments
- `L :: Int`        : time length of integrating Hamiltonian's equations
- `ϵ :: Real`       : step size

"""
struct HMC <: Sampler
    L :: Int                        # length of integrating Hamiltonian's equations
    ϵ :: Real                    # step size
end


"""
function sample(lπ::Function, gradlπ::Function, sampler::HMC, n::Int64, x0::Vector{<:Real})

Draw samples from target distribution π(x) using Hamiltonian Monte Carlo (HMC) from the AdvancedHMC package. Assumes multi-dimensional support (x::Vector).

# Arguments
- `lπ :: Function`        : log likelihood of target π
- `gradlπ :: Function`    : gradient of log likelihood of π
- `sampler :: HMC`        : Sampler struct specifying sampling algorithm
- `n::Int64`              : number of samples
- `x0::Vector{<:Real}`    : initial state

# Outputs
- `samples::Vector{Vector}`    : vector of samples from π

"""
# multi-d x: use AdvancedHMC
function sample(lπ::Function, gradlπ::Function, sampler::HMC, n::Int64, x0::Vector{<:Real})
    D = length(x0)
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, lπ, gradlπ)

    integrator = Leapfrog(sampler.ϵ)
    proposal = HMCKernel(StaticTrajectory(integrator, sampler.L))
    samples, _ = AdvancedHMC.sample(hamiltonian, proposal, x0, n; progress=false)

    return samples
end


# 1-d x: proposal function
function propose(x0::Real, lπ::Function, gradlπ::Function, hmc::HMC)
    x = x0
    p = randn() # sample momentum
    H0 = lπ(x0) .- 0.5 * p' * p # should this be +? # Euclidean-Gaussian kinetic energy # integrate in phase space

    # Start leapfrog
    p += 0.5 * hmc.ϵ * gradlπ(x)
    for i = 1:hmc.L-1
        x -= hmc.ϵ * p
        p += hmc.ϵ * gradlπ(x)
    end
    x -= hmc.ϵ * p
    p += 0.5 * hmc.ϵ * gradlπ(x)
    p = -p # revert negative momentum

    H = lπ(x) .- 0.5 * p' * p
    flag = accept_or_reject(H - H0)
    return flag ? x : x0, flag
end