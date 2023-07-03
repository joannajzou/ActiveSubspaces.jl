"""
struct HMC <: Sampler

Defines the struct with parameters of the Hamiltonian Monte Carlo algorithm for sampling.

# Arguments
- `L :: Int`        : time length of integrating Hamiltonian's equations
- `ϵ :: Float64`    : step size

"""
struct HMC <: Sampler
    L :: Int                        # length of integrating Hamiltonian's equations
    ϵ :: Float64                    # step size
end


"""
function sample(lπ::Function, gradlπ::Function, sampler::HMC, n::Int64, x0::Vector{<:Real})

Draw samples from target distribution π(x) using Hamiltonian Monte Carlo (HMC) from the AdvancedHMC package. Assumes multi-dimensional support (x::Vector).

# Arguments
- `lπ :: Function`        : log likelihood of target π
- `gradlπ :: Function`    : gradient of log likelihood of π
- `sampler :: HMC`        : Sampler struct specifying sampling algorithm
- `n::Int64`              : number of samples (include number of samples for burn-in)
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