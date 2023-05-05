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