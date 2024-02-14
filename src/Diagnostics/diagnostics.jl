include("subspace_metrics.jl")
include("sampling_metrics.jl")



"""
function compute_metrics(θ::Vector, h::Vector, w::Vector)

Computes importance sampling diagnostic metrics, including variance of IS weights ("wvar"), ESS of IS weights ("wESS"), and diagnostic measure ("wdiag").

# Arguments
- `θ :: Vector`                 : vector of parameters
- `h :: Vector`                 : evaluations of summand function h(x) at samples
- `w :: Vector`                 : evaluations of importance sampling weights at samples

# Outputs
- `metrics :: Dict{String, Vector}`        : dictionary of importance sampling diagnostics

"""
function compute_metrics(θ::Vector, h::Vector, w::Vector)
    nsamp = length(θ)
    n = length(h[1])
    
    w̃ = [[h[i][j] * w[i][j] for j = 1:n] for i = 1:nsamp]

    metrics = Dict{String, Vector}()
    metrics["θsamp"] = θ
    metrics["wvar"] = [norm(ISWeightVariance(wi)) for wi in w̃]
    metrics["wESS"] = [ISWeightESS(wi)/n for wi in w̃]
    metrics["wdiag"] = [ISWeightDiagnostic(wi) for wi in w]

    return metrics
end

export compute_metrics
export MCSE, MCSEbm, MCSEobm, EffSampleSize, ISWeightVariance, ISWeightESS, ISWeightDiagnostic 
export ForstnerDistance, EuclideanDistance, WeightedSubspaceDistance