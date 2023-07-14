include("subspace_metrics.jl")
include("sampling_metrics.jl")


function init_metrics_dict(n::Int64)
    metrics = Dict{String, Vector}()
    metrics["θsamp"] = Vector{Vector{Float64}}(undef, n)
    metrics["wvar"] = Vector{Float64}(undef, n)
    metrics["wESS"] = Vector{Float64}(undef, n)
    metrics["wdiag"] = Vector{Float64}(undef, n)
    return metrics
end


export ForstnerDistance, EuclideanDistance, WeightedSubspaceDistance, EffSampleSize, ISWeightVariance, ISWeightESS, ISWeightDiagnostic, init_metrics_dict