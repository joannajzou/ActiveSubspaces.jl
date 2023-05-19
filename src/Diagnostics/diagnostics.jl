include("subspace_metrics.jl")
include("sampling_metrics.jl")

function init_metrics_dict(nrepl)
    metrics = Dict{String, Vector{Vector}}()
    metrics["θsamp"] = Vector{Vector{Float64}}(undef, nrepl)
    metrics["wvar"] = Vector{Vector{Float64}}(undef, nrepl)
    metrics["wESS"] = Vector{Vector{Float64}}(undef, nrepl)
    metrics["wdiag"] = Vector{Vector{Float64}}(undef, nrepl)
    metrics["w̃ESS"] = Vector{Vector{Float64}}(undef, nrepl)
    return metrics
end

export ForstnerDistance, WeightedSubspaceDistance, EffSampleSize, ISWeightVariance, ISWeightESS, ISWeightDiagnostic