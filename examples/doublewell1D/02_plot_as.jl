using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using CairoMakie

include("model_param1.jl")
include("qoi_meanenergy.jl")
include("plotting_utils.jl")


function import_data(dir::String, IntType::String, nsamp_arr::Vector{Int64}, nrepl::Int64; nsamptot::Int64=nsamp_arr[end])
    Cdict = Dict{Int64, Vector{Matrix{Float64}}}()

    for nsamp in nsamp_arr
        Cdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)
        for j = 1:nrepl
            Cdict[nsamp][j] = JLD.load("$(dir)/repl$j/DW1D_$(IntType)_nsamp=$nsamptot.jld")["C"][nsamp]
        end
    end

    return Cdict
end

function import_data(dir::String, IntType::String, nsamp_arr::Vector{Int64}, nrepl::Int64, β::Float64; nsamptot::Int64=nsamp_arr[end])
    Cdict = Dict{Int64, Vector{Matrix{Float64}}}()

    for nsamp in nsamp_arr
        Cdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)
        for j = 1:nrepl
            Cdict[nsamp][j] = JLD.load("$(dir)/repl$j/DW1D_$(IntType)_nsamp=$nsamptot.jld")["C"][β][nsamp]
        end
    end

    return Cdict
end

function compute_eigen_(Cdict::Dict, nsamp_arr::Vector{Int64}, nrepl::Int64, rdim::Int64)
    # initialize
    λdict = Dict{Int64, Vector{Vector{Float64}}}()
    Wdict = Dict{Int64, Vector{Matrix{Float64}}}()

    # compute eigenvalues and eigenvectors
    for nsamp in nsamp_arr
        λdict[nsamp] = Vector{Vector{Float64}}(undef, nrepl)
        Wdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)

        for j = 1:nrepl
            _, λdict[nsamp][j], Wdict[nsamp][j] = select_eigendirections(Cdict[nsamp][j], rdim)
        end
    end

    return λdict, Wdict
end

function compute_val(Cref::Matrix{Float64}, Cdict::Dict, rdim::Int64)
    # extract values
    nsamp_arr = collect(keys(Cdict))
    nrepl = length(Cdict[nsamp_arr[1]])
    
    # compute eigenvalues and eigenvectors
    _, λref, _ = select_eigendirections(Cref, rdim)
    λdict, _ = compute_eigen_(Cdict, nsamp_arr, nrepl, rdim)

    # compute validation metrics
    val = Dict{String, Dict}()
    val["λ1_err"] = Dict{Int64, Vector{Float64}}()
    val["λ2_err"] = Dict{Int64, Vector{Float64}}()
    val["Forstner"] = Dict{Int64, Vector{Float64}}()
    val["WSD"] = Dict{Int64, Vector{Float64}}()

    for nsamp in nsamp_arr
        val["λ1_err"][nsamp] = [(λi[1] - λref[1]) / λref[1] for λi in λdict[nsamp]]
        val["λ2_err"][nsamp] = [(λi[2] - λref[2]) / λref[2] for λi in λdict[nsamp]]
        val["Forstner"][nsamp] = [ForstnerDistance(Cref, Ci) for Ci in Cdict[nsamp]]
        val["WSD"][nsamp] = [WeightedSubspaceDistance(Cref, Ci, rdim) for Ci in Cdict[nsamp]]
    end

    return val

end



# load data ##############################################################################
nsamp_arr = [1000, 2000, 4000, 8000, 16000]
nsamptot = nsamp_arr[end]
nrepl = 4

# biasing distribution
βarr = JLD.load("data1/repl1/DW1D_ISG_nsamp=$nsamptot.jld")["βarr"]
πg1 = Gibbs(πgibbs0, β=βarr[1], θ=[3,3])
πg2 = Gibbs(πgibbs0, β=βarr[2], θ=[3,3])
πg3 = Gibbs(πgibbs0, β=βarr[3], θ=[3,3])

πu = JLD.load("data1/repl1/DW1D_ISU_nsamp=$nsamptot.jld")["π_bias"]
ll = πu.a; ul = πu.b

# misc.
ξx, wx = gausslegendre(200, -10, 10)
d = 2

# covariance matrices
Cref = JLD.load("data1/DW1D_Ref.jld")["Cref"]
Cmc = import_data("data1", "MC", nsamp_arr, nrepl; nsamptot=nsamptot)
Cis_u = import_data("data1", "ISU", nsamp_arr, nrepl; nsamptot=nsamptot)
Cis_g1 = import_data("data1", "ISG", nsamp_arr, nrepl, βarr[1]; nsamptot=nsamptot)
Cis_g2 = import_data("data1", "ISG", nsamp_arr, nrepl, βarr[2]; nsamptot=nsamptot)
Cis_g3 = import_data("data1", "ISG", nsamp_arr, nrepl, βarr[3]; nsamptot=nsamptot)



# compute error metrics ##########################################################
println(" ====== Evaluate error ======")

val_mc = compute_val(Cref, Cmc, 1)
val_u = compute_val(Cref, Cis_u, 1)
val_g1 = compute_val(Cref, Cis_g1, 1)
val_g2 = compute_val(Cref, Cis_g2, 1)
val_g3 = compute_val(Cref, Cis_g3, 1)



# plot error metrics ##################################################################

f1 = plot_val_metric("λ1_err",
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "RE(λ - λ̂)", 
                "Rel. error in eigenvalue λ1(C)",
                logscl=false
)
f2 = plot_val_metric("λ2_err",
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "RE(λ - λ̂)", 
                "Rel. error in eigenvalue λ2(C)",
                logscl=false
)
f3 = plot_val_metric("Forstner",
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "d(C, Ĉ)", 
                "Forstner distance from reference covariance matrix",
                logscl=false
)
f4 = plot_val_metric("WSD",
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "d(C, Ĉ)", 
                "Weighted subspace distance (first eigenvector)",
                logscl=false
)


# plot importance sampling diagnostics ###############################################
metrics_u = JLD.load("data1/repl1/DW1D_ISU_nsamp=$nsamptot.jld")["metrics"]
metrics_g = JLD.load("data1/repl1/DW1D_ISG_nsamp=$nsamptot.jld")["metrics"]

f5 = plot_IS_diagnostic("wvar", 
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("IS, U[-5, 5]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "log(Var(w))",
                "Log variance of IS weights",
                logscl=true
)

f6 = plot_IS_diagnostic("wESS", 
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "ESS(w)",
                "Effective sample size (ESS)",
                logscl=false
)

f7 = plot_IS_diagnostic("wdiag", 
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "IS diag.",
                "IS diagnostic",
                logscl=true
)

    

# compute CDF plots #############################################################################

# compare MC sample size
# f8 = plot_IS_cdf("wvar",
#                 "ISU",
#                 nsamp_arr, nrepl_arr
# )

# f9 = plot_IS_cdf("wvar",
#                 "ISG",
#                 nsamp_arr, nrepl_arr,
#                 β=0.02
# )

# f10 = plot_IS_cdf("wvar",
#                 "ISG",
#                 nsamp_arr, nrepl_arr,
#                 β=0.2
# )

# f11 = plot_IS_cdf("wvar",
#                 "ISG",
#                 nsamp_arr, nrepl_arr,
#                 β=1.0
# )

# compare IS biasing distribution
f12 = plot_IS_cdf("wvar",
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "P[Var(w) > t]",
                "Variance of IS weights",
                limtype=(1e-4, 1e3)
)

f13 = plot_IS_cdf("wESS",
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "P[ESS(w) > t]",
                "ESS of IS weights",
                limtype = "max"
)

f14 = plot_IS_cdf("wdiag",
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "P[D(w) > t]",
                "IS Diagnostic",
                limtype = "max"
)



# plot extraordinary samples ####################################################################

th = 5 # 10^(1.0)
fig15 = plot_IS_samples(πgibbs0,
                "wvar", 
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                (πu, πg1, πg2, πg3),
                "Samples with high variance of IS weights",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=true
)

th = 10^(-0.6)
fig16 = plot_IS_samples(πgibbs0,
                "wESS", 
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                (πu, πg1, πg2, πg3),
                "Samples with low mod. ESS",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=false
)

th = 4.0
fig17 = plot_IS_samples(πgibbs0,
                "wdiag", 
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                (πu, πg1, πg2, πg3),
                "Samples with high IS diagnostic measure",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=true
)



