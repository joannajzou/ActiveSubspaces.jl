using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using CairoMakie

include("model_param1.jl")
include("qoi_meanenergy.jl")
include("plotting_utils.jl")

function init_eig_arrays(d, nsamp_arr)
    λarr = Matrix{Float64}(undef, (d, length(nsamp_arr)))
    Warr = Vector{Matrix{Float64}}(undef, length(nsamp_arr))
    return λarr, Warr
end

function compute_val(nsamp_arr::Vector{Int64}, λref::Vector{Float64}, Cref::Matrix{Float64}, λmc::Matrix{Float64}, Cmc::Dict)
    val = Dict{String, Vector{Float64}}()
    val["λ1_err"] = [(λmc[1,i] - λref[1]) / λref[1] for i = 1:length(nsamp_arr)]
    val["λ2_err"] = [(λmc[2,i] - λref[2]) / λref[2] for i = 1:length(nsamp_arr)]
    val["SS_err"] = [ForstnerDistance(Cref, Cmc[nsamp]) for nsamp in nsamp_arr]
    val["WSD"] = [WeightedSubspaceDistance(Cref, Cmc[nsamp], 1) for nsamp in nsamp_arr]
    return val
end



# load data ##############################################################################
Cref = JLD.load("data1/DW1D_Ref.jld")["Cref"]

nsamp_arr = [1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000]
nrepl_arr = [1, 10, 1, 1, 1, 1, 1, 1]

Cmc = Dict{Int64, Matrix{Float64}}()
Cis_u = Dict{Int64, Matrix{Float64}}()
Cis_g1 = Dict{Int64, Matrix{Float64}}()
Cis_g2 = Dict{Int64, Matrix{Float64}}()
Cis_g3 = Dict{Int64, Matrix{Float64}}()

for i = 1:length(nsamp_arr)
    nsamp = nsamp_arr[i]
    nrepl = nrepl_arr[i]

    Cmc[nsamp] = mean(JLD.load("data1/DW1D_MC_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"])
    Cis_u[nsamp]= mean(JLD.load("data1/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"])
    Cis_g1[nsamp]= mean(JLD.load("data1/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][0.01])
    Cis_g2[nsamp]= mean(JLD.load("data1/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][0.2])
    Cis_g3[nsamp]= mean(JLD.load("data1/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][1.0])
end

nsamp = 8000; nrepl = 1
πu = JLD.load("data1/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld")["π_bias"]
ll = πu.a; ul = πu.b
βarr = JLD.load("data1/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["βarr"]
d = 2
ξx, wx = gausslegendre(200, -10, 10)

πg1 = Gibbs(πgibbs0, β=0.01, θ=[3,3])
πg2 = Gibbs(πgibbs0, β=0.2, θ=[3,3])
πg3 = Gibbs(πgibbs0, β=1.0, θ=[3,3])



# compute eigendecomposition ##########################################################
println(" ====== Evaluate error ======")

# reference
_, λref, Wref = select_eigendirections(Cref, d)

# MC estimates
λmc, Wmc = init_eig_arrays(d, nsamp_arr)
λu, Wu = init_eig_arrays(d, nsamp_arr)
λg1, Wg1 = init_eig_arrays(d, nsamp_arr)
λg2, Wg2 = init_eig_arrays(d, nsamp_arr)
λg3, Wg3 = init_eig_arrays(d, nsamp_arr)

for i = 1:length(nsamp_arr)
    nsamp = nsamp_arr[i]
    _, λmc[:,i], Wmc[i] = select_eigendirections(Cmc[nsamp], d)
    _, λu[:,i], Wu[i] = select_eigendirections(Cis_u[nsamp], d)
    _, λg1[:,i], Wg1[i] = select_eigendirections(Cis_g1[nsamp], d)
    _, λg2[:,i], Wg2[i] = select_eigendirections(Cis_g2[nsamp], d)
    _, λg3[:,i], Wg3[i] = select_eigendirections(Cis_g3[nsamp], d)
end



# compute error metrics ###############################################################
val_mc = compute_val(nsamp_arr, λref, Cref, λmc, Cmc)
val_u = compute_val(nsamp_arr, λref, Cref, λu, Cis_u)
val_g1 = compute_val(nsamp_arr, λref, Cref, λg1, Cis_g1)
val_g2 = compute_val(nsamp_arr, λref, Cref, λg1, Cis_g2)
val_g3 = compute_val(nsamp_arr, λref, Cref, λg1, Cis_g3)



# plot error metrics ##################################################################

f1 = plot_val_metric("λ1_err", nsamp_arr,
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "RE(λ - λ̂)", 
                "Rel. error in eigenvalue λ1(C)"
)
f2 = plot_val_metric("λ2_err", nsamp_arr,
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "RE(λ - λ̂)", 
                "Rel. error in eigenvalue λ2(C)"
)
f3 = plot_val_metric("SS_err", nsamp_arr,
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "d(C, Ĉ)", 
                "Forstner distance from reference covariance matrix",
                logscl=false
)
f4 = plot_val_metric("WSD", nsamp_arr,
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "d(C, Ĉ)", 
                "Weighted subspace distance (first eigenvector)",
                logscl=false
)


# plot importance sampling diagnostics ###############################################
nsamp = 12000; nrepl = 1
metrics_u = JLD.load("data1/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
metrics_g = JLD.load("data1/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]

f5 = plot_IS_diagnostic("wvar", 
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "log(Var(w))",
                "Log variance of IS weights",
                logscl=true
)

f6 = plot_IS_diagnostic("wESS", 
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "ESS(w)",
                "Effective sample size (ESS)",
                logscl=false
)

f7 = plot_IS_diagnostic("wdiag", 
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "IS diag.",
                "IS diagnostic",
                logscl=true
)

f8 = plot_IS_diagnostic("w̃ESS", 
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "Mod. ESS(w)",
                "Modified effective sample size (ESS)",
                logscl=false
)

    

# compute CDF plots #############################################################################

# compare MC sample size
f9 = plot_IS_cdf("wvar",
                "ISU",
                nsamp_arr, nrepl_arr
)

f10 = plot_IS_cdf("wvar",
                "ISG",
                nsamp_arr, nrepl_arr,
                β=0.01
)

f11 = plot_IS_cdf("wvar",
                "ISG",
                nsamp_arr, nrepl_arr,
                β=0.2
)

f11 = plot_IS_cdf("wvar",
                "ISG",
                nsamp_arr, nrepl_arr,
                β=1.0
)

# compare IS biasing distribution
f12 = plot_IS_cdf("wvar",
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "Variance of IS weights",
                limittype = "med"
)

f13 = plot_IS_cdf("wESS",
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "ESS of IS weights",
                limittype = "max"
)

f14 = plot_IS_cdf("w̃ESS",
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "Mod. ESS of IS weights",
                limittype = "max"
)

f15 = plot_IS_cdf("wdiag",
                (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                "IS Diagnostic",
                limittype = "max"
)



# plot extraordinary samples ####################################################################


figs = Vector{Figure}(undef, length(nsamp_arr))
for i = 1:length(nsamp_arr)
    nsamp = nsamp_arr[i]
    nrepl = nrepl_arr[i]

    metrics_u = JLD.load("data1/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
    metrics_g = JLD.load("data1/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]

    # th = 1e1
    # figs[i] = plot_IS_samples(πgibbs0,
    #                 "wvar", 
    #                 (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
    #                 ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
    #                 (πu, πg1, πg2, πg3),
    #                 "Samples with high variance of IS weights",
    #                 [ll, ul],
    #                 (ξx, wx),
    #                 thresh=th,
    #                 rev=true
    # )

    # th = 10^(-0.5)
    # figs[i] = plot_IS_samples(πgibbs0,
    #                 "wESS", 
    #                 (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
    #                 ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
    #                 (πu, πg1, πg2, πg3),
    #                 "Samples with low ESS",
    #                 [ll, ul],
    #                 (ξx, wx),
    #                 thresh=th,
    #                 rev=false
    # )

    # th = 10^(-1)
    # figs[i] = plot_IS_samples(πgibbs0,
    #                 "w̃ESS", 
    #                 (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
    #                 ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
    #                 (πu, πg1, πg2, πg3),
    #                 "Samples with low mod. ESS",
    #                 [ll, ul],
    #                 (ξx, wx),
    #                 thresh=th,
    #                 rev=false
    # )

    th = 4.0
    figs[i] = plot_IS_samples(πgibbs0,
                    "wdiag", 
                    (metrics_u, metrics_g[0.01], metrics_g[0.2], metrics_g[1.0]),
                    ("U[-5, 5]", "Gibbs (β=0.01)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                    (πu, πg1, πg2, πg3),
                    "Samples with high IS diagnostic measure",
                    [ll, ul],
                    (ξx, wx),
                    thresh=th,
                    rev=true
    )

end



# misc. plots ###################################################################
col=[:black, :purple, :blue, :green, :orange, :red]         # set color scheme
xplot = LinRange(ll, ul, 1000)


# plot biasing distributions
fig = Figure(resolution = (700, 600))
ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)",
          title="Importance sampling biasing distributions")

lines!(ax, xplot, pdf.(πu, xplot), color=col[2], label="U[$ll, $ul]")
for i = 1:length(βarr)
    βi = βarr[i]
    πg = Gibbs(πgibbs0, β=βi, θ=[3,3])
    lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, ξx, wx), color=col[i+2], label="Gibbs, β=$βi")
end
axislegend(ax)
fig


# plot distribution of samples
θ1rng = LinRange(0.5, 5.5, 100); θ2rng = θ1rng              # grid across θ-domain
c_mat = [pdf(ρθ, [θi, θj]) for θi in θ1rng, θj in θ2rng]   # PDF values                                  

fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],  xlabel="θ1", ylabel="θ2",
          title="Distribution of samples (nsamp=$nsamp)")

scatter!(θmat[1,:], θmat[2,:], color=(:blue, 0.1))
contour!(ax, θ1rng, θ2rng, c_mat, levels=0:0.1:1)
fig


# plot samples from original sampling density
xplot = LinRange(-5, 5, 1000)
fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)",
          title="Gibbs distribution of samples")
for θi in θsamp[1:10:1000]
    πg = Gibbs(πgibbs0, β=1.0, θ=θi)
    lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, ξx, wx))
end
fig

# plot samples from active subspace




## plot variation with q
nMC = 10000                                 # number of MC/MCMC samples
eps = 1e-1                                  # step size 
nuts = NUTS(eps)    

Qmc(θ) = Q(θ, nuts, nsamp=nMC)