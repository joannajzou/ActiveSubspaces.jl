using ActiveSubspaces
using Distributions
using LinearAlgebra
using StatsBase
using JLD

# select model
modnum = 4
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("data_utils.jl")
include("plotting_utils.jl")



# load data ##############################################################################
nsamp_arr = [1000, 2000, 4000, 8000]
nsamptot = nsamp_arr[end]
nrepl = 5


## covariance matrices
# A) load by file
Cref = JLD.load("data$modnum/DW1D_Ref.jld")["Cref"]
# Cref = JLD.load("data$modnum/DW1D_Ref_MC_nsamp=20000.jld")["C"]
Cmc = import_data("data$modnum", "MC", nsamp_arr, nrepl; nsamptot=nsamptot)
# Cis_u = import_data("data$modnum", "ISU", nsamp_arr, nrepl; nsamptot=nsamptot)
Cis_g1 = import_data("data$modnum", "ISG", nsamp_arr, nrepl, βarr[1]; nsamptot=nsamptot)
Cis_g2 = import_data("data$modnum", "ISG", nsamp_arr, nrepl, βarr[2]; nsamptot=nsamptot)
Cis_g3 = import_data("data$modnum", "ISG", nsamp_arr, nrepl, βarr[3]; nsamptot=nsamptot)
Cis_m = import_data("data$modnum", "ISM", nsamp_arr, nrepl, 10.0; nsamptot=nsamptot)

# JLD.save("data$modnum/MCcovmatrices.jld",
#     "Cmc", Cmc,
#     "Cis_u", Cis_u,
#     "Cis_g1", Cis_g1,
#     "Cis_g2", Cis_g2,
#     "Cis_g3", Cis_g3,

#     )

# or B) load consolidated data
# Cref = JLD.load("data$modnum/DW1D_Ref.jld")["Cref"]
# Cmc = JLD.load("data$modnum/MCcovmatrices.jld")["Cmc"]
# Cis_u = JLD.load("data$modnum/MCcovmatrices.jld")["Cis_u"]
# Cis_g1 = JLD.load("data$modnum/MCcovmatrices.jld")["Cis_g1"]
# Cis_g2 = JLD.load("data$modnum/MCcovmatrices.jld")["Cis_g2"]
# Cis_g3 = JLD.load("data$modnum/MCcovmatrices.jld")["Cis_g3"]

data = JLD.load("data$modnum/mixturedist.jld") 
πc = [Gibbs(πgibbs0, β=data["temp"], θ=c) for c in data["centers"]]
mm = MixtureModel(πc, data["weights"])


# compute error metrics ################################################################
println(" ====== Evaluate error ======")

val_mc = compute_val(Cref, Cmc, 1)
# val_u = compute_val(Cref, Cis_u, 1)
val_g1 = compute_val(Cref, Cis_g1, 1)
val_g2 = compute_val(Cref, Cis_g2, 1)
val_g3 = compute_val(Cref, Cis_g3, 1)
val_m = compute_val(Cref, Cis_m, 1)


# plot biasing distributions #################################################################################

πg = [Gibbs(πgibbs0, β=βi, θ=ρθ.μ) for βi in βarr]
plot_multiple_pdfs([πg[1], πg[2], πg[3], mm],
                    Vector(xplot), ξx, wx,
                    ["Gibbs, β=0.005", "Gibbs, β=0.5", "Gibbs, β=1.0", "Mixture, β=0.5"],
                    [:skyblue1, :seagreen, :goldenrod1, :darkorange2],
                    ttl="Importance sampling biasing distributions")
# save("figures/biasingdist.png", fig)



# plot eigenspectrum ##################################################################
C_tup = (Cref, Cmc[8000][1], Cis_g1[8000][1], Cis_g2[8000][1], Cis_g3[8000][1], Cis_m[8000][1])
C_lab = ("Ref.", "MC", "Gibbs, β=0.005", "Gibbs, β=0.5", "Gibbs, β=1.0", "Mixture, β=0.5")
λ_vec = [compute_eigenbasis(C)[2] for C in C_tup]
f1 = plot_eigenspectrum(λ_vec, predlab=C_lab)
f2 = plot_cossim(Cref, C_tup[2:end], C_lab[2:end])



# plot error metrics ##################################################################

# f1 = plot_val_metric("λ1_err",
#                 (val_mc, val_u, val_g1, val_g2, val_g3),
#                 ("MC","IS, U[-3, 3]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
#                 "RE(λ - λ̂)", 
#                 "Rel. error in eigenvalue λ1(C)",
#                 logscl=false
# )

# f2 = plot_val_metric("λ2_err",
#                 (val_mc, val_u, val_g1, val_g2, val_g3),
#                 ("MC","IS, U[-3, 3]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
#                 "RE(λ - λ̂)", 
#                 "Rel. error in eigenvalue λ2(C)",
#                 logscl=false
# )

# f3 = plot_val_metric("Forstner",
#                 (val_mc, val_u, val_g1, val_g2, val_g3, val_m),
#                 ("MC","IS, U[-3, 3]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)", "IS, Mixture"),
#                 "d(C, Ĉ)", 
#                 "Forstner distance from reference covariance matrix",
#                 logscl=false
# )
# f3

f3 = plot_val_metric("Forstner",
                (val_mc, val_g1, val_g2, val_g3, val_m),
                ("MC", "IS, Gibbs (β=0.005)", "IS, Gibbs (β=0.5)", "IS, Gibbs (β=1.0)", "IS, Mixture (β=0.5)"),
                "d(C, Ĉ)", 
                "Forstner distance from reference covariance matrix",
                logscl=false
)
f3

f4 = plot_val_metric("WSD",
                # (val_mc, val_u, val_g1, val_g2, val_g3, val_m),
                # ("MC","IS, U[-3, 3]", "IS, Gibbs (β=0.02)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)", "IS, Mixture"),
                (val_mc, val_g1, val_g2, val_g3, val_m),
                ("MC", "IS, Gibbs (β=0.005)", "IS, Gibbs (β=0.5)", "IS, Gibbs (β=1.0)", "IS, Mixture (β=0.5)"),
                "d(W₁,Ŵ₁)", 
                "Weighted subspace distance",
                logscl=false
    )
# save("figures/WSD.png", f4)



# plot importance sampling diagnostics ###############################################
# metrics_u = JLD.load("data$modnum/repl1/DW1D_ISU_nsamp=$nsamptot.jld")["metrics"]
metrics_g = JLD.load("data$modnum/repl2/DW1D_ISG_nsamp=$nsamptot.jld")["metrics"]
metrics_m = JLD.load("data$modnum/repl2/DW1D_ISM_nsamp=$nsamptot.jld")["metrics"][10.0]

# project onto 2D with active subspace
as = compute_as(Cref, ρθ, 2)

f5 = plot_IS_diagnostic("wvar", 
                # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0], metrics_m),
                # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)", "Mixture"),
                (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
                ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
                "log(Var(w))",
                "Log variance of IS weights",
                as.W1,
                logscl=true
)
# save("figures/scatter_wvar.png", f5)

f6 = plot_IS_diagnostic("wESS", 
            # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0], metrics_m),
            # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)", "Mixture"),
            (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
            ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
            "ESS(w)/n",
            "Normalized effective sample size (ESS)",
            as.W1,
            logscl=false
)
# save("figures/scatter_wess.png", f6)

# f7 = plot_IS_diagnostic("wdiag", 
#                 (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
#                 ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
#                 "IS diag.",
#                 "IS diagnostic",
#                 logscl=true
# )

    

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
                # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0], metrics_m),
                # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)", "Mixture"),
                (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
                ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
                "P[ Var(w) > t ]",
                "Variance of IS weights",
                limtype=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                xticklab=["1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3"],
)
# save("figures/cdf_wvar.png", f12)


f13 = plot_IS_cdf("wESS",
                # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0], metrics_m),
                # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)", "Mixture"),
                (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
                ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
                "P[ ESS(w)/n < t ]",
                "ESS of IS weights",
                limtype = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                xticklab=[string(i) for i in 0:0.1:0.5], # [string(i) for i in 0:0.2:1],
                rev = false,
                logscl=false
)
# save("figures/cdf_wess.png", f13)


# f14 = plot_IS_cdf("wdiag",
#                 (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
#                 ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
#                 "P[D(w) > t]",
#                 "IS Diagnostic",
#                 limtype = "max"
# )



# plot extraordinary samples ####################################################################

th = 80 # 10^(1.0) # threshold
fig15 = plot_IS_samples(πgibbs0,
                "wvar", 
                # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                # (πu, πg1, πg2, πg3),
                (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
                ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
                (πg[1], πg[2], πg[3], mm),
                "Samples with high variance of IS weights",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=true
)
save("figures/badsamp_war.png", fig15)


th = 0.05
fig15b = plot_IS_samples(πgibbs0,
                "wvar", 
                # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                # (πu, πg1, πg2, πg3),
                (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
                ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
                (πg[1], πg[2], πg[3], mm),
                "Samples with low variance of IS weights",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=false
)
save("figures/badsamp_war.png", fig15)


th = 0.001 # 10^(-0.6) # threshold
fig16 = plot_IS_samples(πgibbs0,
                "wESS", 
                # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                # (πu, πg1, πg2, πg3),
                (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
                ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
                (πg[1], πg[2], πg[3], mm),
                "Samples with low ESS",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=false
)

th = 0.25 # 10^(-0.6) # threshold
fig16 = plot_IS_samples(πgibbs0,
                "wESS", 
                # (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                # ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                # (πu, πg1, πg2, πg3),
                (metrics_g[0.005], metrics_g[0.5], metrics_g[1.0], metrics_m),
                ("Gibbs (β=0.005)", "Gibbs (β=0.5)", "Gibbs (β=1.0)", "Mixture (β=0.5)"),
                (πg[1], πg[2], πg[3], mm),
                "Samples with high ESS",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=true
)


th = 4.0 # threshold
fig17 = plot_IS_samples(πgibbs0,
                "wdiag", 
                (metrics_u, metrics_g[0.02], metrics_g[0.2], metrics_g[1.0]),
                ("U[-3, 3]", "Gibbs (β=0.02)", "Gibbs (β=0.2)", "Gibbs (β=1.0)"),
                (πu, πg1, πg2, πg3),
                "Samples with high IS diagnostic measure",
                [ll, ul],
                (ξx, wx),
                thresh=th,
                rev=true
)


