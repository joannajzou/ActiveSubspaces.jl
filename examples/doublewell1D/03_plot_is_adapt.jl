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
nsamp_arr = [625, 1250, 2500, 5000]
nsamptot = nsamp_arr[end]
neff = nsamptot
nrepl = 5

triallbls = ["MC", "ISM", "ISMw", "ISMa", "ISMaw"]
## covariance matrices
# A) load by file
Cref = JLD.load("data$modnum/DW1D_Ref.jld")["Cref"]
# Cref = JLD.load("data$modnum/DW1D_Ref_MC_nsamp=20000.jld")["C"]


C_arr = [import_data("data$modnum", lbl, nsamp_arr, nrepl; nsamptot=nsamptot) for lbl in triallbls]


# gradients 
repl = 1
∇Qref = JLD.load("data$modnum/repl$repl/DW1D_Quad_grad_nsamp=$nsamptot.jld")["∇Q"][1:neff]
∇Q_arr = [JLD.load("data$modnum/repl$repl/DW1D_$(lbl)_grad_nsamp=$nsamptot.jld")["∇Q"][1:neff] for lbl in triallbls]

# metrics 
met_arr = [JLD.load("data$modnum/repl$repl/DW1D_$(lbl)_cov_nsamp=$nsamptot.jld")["metrics"] for lbl in triallbls[2:end]]
θsamp = met_arr[1]["θsamp"]

# active subspaces 
asis_arr = [compute_as(∇Q, ρθ, 1) for ∇Q in ∇Q_arr]


# load biasing dist.
# πg = Gibbs(πgibbs2, θ=ρθ.μ)
λu = JLD.load("data$modnum/repl$repl/DW1D_ISM_cov_nsamp=$(nsamptot).jld")["mm_cent"]
mmu = MixtureModel(λu, πgibbs2)
λa = JLD.load("data$modnum/repl$repl/DW1D_ISMa_cov_nsamp=$(nsamptot).jld")["mm_cent"][end]
# αa = JLD.load("data$modnum/repl$repl/DW1D_ISaM_nsamp=$(nsamptot).jld")["mm_wts"][end]
mma = MixtureModel(λa, πgibbs2) # αa)

biasdist = [mmu, mma]


# compute error metrics ################################################################
println(" ====== Evaluate error ======")

val_arr = [compute_val(Cref, C, 1) for C in C_arr]


# define labels ###########################################################################
lab_arr = ["Ref.", "MC", "Mixture", "Mixture (v)", "Mixture (a)", "Mixture (av)"]
# lab_arr = ["Ref.", "MC", "Gibbs", "Mixture", "Mixture (adapt.)"]
# lab_bias_arr = ["Gibbs, β=0.005", "Gibbs, β=0.5", "Gibbs, β=1.0", "Mixture, β=0.5"]
lab_bias_arr = ["Mixture", "Mixture (v)", "Mixture (a)", "Mixture (av)"]
col_arr = [:mediumpurple1, :skyblue1, :seagreen, :goldenrod1, :darkorange2]


# plot biasing distributions #################################################################################

# plot pdfs
plot_multiple_pdfs(biasdist,
                    Vector(xplot),
                    lab_bias_arr[[1,3]],
                    col_arr[[2,4]],
                    normint = GQint,
                    ttl="Importance sampling biasing distributions")
# save("figures/biasingdist.png", fig)


# plot locations on parameter space
θ1_rng = log.(Vector(LinRange(exp.(θbd[1][1]), exp.(3.8), 51))) # θbd[1][2]
θ2_rng = log.(Vector(LinRange(exp.(θbd[2][1]), exp.(3.1), 51))) # θbd[2][2]
# θ1_rng = Vector(LinRange(θbd[1][1], θbd[1][2], 31))
# θ2_rng = Vector(LinRange(θbd[2][1], θbd[2][2], 31))
m = length(θ1_rng)
θ_plot = [[θi, θj] for θi in θ1_rng, θj in θ2_rng]
P_plot = [pdf(ρθ, [θi, θj]) for θi in θ1_rng, θj in θ2_rng]
Q_plot = [expectation([θi, θj], q, GQint) for θi in θ1_rng, θj in θ2_rng]

fa = plot_parameter_space(θ1_rng, θ2_rng, 
                    Q_plot, P_plot,
                    λu, λa)


# plot before & after adaptation
nλ = length(λu)
niter = 2
colscheme = [RGB((i-1)*0.7/niter, 0.3 + (i-1)*0.7/niter, 0.6 + (i-1)*0.2/niter) for i = 1:niter] # color scheme for model 1
figs = Vector{Figure}(undef, nλ)
for j = 1:nλ
    with_theme(custom_theme) do
        figs[j] = Figure(resolution = (600, 600))
        ax = Axis(figs[j][1, 1],  xlabel="x", ylabel="ϕ(x) π(x)")
        # target: integrand
        lines!(ax, xplot, integrand(xplot, λu[j], V, πgibbs0, GQint), linewidth=2, color=:black, label="target")
        # original distribution
        g0 = Gibbs(πgibbs0, β=0.5, θ=λu[j])
        lines!(ax, xplot, updf.((g0,), xplot) ./ normconst(g0, GQint), color=:red, label="original")
        # adapted distribution
        gk = Gibbs(πgibbs0, β=0.5, θ=λa[j])
        lines!(ax, xplot, updf.((gk,), xplot) ./ normconst(gk, GQint), color=colscheme[end], label="adapted")

        axislegend(ax)
    end
end


# plot active subspaces ##################################################################

# random samples
ny = 200
θsamp = [rand(ρθ) for i = 1:ny]   
θmat = reduce(hcat, θsamp)

fa = plot_parameter_space(θ1_rng, θ2_rng, 
                    Q_plot, P_plot,
                    θmat, asis_arr, col_arr, lab_arr[2:end]; nas=100)



# plot eigenspectrum ##################################################################
C_vec = [C[nsamptot][1] for C in C_arr]
C_vec = reduce(vcat, [[Cref], C_vec])
λ_vec = [compute_eigenbasis(C)[2] for C in C_vec]
f1 = plot_eigenspectrum(λ_vec, predlab=lab_arr)
f2 = plot_cossim(Cref, C_vec[2:end], lab_arr[2:end])



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
                val_arr,
                lab_arr[2:end],
                "d(C, Ĉ)", 
                "Forstner distance from reference covariance matrix",
                logscl=false
)
f3

f4 = plot_val_metric("WSD",
                val_arr,
                lab_arr[2:end],
                "d(W₁,Ŵ₁)", 
                "Weighted subspace distance",
                logscl=false
    )
# save("figures/WSD.png", f4)


f = plot_error_hist(∇Qref, ∇Q_arr, col_arr, lab_arr[2:end],
               "Error (log Euclid. dist.)", "Error in estimated ∇Q")

f = plot_error_scatter(∇Qref, ∇Q_arr, θsamp, col_arr, lab_arr[2:end],
                "Error (log Euclid. dist.)", "Error in estimated ∇Q")

# plot importance sampling diagnostics ###############################################
# metrics_u = JLD.load("data$modnum/repl1/DW1D_ISU_nsamp=$nsamptot.jld")["metrics"]


f5 = plot_IS_diagnostic("wvar", 
                met_arr,
                lab_bias_arr,
                "log(Var(w))",
                "Log variance of IS weights",
                logscl=true
)
# save("figures/scatter_wvar.png", f5)

f6 = plot_IS_diagnostic("wESS", 
            met_arr,
            lab_bias_arr,
            "ESS(w)/n",
            "Normalized effective sample size (ESS)",
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
                met_arr,
                lab_bias_arr,
                "P[ Var(w) > t ]",
                "Variance of IS weights",
                limtype=[1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                xticklab=["1e-1", "1e0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6", "1e7"],
)
# save("figures/cdf_wvar.png", f12)


f13 = plot_IS_cdf("wESS",
                met_arr,
                lab_bias_arr,
                "P[ ESS(w)/n < t ]",
                "ESS of IS weights",
                limtype = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                xticklab=[string(i) for i in 0:0.1:0.8], # [string(i) for i in 0:0.2:1],
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





# functions ###########################################################################

function integrand(x::Real, θ::Vector, V::Function, π0::Gibbs, normint::Integrator; β=1.0)
    πg = Gibbs(π0, β=β, θ=θ)
    return abs(V(x, θ)) * updf(πg, x) ./ normconst(πg, normint)
end

function integrand(x::Vector{<:Real}, θ::Vector, V::Function, π0::Gibbs, normint::Integrator; β=1.0)
    πg = Gibbs(π0, β=β, θ=θ)
    return abs.(V.(x, (θ,))) .* updf.((πg,), x) ./ normconst(πg, normint)
end