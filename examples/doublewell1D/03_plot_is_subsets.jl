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
include("mc_utils.jl")
include("adaptive_is_utils.jl")
include("plotting_utils.jl")



# load data ##############################################################################
splits = [11, 12, 13, 14, 15, 16, 17, 18,
          21, 22, 23, 24, 25, 26, 27, 28,
          31, 32, 33, 34, 35, 36, 37, 38,
          41, 42, 43, 44, 45, 46, 47, 48,
          51, 52, 53, 54, 55, 56, 57, 58]

nsamp_arr = [250, 500, 1000, 2000]
nsamptot = 8000
triallbls = ["MC", "ISM", "ISMa", "ISMaw"]
lab_arr = ["Ref.", "MC", "Mixture", "Mixture (a)", "Mixture (aw)"]


# load gradients
∇Qref = [JLD.load("data$modnum/subsets/DW1D_Quad_grad_split=$spnum.jld")["∇Q"] for spnum in splits]
∇Qmc = [JLD.load("data$modnum/subsets/DW1D_MC_grad_split=$spnum.jld")["∇Q"] for spnum in splits]
∇Qis = [JLD.load("data$modnum/subsets/DW1D_ISM_grad_split=$spnum.jld")["∇Q"] for spnum in splits]
∇Qisa = [JLD.load("data$modnum/subsets/DW1D_ISMa_grad_split=$spnum.jld")["∇Q"] for spnum in splits]
∇Qisaw = [JLD.load("data$modnum/subsets/DW1D_ISMaw_grad_split=$spnum.jld")["∇Q"] for spnum in splits]

# load biasing distributions
λu = [JLD.load("data$modnum/repl$j/DW1D_ISM_cov_nsamp=$(nsamptot).jld")["mm_cent"] for j = 1:5]
mmu = [MixtureModel(λ, πgibbs2) for λ in λu]
λa = [JLD.load("data$modnum/repl$j/DW1D_ISMa_cov_nsamp=$(nsamptot).jld")["mm_cent"][end] for j = 1:5]
mma = [MixtureModel(λ, πgibbs2) for λ in λa]

# compute the Monte Carlo estimate at biasing dist centers
∇Qu = [compute_gradQ(λ, q, MCint) for λ in λu]
# compute the IS estimate at biasing dist centers
xsamp = [rand(mm, nMC, nuts, ρx0; burn=2000) for mm in mmu]
ISint = [ISSamples(mm, x, GQint) for (mm,x) in zip(mmu, xsamp)]
∇Qu_is = [compute_gradQ(λ, q, IS)[1] for (λ,IS) in zip(λu,ISint)]

# compute the Monte Carlo estimate at biasing dist centers
∇Qa = [compute_gradQ(λ, q, MCint) for λ in λa]
# compute the IS estimate at biasing dist centers
xsamp = [rand(mm, nMC, nuts, ρx0; burn=2000) for mm in mma]
ISint = [ISSamples(mm, x, GQint) for (mm,x) in zip(mma, xsamp)]
∇Qa_is = [compute_gradQ(λ, q, IS)[1] for (λ,IS) in zip(λa,ISint)]

# compute covariance matrices
# Cref = JLD.load("data$modnum/DW1D_Ref.jld")["Cref"]
# Cmc = concat_dicts([compute_covmatrix(∇Qi, nsamp_arr) for ∇Qi in ∇Qmc]) # single high fidelity
# Ch = concat_dicts([compute_covmatrix(∇Qi, [10]) for ∇Qi in ∇Qmc]) # single high fidelity
# Cis = concat_dicts([compute_covmatrix(∇Qu[mmid[i]], ∇Qu_is[mmid[i]], ∇Qis[i], nsamp_arr) for i = 1:length(splits)])
# Cisa = concat_dicts([compute_covmatrix(∇Qa[mmid[i]], ∇Qa_is[mmid[i]], ∇Qisa[i], nsamp_arr) for i = 1:length(splits)])
# Cisaw = concat_dicts([compute_covmatrix(∇Qa[mmid[i]], ∇Qa_is[mmid[i]], ∇Qisaw[i], nsamp_arr) for i = 1:length(splits)])

# Cis = concat_dicts([compute_covmatrix(∇Qi, nsamp_arr) for ∇Qi in ∇Qis])
# Cisa = concat_dicts([compute_covmatrix(∇Qi, nsamp_arr) for ∇Qi in ∇Qisa])
# Cisaw = concat_dicts([compute_covmatrix(∇Qi, nsamp_arr) for ∇Qi in ∇Qisaw])

# M = 200
# nsamp_arr = [250, 500, 1000]
# Cmc = concat_dicts([compute_covmatrix(∇Qi, nsamp_arr) for ∇Qi in ∇Qmc]) # single high fidelity
# Cis = concat_dicts([compute_covmatrix(∇Qmc[i][1:M], ∇Qis[i][1:M], ∇Qis[i][M+1:end], nsamp_arr) for i = 1:length(splits)])
# Cisa = concat_dicts([compute_covmatrix(∇Qmc[i][1:M], ∇Qisa[i][1:M], ∇Qisa[i][M+1:end], nsamp_arr) for i = 1:length(splits)])
# Cisaw = concat_dicts([compute_covmatrix(∇Qmc[i][1:M], ∇Qisaw[i][1:M], ∇Qisaw[i][M+1:end], nsamp_arr) for i = 1:length(splits)])

# compute costs 
th = @elapsed grad_expectation(rand(ρθ), q, MCint)
tl = @elapsed grad_expectation(rand(ρθ), q, ISint[1])

budget = Vector(50:10:200)
M = Int.(floor.(budget./th))
Mh = Int.(ceil.(budget./(2*th)))
Ml = Int.(ceil.(budget./(2*tl)))
Cmc = concat_dicts([compute_covmatrix(∇Qi, M) for ∇Qi in ∇Qmc]) # single high fidelity
Cmc = Dict{Int64, Vector{Matrix}}( b => Cmc[k] for (b,k) in zip(budget, sort(collect(keys(Cmc))))) # single high fidelity

Cis = concat_dicts([compute_covmatrix(∇Qi, ∇Qj, Mh, Ml, budget) for (∇Qi,∇Qj) in zip(∇Qmc, ∇Qis)])
Cisa = concat_dicts([compute_covmatrix(∇Qi, ∇Qj, Mh, Ml, budget) for (∇Qi,∇Qj) in zip(∇Qmc, ∇Qisa)])
Cisaw = concat_dicts([compute_covmatrix(∇Qi, ∇Qj, Mh, Ml, budget) for (∇Qi,∇Qj) in zip(∇Qmc, ∇Qisaw)])



val_arr = [compute_val(Cref, C, 1) for C in [Cmc, Cisaw]]

# active subspaces
asref = compute_as(Cref, ρθ, 1)
asis_arr = [compute_as(∇Q, ρθ, 1) for ∇Q in ∇Q_arr]

f3 = plot_val_metric_mf("Forstner",
                val_arr,
                ["Monte Carlo", "MFMC"],
                "d(C, Ĉ)", 
                "Forstner distance from reference covariance matrix",
                logscl=true
)
f3

f4 = plot_val_metric_mf("WSD",
                val_arr,
                ["Monte Carlo", "MFMC"],
                "d(W₁,Ŵ₁)", 
                "Median weighted subspace distance error",
                logscl=true
    )

f4 = plot_val_metric("λ1_err",
    val_arr,
    lab_arr[2:end],
    "d(λ₁,λ̂₁)", 
    "Error in first eigenvalue",
    logscl=false
)




function plot_val_metric_mf(
    val_type::String,
    val_tup::Union{Tuple,Vector},
    lab_tup::Union{Tuple,Vector},
    ylab::String,
    ttl::String;
    logscl=true)

    colors = [:mediumpurple4, :goldenrod1, :seagreen, :skyblue1, :darkorange2, :firebrick2] 
    markers = [:utriangle, :hexagon, :circle, :diamond, :rect, :star]
    nsamp_arr = sort(collect(keys(val_tup[1][val_type])))
    N = length(nsamp_arr)

    with_theme(custom_theme) do
        fig = Figure(resolution=(900, 600))
        if logscl == true
            ax = Axis(fig[1,1], xlabel="budget", ylabel=ylab, # xticks=(nsamp_arr, [string(nsamp) for nsamp in nsamp_arr]),
            title=ttl, yscale=log10)   
        else
            ax = Axis(fig[1,1], xlabel="budget", ylabel=ylab, # xticks=(nsamp_arr, [string(nsamp) for nsamp in nsamp_arr]),
            title=ttl) #  yticks=0.0:0.005:0.025) 
        end
    
        for (i, val_i, lab_i) in zip(1:length(val_tup), val_tup, lab_tup)
            val_med = [median(val_i[val_type][k]) for k in nsamp_arr]
            val_hi = [percentile(val_i[val_type][k], 75) for k in nsamp_arr] # [maximum(val_i[val_type][k]) for k in nsamp_arr] # 
            val_lo = [percentile(val_i[val_type][k], 25) for k in nsamp_arr] # [minimum(val_i[val_type][k]) for k in nsamp_arr] #

            scatterlines!(ax, nsamp_arr, val_med, color=colors[i], marker=markers[i], markersize=15, label=lab_i)
            # band!(ax, nsamp_arr, val_lo, val_hi, color=(colors[i], 0.3))

        end
        fig[1, 2] = Legend(fig, ax, framevisible = false)
        # axislegend(ax, position=:ct)
        return fig
    end
end