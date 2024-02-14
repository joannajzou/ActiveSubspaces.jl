include("main.jl")
include("data_utils.jl")
include("mc_utils.jl")
include("plotting_utils.jl")

using DelimitedFiles

# load data ##################################################################################

# coeff sampling distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)
βdim = length(μ)

# coeff samples
βsamp = JLD.load("$(simdir)coeff_samples_1-1948.jld")["βsamp"]
nsamp = length(βsamp)

# skipped ids
idskip = JLD.load("$(simdir)coeff_skip_1-1948.jld")["id_skip"]
idkeep = symdiff(1:1450, idskip) 
βsamp_2 = βsamp[idkeep]

# load MC estimates
niter = 8
∇Qmc = [JLD.load("$(simdir)coeff_$j/gradQ_meanenergy.jld")["∇Q"] for j in idkeep]
# ∇Qis_e = [JLD.load("$(simdir)/gradQ_IS_EW_nc=50/gradQ_ISM_$(iter).jld")["∇Q"][1:1445] for iter in 1:niter]
∇Qis_et = [JLD.load("$(simdir)/gradQ_IS_EW_nc=150/gradQ_ISM_Temp_$(iter).jld")["∇Q"][1:1445] for iter in 1:niter]

# compute costs
th = 150
tl = 0.8
Mh = [75, 100, 150]
Mh = [50, 100, 150]
Ml = 1445 .- Mh
budget = Int.(Ml.*tl + Mh.*th)
M = Int.(floor.(budget ./ th))

Cref = compute_covmatrix(∇Qmc)
Cmc = [[compute_covmatrix(∇Qmc[Mi*(k-1)+1:Mi*k]) for k = 1:niter] for Mi in M] # single high fidelity
Cise = [[compute_covmatrix(∇Qmc[1:Mhi], ∇Q[1:Mhi], ∇Q[(Mhi+1):(Mhi+Mli)]) for ∇Q in ∇Qis_et] for (Mhi, Mli) in zip(Mh, Ml)]
Cmc = Dict{Int64, Vector{Matrix}}( b => Cmc[k] for (b,k) in zip(Mh, 1:length(Mh)))
Cise = Dict{Int64, Vector{Matrix}}( b => Cise[k] for (b,k) in zip(Mh, 1:length(Mh)))

val_arr = [compute_val(Cref, C, 4) for C in [Cmc, Cise]]

f3 = plot_val_boxplot("Forstner",
                val_arr,
                ["Monte Carlo", "MFMC"],
                "d(C, Ĉ)", 
                "Forstner distance from reference covariance matrix",
                logscl=true
)
f3 = plot_val_boxplot("WSD",
                val_arr,
                ["Monte Carlo", "MFMC"],
                "WSD(C, Ĉ)", 
                "Weighted subspace distance between active subspaces",
                logscl=true
)


function plot_val_boxplot(
    val_type::String,
    val_tup::Union{Tuple,Vector},
    lab_tup::Union{Tuple,Vector},
    ylab::String,
    ttl::String;
    logscl=true)

    nsamp_arr = sort(collect(keys(val_tup[1][val_type])))
    niter = length(val_tup[1][val_type][nsamp_arr[1]])
    Ntype = length(val_tup)
    Nbudget = length(nsamp_arr)

    idx = reduce(vcat, [reduce(vcat, [k*ones(niter) for k = 1:Nbudget]) for i = 1:2])
    plotval = reduce(vcat, [reduce(vcat, [val_tup[k][val_type][nsamp] for nsamp in nsamp_arr]) for k = 1:Ntype])
    dodge = Int.(reduce(vcat, [k*ones(niter*Nbudget) for k = 1:2]))

    with_theme(custom_theme) do
        fig = Figure(resolution=(850, 600))
        if logscl == true
            ax = Axis(fig[1,1], ylabel=ylab, xlabel="budget", xticks=(1:Nbudget, [string(nsamp) for nsamp in nsamp_arr]),
            title=ttl, yscale=log10)   
        else
            ax = Axis(fig[1,1], ylabel=ylab, xlabel="budget", xticks=(1:Nbudget, [string(nsamp) for nsamp in nsamp_arr]),
            title=ttl) 
        end
        boxplot!(ax, idx, plotval,
            dodge = dodge,
            width = 0.5,
            gap = 0.05,
            whiskerwidth=0.5,
            dodge_gap = 0.1,
            color=map(d->d==1 ? :skyblue3 : :orange, dodge))

        # create Legend
        elem1 = PolyElement(color = :skyblue3)
        elem2 = PolyElement(color = :orange)
        Legend(fig[1,2], [elem1, elem2], lab_tup)

        return fig
    end
end