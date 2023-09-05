include("00_spec_model.jl")
include("qoi_meanenergy.jl")
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
βsamp = JLD.load("$(simdir)coeff_samples_aug.jld")["βsamp"]
nsamp = length(βsamp)

# skipped ids
idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]
idkeep = symdiff(1:1450, idskip) 
βsamp_2 = βsamp[idkeep]

# load centers
cents_iso = JLD.load("$(simdir)/gradQ_IS_EW_nc=150/gradQ_ISM_$(iternum).jld")["centers"]
cents_cov = JLD.load("$(simdir)/gradQ_IS_EW_nc=150/gradQ_ISM_$(iternum+4).jld")["centers"]

# load MC estimates
∇Qmc = [JLD.load("$(simdir)coeff_$j/gradQ_meanenergy.jld")["∇Q"] for j in idkeep]
# JLD.save("$(simdir)gradQ_MC.jld", "∇Q", ∇Qmc)

nprop_arr = [150]
niter = 3
iternum = 1 

∇Qis_iso = Dict{Int64, Vector{Vector}}()
∇Qis_isot = Dict{Int64, Vector{Vector}}()
∇Qis_cov = Dict{Int64, Vector{Vector}}()
∇Qis_covt = Dict{Int64, Vector{Vector}}()

met_iso = Dict{Int64, Dict}()
met_isot = Dict{Int64, Dict}()
met_cov = Dict{Int64, Dict}()
met_covt = Dict{Int64, Dict}()

asis_iso = Dict{Int64, Vector{Subspace}}()
asis_isot = Dict{Int64, Vector{Subspace}}()
asis_cov = Dict{Int64, Vector{Subspace}}()
asis_covt = Dict{Int64, Vector{Subspace}}()

# load IS estimates
for nprop in nprop_arr
    # gradient of Q
    ∇Qis_iso[nprop] = [JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_$(iter).jld")["∇Q"][1:1445] for iter = 1:niter]
    ∇Qis_isot[nprop] = [JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_Temp_$(iter).jld")["∇Q"][1:1445] for iter = 1:niter]
    ∇Qis_cov[nprop] = [JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_$(iter+4).jld")["∇Q"][1:1445] for iter = 1:niter]
    ∇Qis_covt[nprop] = [JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_Temp_$(iter+4).jld")["∇Q"][1:1445] for iter = 1:niter]

    # IS metrics
    met_iso[nprop] = JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_$(iternum).jld")["metrics"]
    met_isot[nprop] = JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_Temp_$(iternum).jld")["metrics"]
    met_cov[nprop] = JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_$(iternum+4).jld")["metrics"]
    met_covt[nprop] = JLD.load("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_Temp_$(iternum+4).jld")["metrics"]

    asis_iso[nprop] = [compute_as(∇Qis, πβ, βdim) for ∇Qis in ∇Qis_iso[nprop]]
    asis_isot[nprop] = [compute_as(∇Qis, πβ, βdim) for ∇Qis in ∇Qis_isot[nprop]]
    asis_cov[nprop] = [compute_as(∇Qis, πβ, βdim) for ∇Qis in ∇Qis_cov[nprop]]
    asis_covt[nprop] = [compute_as(∇Qis, πβ, βdim) for ∇Qis in ∇Qis_covt[nprop]]
end


# assemble arrays of inputs ############################################################################

∇Qis_arr = [[∇Qis_iso[nprop][iternum], ∇Qis_isot[nprop][iternum],
             ∇Qis_cov[nprop][iternum], ∇Qis_covt[nprop][iternum]] for nprop in nprop_arr]
∇Qis_arr = reduce(vcat, ∇Qis_arr)

met_arr = [[met_iso[nprop], met_isot[nprop],
            met_cov[nprop], met_covt[nprop]] for nprop in nprop_arr]
met_arr = reduce(vcat, met_arr)
            
asis_arr = [[asis_iso[nprop], asis_isot[nprop],
             asis_cov[nprop], asis_covt[nprop]] for nprop in nprop_arr]
asis_arr = reduce(vcat, asis_arr)
asmc = compute_as(∇Qmc, πβ, βdim)



# plot settings ############################################################################
col_arr = [:dodgerblue4, :dodgerblue4, 
           :orangered3, :orangered3]

ls_arr = reduce(vcat, [[nothing, :dash] for i = 1:2])
lab_arr = ["iso. kernel", "iso. kernel (temp.)", "cov. kernel", "cov. kernel (temp.)"]

mark_arr = [:hexagon, :hexagon,
            :circle, :circle,
            :diamond, :diamond,
            :rect, :rect,
            :star4, :star4,
            :utriangle, :utriangle]                      

# compare gradient calculations ############################################################
f = plot_error(∇Qmc[1:1445], ∇Qis_arr, col_arr, lab_arr,
               "Error (log Euclid. dist.)", "Error in estimated ∇Q")


# compare subspaces ########################################################################
f1 = plot_eigenspectrum(asmc, asis_arr, lab_arr, col_arr, markers=mark_arr, ls=ls_arr)

f2 = plot_wsd(asmc, asis_arr, lab_arr, col_arr, markers=mark_arr, ls=ls_arr)

f3 = plot_cossim(asmc, asis_arr, lab_arr, col_arr, markers=mark_arr, ls=ls_arr)

f4 = plot_cossim_mat(asmc, asis_arr, lab_arr)

# check convergence of eigenspectrum #######################################################
n_arr = [30, 60, 120, 240, 480, 960, 1920]
n_arr = [250, 500, 750, 1000, 1250, 1500, 1750, 1943]
λ_arr = Vector{Vector{Float64}}(undef, length(n_arr))
for i = 1:length(n_arr)
    _, λ_arr[i], _ = compute_eigenbasis(∇Qmc[1:n_arr[i]])
end
f = plot_eigenspectrum(λ_arr, labs=[string(n) for n in n_arr])



# check IS diagnostics #####################################################################
# check for nan values
# id1 = findall(x -> isnan(x), metrics_v["wvar"])
# id2 = findall(x -> isnan(x), metrics_v["wESS"])
# id = union(id1, id2)
# metrics_2 = Dict{String, Vector}()
# for k in collect(keys(metrics))
#     metrics_2[k] = metrics[k][Not(id)]
# end

# transform samples into 2D
f4 = plot_IS_diag_2D_AS("wvar",
                        met_arr,
                        lab_arr,
                        asmc.C,
                        πβ,
                        βsamp_2,
                        cents_cov,
                        "log Var(w)";
                        logscl=true)

f5 = plot_IS_diag_2D_AS("wESS",
                        met_arr,
                        lab_arr,
                        asmc.C,
                        πβ,
                        βsamp_2,
                        cents_cov,
                        "ESS(w)")

# plot "cdf"
f6 = plot_IS_cdf("wvar",
                met_arr,
                lab_arr,
                "P[ Var(w) > t ]",
                "Variance of IS weights",
                col_arr,
                limtype=[2^6, 2^7, 2^8, 2^9], # [1e1, 1e2, 1e2.5],
                xticklab=["2^6", "2^7", "2^8", "2^9"], # ["1e1.5", "1e2", "1e2.5"],
                rev = false,
                ls = ls_arr
)

f7 = plot_IS_cdf("wESS",
                met_arr,
                lab_arr,
                "P[ ESS(w)/n > t ]",
                "ESS of IS weights",
                col_arr,
                limtype = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                xticklab=[string(i) for i in 0:0.1:0.7], # [string(i) for i in 0:0.2:1],
                rev = false,
                logscl = false,
                ls = ls_arr
)



# plot samples with high/low diagnostic values #####################################################################

# Define 2D points
r = Vector(1.0:0.01:5)
box = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
system = [FlexibleSystem([AtomsBase.Atom(:Ar, [0.0, 0.0, 0.0] * u"Å"), AtomsBase.Atom(:Ar, [ri, 0.0, 0.0] * u"Å")],
                        box * u"Å", bcs) for ri in r]
B = [sum(compute_local_descriptors(sys, ace)) for sys in system]

# plot random samples from parameter sampling density
f8 = plot_pairwise_potential(βsamp_2[1:1000], r, B)

# plot samples with high/low IS weight variance
ny = 50
sortid = sortperm(met_cov[150]["wvar"], rev=true) # descending order
hi_err_id = sortid[1:ny]
lo_err_id = sortid[end-ny:end-1]

f9 = plot_pairwise_potential([βsamp_2[hi_err_id], βsamp_2[lo_err_id]],
                            r, B, [:red, :Green], ["High Var(w)", "Low Var(w)"])

# plot samples with high/low ESS
sortid = sortperm(met_cov[150]["wESS"], rev=true) # descending order
hi_err_id = sortid[1:ny]
lo_err_id = sortid[end-ny:end-1]

f10 = plot_pairwise_potential([βsamp_2[lo_err_id], βsamp_2[hi_err_id]],
                            r, B, [:red, :Green], ["Low ESS(w)", "High ESS(w)"])



# plot samples which didn't run #####################################################################
βskip = βsamp[idskip]
f11 = plot_pairwise_potential(βskip, r, B)


# plot samples from the active subspace #####################################################################
ysamp, θysamp = sample_as(50, asmc)
f12 = plot_pairwise_potential(θysamp, r, B)


# plot mode shapes of pairwise potential ############################################

ny = 50 # number of samples
# modes of as
figsa = plot_as_modes(asmc, πβ, ny, Vector(r), B; scl=1e2)
fa = plot_eigenvectors(asmc.W1)

# pca components
_, _, Wpca = compute_eigenbasis(Matrix(πβ.Σ))
figsb = plot_as_modes(Wpca, πβ, ny, Vector(r), B; scl=1e2)
fb = plot_eigenvectors(Wpca)

# parameter study
figsc = plot_parameter_study(πβ, ny, Vector(r), B; scl=1e2)
fc = plot_eigenvectors(Matrix(I(βdim)))



## evaluate quality of reference: ESS and SE ##################################
# se_bm = Vector{Float64}(undef, length(idkeep))
# ess = Vector{Float64}(undef, length(idkeep))

# for i = 1:length(idkeep)
#     println("$i")
#     num = idkeep[i]
#     βi = βsamp_2[i]
#     Φsamp = JLD.load("$(biasdir)coeff_$num/energy_descriptors.jld")["Bsamp"]

#     # for biasing dist
#     πgibbs2 = Gibbs(πgibbs, β=0.48)
#     q = GibbsQoI(h=h, p=πgibbs2)

#     qoim = GibbsQoI(h=x -> q.h(x, βi), p=Gibbs(q.p, θ=βi))
#     se_bm[i] = MCSEbm(qoim, Φsamp)
#     ess[i] = EffSampleSize(qoim, Φsamp)  
# end
# JLD.save("$(biasdir)mcmc_diagnostics.jld",
#         "se", se_bm,
#         "ess", ess)

# or load
Temp_b = 150.0        # higher temp for biasing dist.    
biasdir = "$(simdir)Temp_$(Temp_b)/"
dict = JLD.load("$(simdir)mcmc_diagnostics.jld")

fig = Figure(resolution=(800,400))
ax1 = Axis(fig[1,1], xlabel="MCSE", ylabel="count", title="Monte Carlo standard error")
ax2 = Axis(fig[1,2], xlabel="ESS", ylabel="count", title="ESS")
hist!(ax1, dict["se"])
hist!(ax2, dict["ess"])
fig


## compute QoI with active subspace ########################################### 

# define IS integrator
∇Qmc = [JLD.load("$(simdir)coeff_$j/gradQ_meanenergy.jld")["∇Q"] for j in idkeep]

nsamp = 10000
πg = Gibbs(πgibbs, θ=πβ.μ)
Φsamp = JLD.load("$(simdir)coeff_mean/energy_descriptors.jld")["Bsamp"]
ISint = ISSamples(πg, Φsamp)

# original sampling density
θsamp = [rand(πβ) for i = 1:nsamp]

Q0 = zeros(nsamp)
@elapsed begin
for k = 1:nsamp
    Q0[k], _, _ = expectation(θsamp[k], q, ISint)
end
end

# from active subspace 
as3 = compute_as(∇Qmc, πβ, 3)
Q3 = zeros(nsamp)
_, θysamp = sample_as(nsamp, as3)
for k = 1:nsamp
    Q3[k], _, _ = expectation(θysamp[k], q, ISint)
end

# from as modes
Qas = Vector{Vector{Float64}}(undef, βdim)

for i = 1:βdim
    println("i")
    θyi = draw_samples_as_mode(asmc.W1, i, πβ, nsamp)
    Qas[i] = zeros(nsamp)
    for k = 1:nsamp
        Qas[i][k], _, _ = expectation(θyi[k], q, ISint)
    end
end

Q4 = JLD.load("$(simdir)QoI_as_modes.jld")["Q4"]
Qas = JLD.load("$(simdir)QoI_as_modes.jld")["Qas"]
JLD.save("$(simdir)QoI_as_modes_run2.jld", "Q0", Q0, "Q3", Q3, "Qas", Qas)
bins = LinRange(minimum(Q0), maximum(Q0), 50)
# plot hist
with_theme(custom_theme) do
    fig = Figure(resolution=(800, 550))
    ax = Axis(fig[1,1], xlabel="Q", ylabel="count", title="Distribution of Q")
    hist!(ax, Q0, bins=bins, color=(:blue, 0.2), label="original sampling density")
    hist!(ax, Q3, bins=bins, color=(:red, 0.2), label="active subspace (3)")
    # hist!(ax, Q4, bins=bins, color=(:Green, 0.2), label="active subspace (4)")
    # hist!(ax, Qas[end], bins=bins, color=(:yellow, 0.2), label="inactive subspace")
    axislegend(ax, position=:rt)
    fig
end
# with_theme(custom_theme) do
#     fig = Figure(resolution=(800, 550))
#     # ax = Axis(fig[1,1], ylabel="ESS/n", xlabel="descriptor dimension", xticks=1:βdim, title="Histogram of ESS/n")
#     # [hist!(ax, ESS[i,:], scale_to=-0.75, color=(:red, 0.5), offset=i, bins=10, direction=:x) for i = 1:βdim]
#     ax = Axis(fig[1,1], xlabel="log(Q)", ylabel="count", title="Distribution of Q")
#     hist!(ax, log.(abs.(Q0)), bins=LinRange(log(0.0001), log(3000), 25), color=(:blue, 0.2), label="original sampling density")
#     hist!(ax, log.(abs.(Qy1)), bins=LinRange(log(0.0001), log(3000), 25), color=(:red, 0.2), label="active subspace")
#     hist!(ax, log.(abs.(Qy8)), bins=LinRange(log(0.0001), log(3000), 25), color=(:green, 0.2), label="inactive subspace")
#     axislegend(ax, position=:rt)
#     fig
# end

# boxplot
index = reduce(vcat, [i*ones(nsamp) for i = 1:βdim])
index = reduce(vcat, [zeros(nsamp), index])
Qall = abs.(reduce(vcat, Qas))
Qall = abs.(reduce(vcat, [Q0, Qall]))

fig = Figure(resolution=(800, 500))
# ax = Axis(fig[1,1], ylabel="ESS/n", xlabel="descriptor dimension", xticks=1:βdim, title="Histogram of ESS/n")
# [hist!(ax, ESS[i,:], scale_to=-0.75, color=(:red, 0.5), offset=i, bins=10, direction=:x) for i = 1:βdim]
ax = Axis(fig[1,1], ylabel="Q", yscale=log10, title="Distribution of Q")
boxplot!(ax, index, Qall .+ 1e-10)
fig




function draw_samples_as_mode(W::Matrix, ind::Int64, ρθ::Distribution, ny::Int64)
    dims = Vector(1:size(W,1))
    W1 = W[:,ind:ind] # type as matrix
    W2 = W[:, dims[Not(ind)]]

    # compute sampling density
    π_y = compute_marginal(W1, ρθ)
    π_z = compute_marginal(W2, ρθ)

    # draw samples
    ysamp = [rand(π_y) for i = 1:ny]
    θy = [W1*y + W2*π_z.μ for y in ysamp]

    return θy
end





# plot distribution of mixture weights #############################################

figs = Vector{Figure}(undef, length(met_cov_arr))

for (i, met) in enumerate(met_cov_arr)

    mixwts = reduce(hcat, met["mixwts"])
    ncent, nsamp = size(mixwts)
    mxwts = [abs.(mixwts[i,:]) for i = 1:ncent]

    figs[i] = plot_staggered_hist(mxwts,
                        [string(i) for i = 1:ncent],
                        [:blue for i = 1:ncent],
                        "center num.",
                        "kernel mixture weights",
                        logscl=true
                        )
end



# plot MCMC diagnostics ############################################################
ds = load_data("$(simdir)coeff_nom/data.xyz", ExtXYZ(u"eV", u"Å"))
e_descr = compute_local_descriptors(ds, ace)
Φsamp = sum.(get_values.(e_descr))
Φ = reduce(hcat, Φsamp)
# [ustrip(get_positions(d)) for d in ds]
Mcmc_traceplot(Φ)
Mcmc_autocorrplot(Φ)

ESS = compute_mcmc_ess(βsamp_2[301:400], idkeep[301:400], simdir)
JLD.save("ESS_301-400.jld", "ESS", ESS)

ess1 = JLD.load("ESS_1-100.jld")["ESS"]
ess2 = JLD.load("ESS_101-200.jld")["ESS"]
ess3 = JLD.load("ESS_201-300.jld")["ESS"]
ess4 = JLD.load("ESS_301-400.jld")["ESS"]

ESS = reduce(hcat, [ess1, ess2, ess3, ess4])

# plot ESS hist
fig = Figure(resolution=(800, 500))
# ax = Axis(fig[1,1], ylabel="ESS/n", xlabel="descriptor dimension", xticks=1:βdim, title="Histogram of ESS/n")
# [hist!(ax, ESS[i,:], scale_to=-0.75, color=(:red, 0.5), offset=i, bins=10, direction=:x) for i = 1:βdim]
ax = Axis(fig[1,1], xlabel="ESS/n", ylabel="count", title="Histogram of mean(ESS/n)")
hist!(ax, mean(abs.(ESS), dims=1)[:], bins=20)
fig



# plot projection onto active subspace ##################################################

fig = Figure(resolution=(600,480))
ax = Axis(fig[1,1], ylabel="||Pr * ei ||₂", xlabel="ei", xticks=1:βdim, limits=(0, 9, -0.05, 1.05))
sc = Vector(undef, βdim)
for tr = 1:βdim
    Wtrunc = asis_covt[150][1].W1[:, 1:tr] # asmc.W1[:, 1:tr] # 
    Proj = Wtrunc * Wtrunc'

    infl = zeros(βdim)
    for i = 1:βdim
        ei = zeros(βdim); ei[i] = 1
        infl[i] = norm(Proj * ei)
    end

    sc[tr] = scatterlines!(ax, 1:βdim, infl, color=RGB(0, tr/βdim, 0))
end
Legend(fig[1,2], sc, ["nbasis = $i" for i in 1:βdim])
# axislegend(ax, position=:rb)
fig









########## ##### ##### ##### ##### ##### ##### ##### 



function compute_mcmc_ess(βsamp::Vector, ids::Vector, simdir::String)
    nβ = length(βsamp[1])
    ESS = Matrix{Float64}(undef, (nβ, length(βsamp)))

    for (i, id) in zip(1:length(βsamp), ids)
        println("sample $id")
        ds = load_data("$(simdir)coeff_$id/data.xyz", ExtXYZ(u"eV", u"Å"))
        e_descr = compute_local_descriptors(ds, ace)
        Φsamp = sum.(get_values.(e_descr))
        Φ = reduce(hcat, Φsamp)
        n = length(Φsamp)
        autocorr = [1 + 2*sum(autocor(Φ[j,:], 0:1000)) for j = 1:nβ]
        ESS[:,i] = 1 ./ autocorr
    end
    return minimum(ESS)
end


writedlm( "$(simdir)Cmatrix.csv",  cmc, ',')

