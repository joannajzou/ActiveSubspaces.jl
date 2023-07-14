include("00_spec_model.jl")
include("qoi_meanenergy.jl")
include("mc_utils.jl")
include("plotting_utils.jl")


# load data #####################################################################################
# coeff distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)
βdim = length(μ)
# coeff samples
βsamp = JLD.load("$(simdir)coeff_samples.jld")["βsamp"]
nsamp = length(βsamp)


# Monte Carlo - standard ########################################################################
# ∇Qmc, idskip = compute_gradQ(βsamp, q, simdir; gradh=∇h) 
# JLD.save("$(simdir)coeff_skip.jld", "id_skip", idskip)
idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]

# remove skipped indices
idkeep = symdiff(1:nsamp, idskip)
βsamp_2 = βsamp[idkeep]


# importance sampling - single distribution #####################################################
# πg = Gibbs(πgibbs, θ=πβ.μ)
# # compute descriptors 
# ds = load_data("$(simdir)coeff_nom/data.xyz", ExtXYZ(u"eV", u"Å"))
# e_descr = compute_local_descriptors(ds, ace)
# Φsamp = sum.(get_values.(e_descr))
# JLD.save("$(simdir)coeff_nom/global_energy_descriptors_nontemp.jld", "Φsamp", Φsamp)
# # or load
# # Φsamp = JLD.load("$(simdir)coeff_nom/global_energy_descriptors.jld")["Φsamp"]

# # define integrator
# ISint = ISSamples(πg, Φsamp)

# # compute
# ∇Qis, metrics = compute_gradQ(βsamp_2, q, ISint; gradh=∇h)

# JLD.save("$(simdir)gradQ_meanenergy_IS_all_84.jld", "∇Q", ∇Qis)
# JLD.save("$(simdir)metrics_84.jld", "metrics", metrics)


# importance sampling - mixture distribution, equal weights #############################################
# randomly sample component PDFs
nprop = 100
nMC = 10000
centers = reduce(vcat, ([πβ.μ], βsamp_2[2:nprop]))
# sample from component PDFs
# for i = 1:nprop
#     if i == 1; num = "nom"; else; num = idkeep[i]; end
#     ds = load_data("$(simdir)coeff_$num/data.xyz", ExtXYZ(u"eV", u"Å"))
#     e_descr = compute_local_descriptors(ds, ace)
#     Φsamp = sum.(get_values.(e_descr))
#     JLD.save("$(simdir)coeff_$num/global_energy_descriptors.jld", "Φsamp", Φsamp)

#     # check quality of samples
#     plot_mcmc_trace(Φsamp)
#     plot_mcmc_autocorr(Φsamp)
# end

# load from file
samp_all = Vector(undef, nprop)
for i = 1:nprop
    if i == 1; num = "nom"; else; num = idkeep[i]; end
    Φsamp = JLD.load("$(simdir)coeff_$num/global_energy_descriptors.jld")["Φsamp"][200:end]
    samp_all[i] = Φsamp
end

# define mixture model
πg = [Gibbs(πgibbs, θ=c) for c in centers]
mm = MixtureModel(πg)

# # draw fixed set of samples
# Φsamp_mix, catratio = rand(mm, nMC, samp_all)
# ISint = ISSamples(mm, Φsamp_mix)

# # compute
# ∇Qis, metrics = compute_gradQ(βsamp_2[1:10], q, ISint; gradh=∇h)

# JLD.save("$(simdir)gradQ_meanenergy_IS_all_mix_equalwts.jld", "∇Q", ∇Qis)
# JLD.save("$(simdir)metrics_mix_equalwts.jld", "metrics", metrics)


# importance sampling - mixture distribution, var. weights #############################################
knl = RBF(Euclidean(βdim); ℓ=1e-7)
ISint = ISMixSamples(mm, nMC, knl, samp_all)

# compute
∇Qis, metrics = compute_gradQ(βsamp_2, q, ISint; gradh=∇h)

JLD.save("$(simdir)gradQ_meanenergy_IS_all_mix_varwts.jld", "∇Q", ∇Qis)
JLD.save("$(simdir)metrics_mix_varwts.jld", "metrics", metrics)