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


# run MC trials #################################################################################
# Monte Carlo
∇Qmc, idskip = compute_gradQ(βsamp, q, simdir; gradh=∇h) 
JLD.save("$(simdir)coeff_skip.jld", "id_skip", idskip)
idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]

# remove skipped indices
idkeep = symdiff(1:nsamp, idskip)
βsamp_2 = βsamp[idkeep]

# importance sampling
πg = Gibbs(πgibbs, θ=πβ.μ)
# compute descriptors
ds = load_data("$(simdir)coeff_nom/data.xyz", ExtXYZ(u"eV", u"Å"))
e_descr = compute_local_descriptors(ds, ace)
Φsamp = sum.(get_values.(e_descr))
# JLD.save("$(simdir)coeff_nom/global_energy_descriptors_nontemp.jld", "Φsamp", Φsamp)
# or load
# Φsamp = JLD.load("$(simdir)coeff_nom/global_energy_descriptors.jld")["Φsamp"]

# define integrator
ISint = ISSamples(πg, Φsamp)

# compute
∇Qis, metrics = compute_gradQ(βsamp_2, q, ISint; gradh=∇h)

JLD.save("$(simdir)gradQ_meanenergy_IS_all_84.jld", "∇Q", ∇Qis)
JLD.save("$(simdir)metrics_84.jld", "metrics", metrics)


