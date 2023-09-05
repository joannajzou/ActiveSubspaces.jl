include("00_spec_model.jl")
include("plotting_utils.jl")


# compute posterior on ACE parameters #################################################################
# load dataset
confs = load_data("data/a-Hfo2-300K-NVT.extxyz", ExtXYZ(u"eV", u"Å"))


# Update dataset by adding energy (local) descriptors
println("Computing local descriptors")
e_descr = compute_local_descriptors(confs, ace)
JLD.save("data/global_energy_descriptors_$(nbody)body_$(deg)deg.jld", "e_descr", e_descr)
ds = DataSet(confs .+ e_descr)
ndata = length(ds)

# Split dataset
ind = rand(1:ndata, 4000)
ds_train, ds_test = ds[ind[1:1000]], ds[ind[1001:end]]

# learn using DPP samples
lb = LBasisPotential(ace)
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 800) # 3000
dpp_inds = get_random_subset(dpp)
lb, Σ = learn!(lb, ds_train[dpp_inds]; α = 1e-8)

e_descr_train = sum.(get_values.(get_local_descriptors.(ds_train[dpp_inds])))
e_descr_test = sum.(get_values.(get_local_descriptors.(ds_test)))

JLD.save("$(simdir)fitting_params.jld",
    "dpp_inds", dpp_inds,
    "β", lb.β,
    "Σ", Σ)

# evaluate fit #######################################################################################

# load energy/force data
e_train = get_all_energies(ds_train[dpp_inds])
e_test = get_all_energies(ds_test)
e_train_pred = get_all_energies(ds_train[dpp_inds], lb)
e_test_pred = get_all_energies(ds_test, lb)
e_train_err = (e_train - e_train_pred) ./ e_train
e_test_err = (e_test - e_test_pred) ./ e_test

# Compute error metrics
metrics = get_metrics( e_train_pred, e_train,
                       e_test_pred, e_test)

# plot energy and error 
plot_energy(e_train_pred, e_train)
plot_energy(e_test_pred, e_test)

plot_error(e_train_err)
plot_error(e_test_err)

# define sampling density on coefficients
d = length(lb.β)
Σβ = 2e1*Σ + 1e-12*I(d)
πβ = MvNormal(lb.β, Σβ)
JLD.save("$(simdir)coeff_distribution.jld",
    "μ", lb.β,
    "Σ", Σβ
)
