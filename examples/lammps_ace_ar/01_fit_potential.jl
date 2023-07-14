include("00_spec_model.jl")


# compute posterior on ACE parameters #################################################################
# load dataset
confs, thermo = load_data("data/lj-ar.yaml", YAML(:Ar, u"eV", u"Å"))


# Update dataset by adding energy (local) descriptors
println("Computing local descriptors")
e_descr = compute_local_descriptors(confs, ace)
ds = DataSet(confs .+ e_descr)

# Split dataset
ds_train, ds_test = ds[2001:end], ds[8001:end]

# learn using DPP samples
lb = LBasisPotential(ace)
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 30) # 3000
dpp_inds = get_random_subset(dpp)
lb, Σ = learn!(lb, ds_train[dpp_inds]; α = 1e-8)

e_descr_train = sum.(get_values.(get_local_descriptors.(ds_train[dpp_inds])))
e_descr_test = sum.(get_values.(get_local_descriptors.(ds_test)))



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
Σβ = 100 *Σ + 1e-10*I(d)
πβ = MvNormal(lb.β, Σβ)
JLD.save("$(simdir)coeff_distribution.jld",
    "μ", lb.β,
    "Σ", Σβ
)

