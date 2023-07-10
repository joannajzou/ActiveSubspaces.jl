using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using ACE1, ACE1pack, JuLIP
using Distributions
using JLD

include("molecular_plot_utils.jl")

import PotentialLearning: get_metrics
import OrderedCollections: OrderedDict
function get_metrics(
    e_train_pred,
    e_train,
    e_test_pred,
    e_test
)
    e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
    e_test_mae, e_test_rmse, e_test_rsq = calc_metrics(e_test_pred, e_test)

    metrics = OrderedDict(
        "e_train_mae" => e_train_mae,
        "e_train_rmse" => e_train_rmse,
        "e_train_rsq" => e_train_rsq,
        "e_test_mae" => e_test_mae,
        "e_test_rmse" => e_test_rmse,
        "e_test_rsq" => e_test_rsq,
    )
    return metrics
end



# compute posterior on ACE parameters #################################################################
# load dataset
# Load dataset
confs, thermo = load_data("data/liquify_sodium.yaml", YAML(:Na, u"eV", u"Å"))
confs, thermo = confs[220:end], thermo[220:end]

# Split dataset
conf_train, conf_test = confs[1:1000], confs[1001:end]

# Define ACE
ace = ACE(species = [:Na],         # species
          body_order = 4,          # 4-body
          polynomial_degree = 8,   # 8 degree polynomials
          wL = 1.0,                # Defaults, See ACE.jl documentation 
          csp = 1.0,               # Defaults, See ACE.jl documentation 
          r0 = 1.0,                # minimum distance between atoms
          rcutoff = 5.0)           # cutoff radius 

# Update training dataset by adding energy (local) descriptors
println("Computing local descriptors of training dataset")
e_descr_train = compute_local_descriptors(conf_train, ace)
#e_descr_train = JLD.load("data/sodium_empirical_full.jld", "descriptors")
ds_train = DataSet(conf_train .+ e_descr_train)

# Learn, using DPP
lb = LBasisPotential(ace)
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 200)
dpp_inds = get_random_subset(dpp)
lb, Σ = learn!(lb, ds_train[dpp_inds]; α = 1e-6)



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

# define posterior distribution on coefficients
d = length(lb.β)
Σβ = 100 * Hermitian(Σ) + 1e-12*I(d)
πβ = MvNormal(lb.β, Σβ)
JLD.save("coeff_posterior_2body.jld", "π", πβ)



