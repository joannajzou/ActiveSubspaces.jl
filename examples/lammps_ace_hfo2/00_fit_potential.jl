# define ACE basis --------------------------------------------------------------------------------------
nbody = 5
deg = 5

ace = ACE(species = [:Hf, :O],          # species
          body_order = nbody,           # n-body
          polynomial_degree = deg,      # degree of polynomials
          wL = 1.0,                     # Defaults, See ACE.jl documentation 
          csp = 1.0,                    # Defaults, See ACE.jl documentation 
          r0 = 1.0,                     # minimum distance between atoms
          rcutoff = 5.0)                # cutoff radius 


# compute posterior on ACE parameters -------------------------------------------------------------------
# load dataset
confs = load_data("data/a-Hfo2-300K-NVT.extxyz", ExtXYZ(u"eV", u"Å"))


# Update dataset by adding energy (local) descriptors
println("Computing local descriptors")
e_descr = compute_local_descriptors(confs, ace)
f_descr = compute_force_descriptors(confs, ace)

JLD.save("data/global_energy_descriptors_$(nbody)body_$(deg)deg.jld", 
    "e_descr", e_descr,
)
JLD.save("data/global_force_descriptors_$(nbody)body_$(deg)deg.jld", 
    "f_descr", f_descr,
)

# or load
e_descr = JLD.load("data/global_energy_descriptors_$(nbody)body_$(deg)deg.jld")["e_descr"]
f_descr = JLD.load("data/global_force_descriptors_$(nbody)body_$(deg)deg.jld")["f_descr"]

ds = DataSet(confs .+ e_descr + f_descr)
ndata = length(ds)


# learn using DPP samples
lb = LBasisPotentialExt(ace)
dpp = kDPP(ds, GlobalMean(), DotProduct(); batch_size = 1000) # 3000
dpp_inds = get_random_subset(dpp)
test_inds = symdiff(dpp_inds, 1:ndata)
ds_train = ds[dpp_inds]; ds_test = ds[test_inds]
lp = learn!(lb, ds_train, 1e-8)


JLD.save("data/fitted_params_$(nbody)body_$(deg)deg.jld",
    "dpp_inds", dpp_inds,
    "β", lp.β,
    "Σ", lp.Σ)


# evaluate fit ----------------------------------------------------------------------------------------

# load energy/force data
e_train = get_all_energies(ds_train)
e_test = get_all_energies(ds_test)
f_train = get_all_forces_mag(ds_train)
f_test = get_all_forces_mag(ds_test)

e_train_pred = get_all_energies(ds_train, lb)
e_test_pred = get_all_energies(ds_test, lb)
f_train_pred = get_all_forces_mag(ds_train, lb)
f_test_pred = get_all_forces_mag(ds_test, lb)

e_train_err = (e_train - e_train_pred) ./ e_train
e_test_err = (e_test - e_test_pred) ./ e_test
f_train_err = (f_train - f_train_pred) ./ f_train
f_test_err = (f_test - f_test_pred) ./ f_test

# Compute error metrics
metrics_e = get_metrics( e_train_pred, e_train,
                       e_test_pred, e_test)
metrics_f = get_metrics( f_train_pred, f_train,
                        f_test_pred, f_test)


# plot energy and error 
plot_energy(e_train_pred, e_train)
plot_energy(e_test_pred, e_test)

plot_energy(f_train_pred, f_train)
plot_energy(f_test_pred, f_test)

plot_error(e_train_err)
plot_error(e_test_err)



