include("00_spec_model.jl")
include("plotting_utils.jl")


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
println("dimension: $d")


# check sampling density in pairwise energy plot
# compute descriptors along radial axis
r = 1.0:0.01:5 # radial distance
box = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] # domain
bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
system = [FlexibleSystem([AtomsBase.Atom(:Ar, [0.0, 0.0, 0.0] * u"Å"), AtomsBase.Atom(:Ar, [ri, 0.0, 0.0] * u"Å")],
                        box * u"Å", bcs) for ri in r]
B = [sum(compute_local_descriptors(sys, ace)) for sys in system] # descriptors

# get predicted energies
βsamp = [rand(πβ) for i=1:1000]
energies_pred = [Bi' * βi for Bi in B, βi in βsamp]
energies_mean = [Bi' * πβ.μ for Bi in B]
# JLD.save("$(simdir)energies_ref.jld", "energies", energies_mean)

# plot
with_theme(custom_theme) do
    f = Figure(resolution=(650,650))
    ax = Axis(f[1, 1], xlabel = "interatomic distance r (Å) ", ylabel = "energies E (eV)", title="Pairwise energy curve for samples from ρ(θ)")

    [lines!(ax, r, energies_pred[:,i], color=(:grey, 0.4), linewidth=1) for i = 1:1000]
    lines!(ax, r, energies_pred[:,end], color=(:grey, 0.4), linewidth=1, label="samples")
    lines!(ax, r, energies_mean, color=:red, linewidth=2, label="mean")
    axislegend(ax, position=:rb)
    f
end





# # Get true energies
# # A - without units
# ϵ = 0.01034
# σ = 1.0
# lj = InteratomicPotentials.LennardJones(ϵ, σ, 4.0, [:Ar])
# energies_a = ustrip.([InteratomicPotentials.potential_energy(sys, lj) for sys in system])

# # B - with units, ustrip
# ϵ = 0.01034 * u"eV"
# σ = 1.0 * u"Å"
# lj = InteratomicPotentials.LennardJones(ϵ, σ, 4.0 * u"Å", [:Ar])
# energies_b = [ustrip.(uconvert.(u"eV", InteratomicPotentials.potential_energy(s, lj))) for s in system]
# energies_b2 = [ustrip.(u"eV", InteratomicPotentials.potential_energy(s, lj)) for s in system]

# # C - with units, austrip
# ϵ = 0.01034 * u"eV"
# σ = 1.0 * u"Å"
# lj = InteratomicPotentials.LennardJones(ϵ, σ, 4.0 * u"Å", [:Ar])
# energies_c = [austrip.(uconvert.(u"eV", InteratomicPotentials.potential_energy(s, lj))) for s in system]

# with_theme(custom_theme) do
#     f = Figure(resolution=(650,650))
#     ax = Axis(f[1, 1], xlabel = "interatomic distance r (Å) ", ylabel = "energies E (eV)", title="Pairwise energy curve for samples from ρ(θ)")
#     lines!(ax, r, energies_a, label="A")
#     lines!(ax, r, energies_b, label="B")
#     lines!(ax, r, energies_c, label="C")
#     axislegend(ax, position=:rb)
#     f
# end
    