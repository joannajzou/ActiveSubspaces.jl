include("00_spec_model.jl")
include("qoi_meanenergy.jl")
include("plotting_utils.jl")


# load posterior distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)


# evaluate MD trajectory #############################################################################
coeff = "mean"
Tend = Int(5E6)       # number of steps
dT = 500
dt = 0.0025

ds0 = load_data("$(biasdir)coeff_$coeff/data.xyz", ExtXYZ(u"eV", u"Å"))
# Filter first configuration (zero energy)
ds = ds0[2001:end]

systems = get_system.(ds)
n_atoms = length(first(systems))
pos = position.(systems)
dists_origin = map(x -> ustrip.(norm.(x)), pos)
energies = get_values.(get_energy.(ds))
Φsamp = sum.(get_values.(compute_local_descriptors(ds, ace)))
time_range = (2001:length(ds0)) .* dT * dt


# trace plots of distance to origin and energies
size_inches = (12, 10)
size_pt = 72 .* size_inches
fig = Figure(resolution=size_pt, fontsize=16)
ax1 = Axis(fig[1, 1], xlabel="τ | ps", ylabel="Distance from origin | Å")
ax2 = Axis(fig[2, 1], xlabel="τ | ps", ylabel="Lennard Jones energy | eV")
for i = 1:n_atoms
    lines!(ax1, time_range, map(x -> x[i], dists_origin))
end
lines!(ax2, time_range, energies)
fig

# trace plots in descriptor space
fig2 = plot_mcmc_trace(Φsamp)
fig3 = plot_mcmc_autocorr(Φsamp)
fig4 = plot_mcmc_marginals(Φsamp)

# standard error
nrng = 1000:50:length(time_range)
qoim = GibbsQoI(h=x -> q.h(x, πβ.μ), p=Gibbs(q.p, θ=πβ.μ))
se_bm = [MCSEbm(qoim, Φsamp[1:n]) for n in nrng]
# se_obm = [MCSEobm(qoim, Φsamp[1:n]) for n in 1000:500:20000]
ess = EffSampleSize(Φsamp)

f = Figure(resolution=(800, 800))
ax = Axis(f[1, 1], xlabel="τ | ps", ylabel="MCSE")
lines!(ax, nrng, se_bm, label="BM")
# lines!(ax, 1000:500:20000, se_obm, label="OBM")
# axislegend(ax)
f


# check sampling density in pairwise energy plot ###########################################################################
# compute descriptors along radial axis
r = 1.0:0.01:5 # radial distance
box = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] # domain
bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
system = [FlexibleSystem([AtomsBase.Atom(:Ar, [0.0, 0.0, 0.0] * u"Å"), AtomsBase.Atom(:Ar, [ri, 0.0, 0.0] * u"Å")],
    box * u"Å", bcs) for ri in r]
B = [sum(compute_local_descriptors(sys, ace)) for sys in system] # descriptors

# get predicted energies
βsamp = [rand(πβ) for i = 1:1000]
energies_pred = [Bi' * βi for Bi in B, βi in βsamp]
energies_mean = [Bi' * πβ.μ for Bi in B]
# JLD.save("$(simdir)energies_ref.jld", "energies", energies_mean)

# plot
with_theme(custom_theme) do
    f = Figure(resolution=(650, 650))
    ax = Axis(f[1, 1], xlabel="interatomic distance r (Å) ",
        ylabel="energies E (eV)",
        title="Pairwise energy curve for samples from ρ(θ)")

    [lines!(ax, r, energies_pred[:, i], color=(:grey, 0.4), linewidth=1) for i = 1:1000]
    lines!(ax, r, energies_pred[:, end], color=(:grey, 0.4), linewidth=1, label="samples")
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


# plot 2d projections of QoI vs. β ###########################################################################

