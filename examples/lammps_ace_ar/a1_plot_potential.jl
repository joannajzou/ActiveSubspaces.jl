include("00_spec_model.jl")
include("plotting_utils.jl")

# load posterior distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)


# check sampling density in pairwise energy plot ###########################################################################
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
    ax = Axis(f[1, 1], xlabel = "interatomic distance r (Å) ",
                       ylabel = "energies E (eV)",
                       title="Pairwise energy curve for samples from ρ(θ)")

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
    

# plot 2d projections of QoI vs. β ###########################################################################

