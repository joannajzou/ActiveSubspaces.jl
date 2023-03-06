using PotentialLearning
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using DPP
using ActiveSubspaces
using LinearAlgebra
using Distributions
using ACE1
using JuLIP
using JLD
using CairoMakie


## load data on configurations
ds, thermo = load_data("examples/potential/argon_LJ/lj.yaml", YAML(:Ar, u"eV", u"Å"))
ds = ds[3:end]
systems = get_system.(ds)
# positions = position.(systems)
# energies = get_values.(get_energy.(ds))
# forces = get_values.(get_forces.(ds))

# po = zeros(length(positions), 13, 3)
# fo = zeros(length(positions), 13, 3)
# for (i, p_i) in enumerate(positions)
#     po[i, :, :] = reduce(hcat, ustrip.(p_i))'
#     fo[i, :, :] = reduce(hcat, forces[i])'
# end

## create ACE basis
n_body = 2  # 2-body
max_deg = 8 # 8 degree polynomials
r0 = 1.0 # minimum distance between atoms
rcutoff = 5.0 # cutoff radius 
wL = 1.0 # Defaults, See ACE.jl documentation 
csp = 1.0 # Defaults, See ACE.jl documentation
ace = ACE([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)

## get ACE descriptors
local_descriptors = compute_local_descriptors.(systems, (ace,))
force_descriptors = compute_force_descriptors.(systems, (ace,))
lb = LBasisPotential(ace)

ds = ds .+ LocalDescriptors.(local_descriptors) .+ ForceDescriptors.(force_descriptors)
# ds = DataSet(ds)

## set up train/test split
n_train = 1000
n_test = length(ds) - n_train
ds_train, ds_test = DataSet(ds[1:n_train]), DataSet(ds[n_train+1:end]);

# fit LJ potential
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 100)
dpp_inds = get_random_subset(dpp)
lb, Σ = PotentialLearning.learn!(lb, ds_train[dpp_inds]; α = 1e-6)

# train errors
train_features = sum.(get_values.(get_local_descriptors.(ds_train[dpp_inds])))
e_train = dot.(train_features, (lb.β,) ./ 108.0)

# test errors
test_features = sum.(get_values.(get_local_descriptors.(ds_test)))
e_test = dot.(test_features, (lb.β,) ./ 108.0)

# define negative potential function
V(Φ::Vector{Float64}, θ::Vector{Float64}) = -Φ' * θ ./ 108.0

function ∇xV(ds::DataSet, θ::Vector{Float64})
    ∇xΦ = sum(get_values(get_force_descriptors(ds))) # descriptors
    return -dot.(∇xΦ, (θ,) ./ 108.0)
end

∇θV(Φ::Vector{Float64}, θ::Vector{Float64}) = -Φ ./ 108.0

# instantiate Gibbs object
πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)

# define QoI as mean energy V
q = GibbsQoI(h=(x,γ) -> V(x,γ), p=πgibbs)

# define sampling density
nsamp = 10
μθ = lb.β
Σθ = 1*I(max_deg) 
ρθ = MvNormal(μθ, Σθ) # prior on θ 
θsamp = [rand(ρθ) for i = 1:nsamp]

# compute active subspace
πg = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=lb.β)
Φsamp = sum.(get_values.(get_local_descriptors.(DataSet(ds))))
# ∇Qistest = grad_expectation(θsamp[1], q; gradh=∇θV, g=πg, xsamp=Φsamp)
∇Qis_g = map(θ -> grad_expectation(θ, q; gradh=∇θV, g=πg, xsamp=Φsamp), θsamp)
λis_g, Wis_g = ActiveSubspaces.select_eigendirections(∇Qis_g, 1e-8)
d_r = size(Wis_g, 2)

# reduced dimensional descriptors
ψ_train = reduce(hcat, (Wis_g',) .* train_features)'
ψ_test = reduce(hcat, (Wis_g',) .* test_features)'
A_train = reduce(hcat, train_features)'
A_test = reduce(hcat, test_features)'


# fit model in reduced dimension
β̂ = pinv(ψ_train, 1e-6) * e_train
train_pred = ψ_train*β̂
test_pred = ψ_test*β̂
train_err = e_train - train_pred
test_err = e_test - test_pred
σe = var(test_err)
Σe = σe * I(n_test)
Σβ̂ = Symmetric( inv(ψ' * inv(Σe) * ψ) ) # + 1e-6 * I(d_r)
μβ̂ = Σβ̂ * (ψ' * inv(Σe) * e_test)
μβ = Wis_g * μβ̂
Σβ = Wis_g * Σβ̂ * Wis_g'


# fit model in original (full) dimension
βf = A_train \ e_train
train_predf = A_train*βf # lb.β
test_predf =  A_test*βf # lb.β
train_errf = e_train - train_predf
test_errf = e_test - test_predf
σef = var(test_errf)
Σf = σef * I(n_test)
Σβf = Symmetric( inv(A' * inv(Σf) * A) )
μβf = Σβf * (A' * inv(Σf) * e_test)

# plot errors
plot_energy([train_predf, train_pred], e_train; predlab=["full model", "reduced model"])
plot_energy([test_predf, test_pred], e_test; predlab=["full model", "reduced model"])





# Plot functions ########################################################################################

function plot_energy(e_pred::Vector, e_true; predlab=nothing)
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10
    n = length(e_pred)

    if predlab === nothing
        predlab = 1:n
    end
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="E MD [eV/atom]", ylabel="E pred. [eV/atom]")
    CairoMakie.scatter!(ax, e_true, e_pred[1], markersize=7, color=(:red, 0.5), label=predlab[1])
    CairoMakie.scatter!(ax, e_true, e_pred[2], markersize=7, color=(:blue, 0.5), label=predlab[2])
    CairoMakie.lines!(ax, r0:rs:r1, r0:rs:r1)
    axislegend(ax)
    return fig
end

function plot_forces(f_pred, f_true)
    r0 = floor(minimum(f_true)); r1 = ceil(maximum(f_true))
    plot( f_true, f_pred, seriestype = :scatter, markerstrokewidth=0,
          label="", xlabel = "F DFT | eV/Å", ylabel = "F predicted | eV/Å", 
          xlims = (r0, r1), ylims = (r0, r1))
    p = plot!( r0:r1, r0:r1, label="")
    return p
end
