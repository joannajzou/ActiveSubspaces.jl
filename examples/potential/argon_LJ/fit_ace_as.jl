using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using DPP
using ACE1, JuLIP
using JLD
using Flux, Zygote
using OrderedCollections # metrics.jl
using CairoMakie

push!(Base.LOAD_PATH, dirname(@__DIR__))
using PotentialLearning
using ActiveSubspaces
# include("examples/potential/utils/utils.jl")


## load data on configurations
ds, thermo = load_data("examples/potential/argon_LJ/lj.yaml", YAML(:Ar, u"eV", u"Å"))
ds = ds[3:end]
systems = get_system.(ds)


## FIT IAP ###############################################################################

## create ACE basis
n_body = 2  # 2-body
max_deg = 10 # 8 degree polynomials
r0 = 1.0 # minimum distance between atoms
rcutoff = 5.0 # cutoff radius 
wL = 1.0 # Defaults, See ACE.jl documentation 
csp = 1.0 # Defaults, See ACE.jl documentation
ace = ACE([:Ar], n_body, max_deg, wL, csp, r0, rcutoff)
lb = LBasisPotential(ace)


## compute ACE descriptors
local_descriptors = compute_local_descriptors.(systems, (ace,))
force_descriptors = compute_force_descriptors.(systems, (ace,))
ds = ds .+ LocalDescriptors.(local_descriptors) .+ ForceDescriptors.(force_descriptors)
ds = DataSet(ds)


## split into train/test
n_train = 1000
n_test = length(ds) - n_train
ds_train, ds_test = ds[1:n_train], ds[n_train+1:end]

## fit potential with DPP samples
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 100)
dpp_inds = get_random_subset(dpp)
lb, Σ = PotentialLearning.learn!(lb, ds_train[dpp_inds]; α = 1e-8)


## compute descriptors
e_descr_train = sum.(get_values.(get_local_descriptors.(ds_train[dpp_inds])))
f_descr_train = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds_train[dpp_inds]]
e_descr_test = sum.(get_values.(get_local_descriptors.(ds_test)))
f_descr_test = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds_test]


## load energy/force data
e_train, f_train = get_all_energies(ds_train[dpp_inds]), get_all_forces(ds_train[dpp_inds])
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train[dpp_inds], lb), get_all_forces(ds_train[dpp_inds], lb)
e_test_pred, f_test_pred = get_all_energies(ds_test, lb), get_all_forces(ds_test, lb)


## Compute error metrics
metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       1.0, 1.0, 1.0)


# OPTION 2: fit with least squares
## fit with least squares
A_train = reduce(hcat, e_descr_train)'
A_test = reduce(hcat, e_descr_test)'
βf = A_train \ e_train
e_train_pred = A_train*βf # lb.β
e_test_pred =  A_test*βf # lb.β
e_train_err = e_train - e_train_pred
e_test_err = e_test - e_test_pred
σef = var(e_test_err)
Σf = σef * I(n_test)
Σβf = Symmetric( inv(A_test' * inv(Σf) * A_test) )
μβf = Σβf * (A_test' * inv(Σf) * e_test)

metrics2 = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       1.0, 1.0, 1.0)


## DIMENSION REDUCTION ####################################################################

# define negative potential function
V(Φ::Vector{Float64}, θ::Vector{Float64}) = -Φ' * θ

function ∇xV(ds::DataSet, θ::Vector{Float64})
    ∇xΦ = reduce(vcat, get_values(get_force_descriptors(ds)))
    return vcat([-dB' * lp.β for dB in ∇xΦ])
end

∇θV(Φ::Vector{Float64}, θ::Vector{Float64}) = -Φ

# instantiate Gibbs object
πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)

# define QoI as mean energy V
q = GibbsQoI(h=V, p=πgibbs) # (x,γ) -> V(x,γ)

# define sampling density
nsamp = 10 # 50000
μθ = lb.β
Σθ = 0.5*Diagonal(abs.(lb.β))
ρθ = MvNormal(μθ, Σθ) # prior on θ 
θsamp = [rand(ρθ) for i = 1:nsamp]

# compute active subspace
πg = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=lb.β)
Φsamp = sum.(get_values.(get_local_descriptors.(ds)))
@time ∇Qistest = grad_expectation(θsamp[1], q; gradh=∇θV, g=πg, xsamp=Φsamp)
@time ∇Qis_g = map(θ -> grad_expectation(θ, q; gradh=∇θV, g=πg, xsamp=Φsamp), θsamp)
JLD.save("examples/potential/argon_LJ/gradQ_lj_M=10.jld", "∇Q", ∇Qis_g)

λis_g, Wis_g = ActiveSubspaces.select_eigendirections(∇Qis_g, 1e-8)
plot_eigenspectrum(λis_g)
d̃ = size(Wis_g, 2)



# reduced dimensional descriptors
Ψ_train = reduce(hcat, (Wis_g',) .* e_descr_train)' # energy descriptors
Ψ_test = reduce(hcat, (Wis_g',) .* e_descr_test)'
# dΨ_train = reduce(hcat, [(Wis_g',) .* fi for fi in f_descr_train)' # force descriptors
# dΨ_test = reduce(hcat, (Wis_g',) .* f_descr_test)'




# fit model in reduced dimension
β̂ = pinv(Ψ_train, 1e-6) * e_train
ẽ_train_pred = Ψ_train*β̂
ẽ_test_pred = Ψ_test*β̂
ẽ_train_err = e_train - ẽ_train_pred
ẽ_test_err = e_test - ẽ_test_pred

# σe = var(test_err)
# Σe = σe * I(n_test)
# Σβ̂ = Symmetric( inv(Ψ_test' * inv(Σe) * Ψ_test) ) # + 1e-6 * I(d_r)
# μβ̂ = Σβ̂ * (Ψ_test' * inv(Σe) * e_test)
# μβ = Wis_g * μβ̂
# Σβ = Wis_g * Σβ̂ * Wis_g'


# compute error metrics
metrics_r = get_metrics( ẽ_train_pred, e_train, f_train_pred, f_train,
                       ẽ_test_pred, e_test, f_test_pred, f_test,
                       1.0, 1.0, 1.0)


# plot errors
plot_energy([e_train_pred, ẽ_train_pred], e_train; predlab=["full model", "reduced model"])
plot_energy([e_test_pred, ẽ_test_pred], e_test; predlab=["full model", "reduced model"])

plot_error([e_train_err, ẽ_train_err]; predlab=["full model", "reduced model"])



# Plot functions ########################################################################################

function plot_energy(e_pred::Vector, e_true; predlab=nothing)
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10
    n = length(e_pred)

    if predlab === nothing
        predlab = 1:n
    end
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="E MD [eV/atom]", ylabel="E pred. [eV/atom]")
    scatter!(ax, e_true, e_pred[1], markersize=7, color=(:red, 0.5), label=predlab[1])
    scatter!(ax, e_true, e_pred[2], markersize=7, color=(:blue, 0.5), label=predlab[2])
    lines!(ax, r0:rs:r1, r0:rs:r1)
    axislegend(ax)
    return fig
end


function plot_error(err_pred::Vector; predlab=nothing)
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="error [eV]", ylabel="count")
    hist!(ax, err_pred[1], color=(:red, 0.5), label=predlab[1])
    hist!(ax, err_pred[2], color=(:blue, 0.5), label=predlab[2])
    axislegend(ax)
    return fig
end



function get_all_energies(ds::DataSet)
    return [get_values(get_energy(ds[c])) for c in 1:length(ds)]
end

function get_all_forces(ds::DataSet)
    return reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c]))
                                    for c in 1:length(ds)]))
end

function get_all_energies(ds::DataSet, lp::LBasisPotential)
    Bs = sum.(get_values.(get_local_descriptors.(ds)))
    return dot.(Bs, [lp.β])
end

function get_all_forces(ds::DataSet, lp::LBasisPotential)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    return vcat([dB' * lp.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
end


function plot_eigenspectrum(λ::Vector{Float64})
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Spectrum of gradient covariance matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λ_i)",
            yscale=log10,
    )
    scatterlines!(ax, 1:length(λ), λ)
    return fig
end