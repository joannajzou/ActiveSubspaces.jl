using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using DPP
using ACE1, JuLIP
using JLD
using Flux, Zygote
using OrderedCollections # metrics.jl
using ProgressBars

push!(Base.LOAD_PATH, dirname(@__DIR__))
using PotentialLearning
using ActiveSubspaces
include("examples/potential/plot_utils.jl")


# path = "examples/potential/sodium_emp/experiment/"
# run(`mkdir -p $path`)

## load data on configurations
ds, thermo = load_data("examples/potential/sodium_emp/liquify_sodium.yaml", YAML(:Na, u"eV", u"Å"))
ds, thermo = ds[220:end], thermo[220:end];
# systems = get_system.(ds)


## FIT IAP ###############################################################################

## create ACE basis
n_body = 4  # 2-body
max_deg = 8 # 8 degree polynomials
r0 = 1.0 # minimum distance between atoms
rcutoff = 5.0 # cutoff radius 
wL = 1.0 # Defaults, See ACE.jl documentation 
csp = 1.0 # Defaults, See ACE.jl documentation
ace = ACE([:Na], n_body, max_deg, wL, csp, r0, rcutoff)
lb = LBasisPotential(ace)


## compute ACE descriptors
e_descr = compute_local_descriptors(ds, ace)
# f_descr = compute_force_descriptors(ds, ace)
# JLD.save(path * "sodium_energy_descr.jld", "e_descr", e_descr)
# e_descr = JLD.load(path * "sodium_energy_descr.jld")["e_descr"]
ds = DataSet(ds .+ e_descr) # .+ f_descr


## split into train/test
n_train = 1000
n_test = length(ds) - n_train
ds_test, ds_train = ds[1:n_test], ds[n_test+1:end]

## fit potential with DPP samples
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 200)
dpp_inds = get_random_subset(dpp)
lb, Σ = PotentialLearning.learn!(lb, ds_train[dpp_inds]; α = 1e-10)


## compute descriptors
e_descr_train = sum.(get_values.(get_local_descriptors.(ds_train[dpp_inds])))
# f_descr_train = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds_train[dpp_inds]]
e_descr_test = sum.(get_values.(get_local_descriptors.(ds_test)))
# f_descr_test = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds_test]


## load energy/force data
e_train = get_all_energies(ds_train[dpp_inds])
e_test = get_all_energies(ds_test)
e_train_pred = get_all_energies(ds_train[dpp_inds], lb)
e_test_pred = get_all_energies(ds_test, lb)
e_train_err = (e_train - e_train_pred) ./ e_train
e_test_err = (e_test - e_test_pred) ./ e_test


## Compute error metrics
metrics = get_metrics( e_train_pred, e_train,
                       e_test_pred, e_test)




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
nsamp = 1000 # 50000
μθ = lb.β
Σθ = 0.1*Diagonal(abs.(lb.β))
ρθ = MvNormal(μθ, Σθ) # prior on θ 
θsamp = [rand(ρθ) for i = 1:nsamp]


# compute active subspace
πg = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=lb.β)
Φsamp = sum.(get_values.(get_local_descriptors.(ds)))
@time ∇Qistest = grad_expectation(θsamp[1], q; gradh=∇θV, g=πg, xsamp=Φsamp)
@time ∇Qis_g = map(θ -> grad_expectation(θ, q; gradh=∇θV, g=πg, xsamp=Φsamp), θsamp)
JLD.save("examples/potential/sodium_emp/gradQ_lj_M=1000.jld", "θsamp", θsamp, "∇Q", ∇Qis_g)


# active subspace 
d = 10
λas, Was = ActiveSubspaces.select_eigendirections(∇Qis_g, d)
nzi = findall(x -> x > 0, λas)


# reduced dimensional descriptors
Ψ_train = (Was',) .* e_descr_train # energy descriptors
Ψ_test = (Was',) .* e_descr_test


# fit model in reduced dimension
βas, Σas = fit(Ψ_train, e_train; α = 1e-12)
ẽ_train_pred = dot.(Ψ_train, (βas,))
ẽ_test_pred = dot.(Ψ_test, (βas,))
ẽ_train_err = (e_train - ẽ_train_pred) ./ e_train
ẽ_test_err = (e_test - ẽ_test_pred) ./ e_test


# compute error metrics
metrics_as = get_metrics( ẽ_train_pred, e_train,
                         ẽ_test_pred, e_test)


## compare to PCA
pca = ActiveSubspaces.PCA(d)
λpca, Wpca = find_subspace(θsamp, pca)
Ψp_train = (Wpca',) .* e_descr_train
Ψp_test = (Wpca',) .* e_descr_test


# fit model in reduced dimension
βpca, Σpca = fit(Ψp_train, e_train; α = 1e-12)
ê_train_pred = dot.(Ψp_train, (βpca,))
ê_test_pred = dot.(Ψp_test, (βpca,))
ê_train_err = (e_train - ê_train_pred) ./ e_train
ê_test_err = (e_test - ê_test_pred) ./ e_test


# compute error metrics
metrics_pca = get_metrics( ê_train_pred, e_train,
                         ê_test_pred, e_test)



# PLOT ########################################################################################

# plot eigenspectrum
plot_eigenspectrum([λpca[nzi], λas[nzi]]; predlab=["PCA", "Active Subspace"])
# plot errors
plot_energy([e_train_pred, ê_train_pred, ẽ_train_pred], e_train; predlab=["full model", "reduced model (PCA)", "reduced model (AS)"])
plot_energy([e_test_pred, ê_test_pred, ẽ_test_pred], e_test; predlab=["full model", "reduced model (PCA)", "reduced model (AS)"])

plot_error([e_train_err, ê_train_err, ẽ_train_err]; predlab=["full model", "reduced model (PCA)", "reduced model (AS)"])
plot_error([e_test_err, ê_test_err, ẽ_test_err]; predlab=["full model", "reduced model (PCA)", "reduced model (AS)"])


# cosine similarity
plot_cosine_sim([Wpca, Was])



# FUNCTIONS ########################################################################################

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


function calc_metrics(x_pred, x)
    x_mae = sum(abs.(x_pred .- x)) / length(x)
    x_rmse = sqrt(sum((x_pred .- x) .^ 2) / length(x))
    x_rsq = 1 - sum((x_pred .- x) .^ 2) / sum((x .- mean(x)) .^ 2)
    return x_mae, x_rmse, x_rsq
end


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


# Compute descriptors of a basis system and dataset
import InteratomicPotentials

function InteratomicPotentials.compute_local_descriptors(ds::DataSet, basis::BasisSystem)
    e_des = Vector{LocalDescriptors}(undef, length(ds))
    #for (j, sys) in ProgressBar(zip(1:length(ds), get_system.(ds)))
    Threads.@threads for (j, sys) in ProgressBar(collect(enumerate(get_system.(ds))))
        e_des[j] = LocalDescriptors(compute_local_descriptors(sys, basis))
    end
    return e_des
end


function InteratomicPotentials.compute_force_descriptors(ds::DataSet, basis::BasisSystem)
    f_des = Vector{ForceDescriptors}(undef, length(ds))
    #for (j, sys) in ProgressBar(zip(1:length(ds), get_system.(ds)))
    Threads.@threads for (j, sys) in ProgressBar(collect(enumerate(get_system.(ds))))
        f_des[j] = ForceDescriptors([[fi[i, :] for i = 1:3] 
                                     for fi in compute_force_descriptors(sys, basis)])
    end
    return f_des
end


function fit(descr_data::Vector{T}, energy_data::Vector{Float64}; α = 1e-8) where T <: Vector{Float64}
    # Form design matrices 
    AtA = sum(v * v' for v in descr_data)
    Atb = sum(v * b for (v, b) in zip(descr_data, energy_data))

    Q = pinv(AtA, α)
    β = Q * Atb
    σ = std(Atb - AtA * β)
    Σ = Symmetric(σ[1]^2 * Q)
    return β, Σ
end


function cosine_sim(v1::Vector{Float64}, v2::Vector{Float64})
    return dot(v1, v2) / (norm(v1) * norm(v2))
end


import PotentialLearning: LinearProblem, UnivariateLinearProblem, CovariateLinearProblem

function LinearProblem(ds::DataSet; T = Float64)
    d_flag, descriptors, energies = try
        #true,  compute_features(ds, GlobalSum()), get_values.(get_energy.(ds))
        true, sum.(get_values.(get_local_descriptors.(ds))), get_values.(get_energy.(ds))
        
        #true, compute_feature.(get_local_descriptors.(ds), [GlobalSum()]), get_values.(get_energy.(ds))
    catch 
        false, 0.0, 0.0 
    end
    fd_flag, force_descriptors, forces = try  
        true, [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds], get_values.(get_forces.(ds))
    catch
        false, 0.0, 0.0
    end
    if d_flag & ~fd_flag 
        dim = length(descriptors[1])
        β = zeros(T, dim)

        p = UnivariateLinearProblem(descriptors, 
                energies, 
                β, 
                [1.0],
                Symmetric(zeros(dim, dim)))
    elseif ~d_flag & fd_flag 
        dim = length(force_descriptors[1][1])
        β = zeros(T, dim)

        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
        p = UnivariateLinearProblem(force_descriptors,
            [reduce(vcat, fi) for fi in forces], 
            β, 
            [1.0], 
            Symmetric(zeros(dim, dim))
        )
        
    elseif d_flag & fd_flag 
        dim_d = length(descriptors[1])
        dim_fd = length(force_descriptors[1][1])

        if  (dim_d != dim_fd) 
            error("Descriptors and Force Descriptors have different dimension!") 
        else
            dim = dim_d
        end

        β = zeros(T, dim)
        forces =  [reduce(vcat, fi) for fi in forces]
        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]

        p = PotentialLearning.CovariateLinearProblem(energies,
                [reduce(vcat, fi) for fi in forces], 
                descriptors, 
                force_descriptors, 
                β, 
                [1.0], 
                [1.0], 
                Symmetric(zeros(dim, dim)))

    else 
        error("Either no (Energy, Descriptors) or (Forces, Force Descriptors) in DataSet")
    end
    p
end
