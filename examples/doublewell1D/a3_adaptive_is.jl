using ActiveSubspaces
using Distributions
using LinearAlgebra
using StatsBase
using JLD

# select model
modnum = 4
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("data_utils.jl")
include("plotting_utils.jl")



# load data ##############################################################################
nsamptot = 8000
# load biasing dist.
λu = JLD.load("data$modnum/repl1/DW1D_ISM_nsamp=$(nsamptot).jld")["mm_cent"]
mmu = MixtureModel(λu, πgibbs2)
λa = JLD.load("data$modnum/repl1/DW1D_ISaM_nsamp=$(nsamptot).jld")["mm_cent"][end]
αa = JLD.load("data$modnum/repl1/DW1D_ISaM_nsamp=$(nsamptot).jld")["mm_wts"][end]
mma = MixtureModel(λa, πgibbs2, αa)




# error in IS expectation ##############################################################


Qgq = [expectation(λ, q, GQint) for λ in λu]
Qmc = [expectation(λ, q, MCint) for λ in λu]


xsamp = rand(mmu, nMC, nuts, ρx0; burn=2000)
ISint = ISSamples(mm, xsamp)
Qis_u = [expectation(λ, q, ISint)[1] for λ in λu]

xsamp = rand(mma, nMC, nuts, ρx0; burn=2000)
ISint = ISSamples(mm, xsamp)
Qis_a = [expectation(λ, q, ISint)[1] for λ in λu]




Qcomp = [Qmc, Qis_u, Qis_a]
err = [error_vector(Qgq, Q) for Q in Qcomp]

function error_vector(vref, v)
    return abs(EuclideanDistance(vref, v))
end
