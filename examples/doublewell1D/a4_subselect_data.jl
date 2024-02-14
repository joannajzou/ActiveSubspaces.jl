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
include("adaptive_is_utils.jl")


# load data ------------------------------------------------------------------
triallbls = ["MC", "ISM", "ISMa", "ISMaw"]
nsamptot = 8000
nrepl = 5

# split total dataset
nsplit = 8
split = Int(nsamptot/nsplit)

for repl in 1:nrepl
    ∇Qref = JLD.load("data$modnum/repl$repl/DW1D_Quad_grad_nsamp=$nsamptot.jld")["∇Q"]
    ∇Qmc = JLD.load("data$modnum/repl$repl/DW1D_MC_grad_nsamp=$nsamptot.jld")["∇Q"]
    ∇Qis = JLD.load("data$modnum/repl$repl/DW1D_ISM_grad_nsamp=$nsamptot.jld")["∇Q"]
    ∇Qisa = JLD.load("data$modnum/repl$repl/DW1D_ISMa_grad_nsamp=$nsamptot.jld")["∇Q"]
    ∇Qisaw = JLD.load("data$modnum/repl$repl/DW1D_ISMaw_grad_nsamp=$nsamptot.jld")["∇Q"]

    ∇Qref_vec = [∇Qref[((k-1)*split+1):k*split] for k = 1:nsplit]
    ∇Qmc_vec = [∇Qmc[((k-1)*split+1):k*split] for k = 1:nsplit]
    ∇Qis_vec = [∇Qis[((k-1)*split+1):k*split] for k = 1:nsplit]
    ∇Qisa_vec = [∇Qisa[((k-1)*split+1):k*split] for k = 1:nsplit]
    ∇Qisaw_vec = [∇Qisaw[((k-1)*split+1):k*split] for k = 1:nsplit]


    for i = 1:nsplit
        JLD.save("data$modnum/subsets/DW1D_Quad_grad_split=$(repl)$(i).jld", "∇Q", ∇Qref_vec[i])
        JLD.save("data$modnum/subsets/DW1D_MC_grad_split=$(repl)$(i).jld", "∇Q", ∇Qmc_vec[i])
        JLD.save("data$modnum/subsets/DW1D_ISM_grad_split=$(repl)$(i).jld", "∇Q", ∇Qis_vec[i])
        JLD.save("data$modnum/subsets/DW1D_ISMa_grad_split=$(repl)$(i).jld", "∇Q", ∇Qisa_vec[i])
        JLD.save("data$modnum/subsets/DW1D_ISMaw_grad_split=$(repl)$(i).jld", "∇Q", ∇Qisaw_vec[i])
    end
end