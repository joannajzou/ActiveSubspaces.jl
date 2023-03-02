abstract type Sampler end

include("hmc.jl")
include("nuts.jl")

export HMC, NUTS, sample
