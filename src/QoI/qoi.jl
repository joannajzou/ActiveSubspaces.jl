abstract type QoI end

include("gibbs_qoi.jl")
include("general_qoi.jl")

export GibbsQoI, GeneralQoI, expectation, grad_expectation
