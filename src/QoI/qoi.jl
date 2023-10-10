abstract type QoI end

include("gibbs_qoi.jl")
# include("general_qoi.jl")

function assign_param(qoi::GibbsQoI, θ::Vector)
    return GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))
end


export GibbsQoI, expectation, grad_expectation, assign_param
# export GeneralQoI
