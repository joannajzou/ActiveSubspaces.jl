""" 
    QoI

    A struct of abstract type QoI computes the expectation of a function h(x, θ) with respect to an invariant measure p(x, θ).
"""
abstract type QoI end

include("gibbs_qoi.jl")
# include("general_qoi.jl")

function assign_param(qoi::GibbsQoI, θ::Vector)
    return GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))
end


export GibbsQoI, expectation, grad_expectation, assign_param
# export GeneralQoI
