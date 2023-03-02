abstract type QoI end

include("gibbs_qoi.jl")
include("general_qoi.jl")

export GibbsQoI, GeneralQoI, expectation, expectation_, grad_expectation


function expectation_(qoi::QoI, n::Int, sampler::Sampler, ρ0::Distribution)  # integration with MCMC samples
    x = rand(qoi.p, n, sampler, ρ0) 
    return sum(qoi.h.(x)) / length(x)
end


function expectation_(qoi::QoI, xsamp::Vector) # integration with samples provided       
    return sum(qoi.h.(xsamp)) / length(xsamp)
end