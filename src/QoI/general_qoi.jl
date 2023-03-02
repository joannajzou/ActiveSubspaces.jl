import Base: @kwdef

"""
@kwdef mutable struct GeneralQoI <: QoI

Defines the struct for computing the quantity of interest (QoI) as the expected value of a function h over a probability measure p given input parameters θ, e. g. E_p[h(θ)]

# Arguments
- `h :: Function`     : function to evaluate expectation, receiving inputs (x,θ)
- `p :: Distribution` : probability distribution with θ == nothing (undefined parameters)

"""
@kwdef mutable struct GeneralQoI <: QoI
    h :: Function                   # function to evaluate expectation, receiving inputs (x,θ)
    p :: Distribution               # probability distribution, receiving inputs (x)
end


function expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI; kwargs...) # for GeneralQoI
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    A = Dict(kwargs) # arguments

    if haskey(A, :ξ) & haskey(A, :w)                                            # quadrature integration
        return expectation_(qoim, A[:ξ], A[:w])

    elseif haskey(A, :g) & haskey(A, :xsamp)                                    # importance sampling (samples provided)
        return expectation_(qoim, A[:g], A[:xsamp])

    elseif haskey(A, :g) & haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0) # importance sampling (by MCMC)
        return expectation_(qoim, A[:g], A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :n)                                        # importance sampling (by MC)                
        return expectation_(qoim, A[:g], A[:n]) 

    elseif haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)                 # MCMC sampling
        return expectation_(qoim, A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :n)                                                        # Monte Carlo sampling
        return expectation_(qoim, A[:n])

    elseif haskey(A, :xsamp)                                                    # samples provided
        return expectation_(qoim, A[:xsamp])

    else
        println("ERROR: key word arguments missing or invalid")

    end
end


function expectation_(qoi::GeneralQoI, ξ::Vector{<:Real}, w::Vector{<:Real})    # quadrature integration
    h̃(x) = qoi.h(x) * pdf(qoi.p, x)
    return sum(w * h̃.(ξ))
end


function expectation_(qoi::GeneralQoI, n::Int)                                  # Monte Carlo integration                       
    x = rand(qoi.p, n) 
    return sum(qoi.h.(x)) / length(x)
end


function expectation_(qoi::GeneralQoI, g::Distribution, xsamp::Vector)          # importance sampling (samples provided)
    x = xsamp
    try # g has an unnormalized pdf
        w(x) = pdf(qoi.p, x) / updf(g, x)
        return sum( qoi.h.(x) .* w.(x) )
    catch # g does not have an unnormalized pdf
        w(x) = pdf(qoi.p, x) / pdf(g, x)
        return sum( qoi.h.(x) .* w.(x) )
    end
end


function expectation_(qoi::GeneralQoI, g::Distribution, n::Int, sampler::Sampler, ρ0::Distribution) # importance sampling (with MCMC samples)
    x = rand(g, n, sampler, ρ0) 
    try # g has an unnormalized pdf
        w(x) = pdf(qoi.p, x) / updf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    catch # g does not have an unnormalized pdf
        w(x) = pdf(qoi.p, x) / pdf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    end
end


function expectation_(qoi::GeneralQoI, g::Distribution, n::Int)                 # importance sampling (with MC samples)          
    x = rand(g, n)
    w(x) = pdf(qoi.p, x) / pdf(g, x)
    return sum( qoi.h.(x) .* w.(x) )
end


function grad_expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI; kwargs...)
   # compute gradient of h
    ∇θh(x, γ) = ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    
    # compute inner expectation E_p[∇θV]
    E_qoi = GeneralQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV = expectation(θ, E_qoi; kwargs...)

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) + qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GeneralQoI(h=hh, p=qoi.p)
    return expectation(θ, hh_qoi; kwargs...)

end
