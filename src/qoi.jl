import Base: @kwdef
abstract type QoI end

"""
@kwdef mutable struct GibbsQoI <: QoI

Defines the struct for computing the quantity of interest (QoI) as the expected value of a function h over Gibbs measure p given input parameters θ, e. g. E_p[h(θ)]

# Arguments
- `h :: Function`   : function to evaluate expectation, receiving inputs (x,θ)
- `p :: Gibbs`      : Gibbs (invariant) distribution with θ == nothing (undefined parameters)

"""
@kwdef mutable struct GibbsQoI <: QoI
    h :: Function                   # function to evaluate expectation, receiving inputs (x,θ)
    p :: Gibbs                      # Gibbs (invariant) distribution, receiving inputs (x,θ)
end


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


"""
function Expectation(θ:: Union{Real,Vector{<:Real}}, qoi::QoI; kwargs...)

Evaluates the expectation E_p[h(θ)], where qoi.h(θ) is the random variable and qoi.p is the probability measure 

# Arguments
- `θ :: Union{Real, Vector{<:Real}}`    : parameters to evaluate expectation
- `qoi :: QoI`                          : QoI object containing h and p
- `kwargs`                              : keyword arguments for specifying method of integration

# Keyword arguments for Gauss quadrature integration
- `ξ :: Vector{<:Real}`                 : quadrature points
- `w :: Vector{<:Real}`                 : quadrature weights

# Keyword arguments for integration by MC sampling
This method may only be implemented when qoi.p can be analytically sampled using rand().
- `n :: Int`                            : number of samples

# Keyword arguments for integration by MCMC sampling
This method is implemented when qoi.p cannot be analytically sampled.
- `n :: Int`                            : number of samples
- `sampler :: Sampler`                  : type of sampler (see `Sampler`)
- `ρ0 :: Distribution`                  : prior distribution of the state

# Keyword arguments for integration with samples provided 
This method is implemented with the user providing samples from qoi.p. 
- `xsamp :: Vector`                     : fixed set of samples

# Keyword arguments for integration by importance sampling (sampling by standard MC)
This method is implemented when the biasing distribution g can be analytically sampled using rand().
- `g :: Distribution`                   : biasing distribution
- `n :: Int`                            : number of samples

# Keyword arguments for integration by importance sampling (sampling by MCMC)
This method is implemented when the biasing distribution g cannot be analytically sampled.
- `g :: Distribution`                   : biasing distribution
- `n :: Int`                            : number of samples
- `sampler :: Sampler`                  : type of sampler (see `Sampler`)
- `ρ0 :: Distribution`                  : prior distribution of the state

# Keyword arguments for integration by importance sampling (with samples provided)
This method is implemented with the user providing samples from the biasing distribution. 
- `g :: Distribution`                   : biasing distribution
- `xsamp :: Vector`                     : fixed set of samples


""" 
function Expectation(θ:: Union{Real, Vector{<:Real}}, qoi::GibbsQoI; kwargs...) # for GibbsQoI
    # fix parameters
    qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))

    A = Dict(kwargs) # arguments
    
    if haskey(A, :ξ) & haskey(A, :w)                                            # quadrature integration
        return Expectation_(qoim, A[:ξ], A[:w])

    elseif haskey(A, :g) & haskey(A, :xsamp)                                    # importance sampling (samples provided)
        return Expectation_(qoim, A[:g], A[:xsamp])

    elseif haskey(A, :g) & haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0) # importance sampling (by MCMC)
        return Expectation_(qoim, A[:g], A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :n)                                        # importance sampling (by MC)                                                        
        return Expectation_(qoim, A[:g], A[:n]) 

    elseif haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)                 # MCMC sampling
        return Expectation_(qoim, A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :xsamp)                                                    # samples provided
        return Expectation_(qoim, A[:xsamp])

    else
        println("ERROR: key word arguments missing or invalid")

    end
end


function Expectation(θ::Union{Real, Vector{<:Real}}, qoi::GeneralQoI; kwargs...) # for GeneralQoI
    # fix parameters
    qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)

    A = Dict(kwargs) # arguments

    if haskey(A, :ξ) & haskey(A, :w)                                            # quadrature integration
        return Expectation_(qoim, A[:ξ], A[:w])

    elseif haskey(A, :g) & haskey(A, :xsamp)                                    # importance sampling (samples provided)
        return Expectation_(qoim, A[:g], A[:xsamp])

    elseif haskey(A, :g) & haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0) # importance sampling (by MCMC)
        return Expectation_(qoim, A[:g], A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :n)                                        # importance sampling (by MC)                
        return Expectation_(qoim, A[:g], A[:n]) 

    elseif haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)                 # MCMC sampling
        return Expectation_(qoim, A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :n)                                                        # Monte Carlo sampling
        return Expectation_(qoim, A[:n])

    elseif haskey(A, :xsamp)                                                    # samples provided
        return Expectation_(qoim, A[:xsamp])

    else
        println("ERROR: key word arguments missing or invalid")

    end
end


"""
function Expectation_(qoi::QoI; kwargs...)

Helper functions for evaluating the expectation. See `Expectation` for more information.

"""
# quadrature integration ##############################################################################
function Expectation_(qoi::GibbsQoI, ξ::Vector{<:Real}, w::Vector{<:Real})   
    Z = normconst(qoi.p, ξ, w)
    h̃(x) = qoi.h(x) * updf(qoi.p, x) / Z
    return sum(w .* h̃.(ξ))
end

function Expectation_(qoi::GeneralQoI, ξ::Vector{<:Real}, w::Vector{<:Real})    
    h̃(x) = qoi.h(x) * pdf(qoi.p, x)
    return sum(w * h̃.(ξ))
end


# MCMC integration ####################################################################################
function Expectation_(qoi::QoI, n::Int, sampler::Sampler, ρ0::Distribution)  
    x = rand(qoi.p, n, sampler, ρ0) 
    return sum(qoi.h.(x)) / length(x)
end


# Integration with samples from p provided ############################################################
function Expectation_(qoi::QoI, xsamp::Vector)                               
    return sum(qoi.h.(xsamp)) / length(xsamp)
end


# Monte Carlo integration #############################################################################
function Expectation_(qoi::GeneralQoI, n::Int)                               
    x = rand(qoi.p, n) 
    return sum(qoi.h.(x)) / length(x)
end


# Importance sampling #############################################################################
function Expectation_(qoi::GibbsQoI, g::Distribution, xsamp::Vector)         
    x = xsamp
    try # g has an unnormalized pdf
        w(x) = updf(qoi.p, x) / updf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    catch # g does not have an unnormalized pdf
        w(x) = updf(qoi.p, x) / pdf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    end
end


function Expectation_(qoi::GeneralQoI, g::Distribution, xsamp::Vector)         
    x = xsamp
    try # g has an unnormalized pdf
        w(x) = pdf(qoi.p, x) / updf(g, x)
        return sum( qoi.h.(x) .* w.(x) )
    catch # g does not have an unnormalized pdf
        w(x) = pdf(qoi.p, x) / pdf(g, x)
        return sum( qoi.h.(x) .* w.(x) )
    end
end


function Expectation_(qoi::GibbsQoI, g::Distribution, n::Int, sampler::Sampler, ρ0::Distribution) # importance sampling
    x = rand(g, n, sampler, ρ0) 
    try # g has an unnormalized pdf
        w(x) = updf(qoi.p, x) / updf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    catch # g does not have an unnormalized pdf
        w(x) = updf(qoi.p, x) / pdf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    end
end


function Expectation_(qoi::GeneralQoI, g::Distribution, n::Int, sampler::Sampler, ρ0::Distribution) # importance sampling
    x = rand(g, n, sampler, ρ0) 
    try # g has an unnormalized pdf
        w(x) = pdf(qoi.p, x) / updf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    catch # g does not have an unnormalized pdf
        w(x) = pdf(qoi.p, x) / pdf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    end
end


function Expectation_(qoi::GibbsQoI, g::Distribution, n::Int)               
    x = rand(g, n)
    w(x) = updf(qoi.p, x) / pdf(g, x)
    return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
end


function Expectation_(qoi::GeneralQoI, g::Distribution, n::Int)              
    x = rand(g, n)
    w(x) = pdf(qoi.p, x) / pdf(g, x)
    return sum( qoi.h.(x) .* w.(x) )
end



"""
function GradExpectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI; kwargs...)

Computes the gradient of the QoI with respect to the parameters θ, e. g. ∇θ E_p[h(θ)].
See `Expectation` for more information.

# Arguments
- `θ :: Union{Real, Vector{<:Real}}`    : parameters to evaluate expectation
- `qoi :: QoI`                          : QoI object containing h and p
- `kwargs`                              : keyword arguments for specifying method of integration

"""
function GradExpectation(θ::Union{Real, Vector{<:Real}}, qoi::GibbsQoI; kwargs...)
    ∇θh(x, γ) = ForwardDiff.gradient(γ -> qoi.h(x,γ), γ)
    # A = Dict(kwargs)
    # println(A)
    
    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoi.p.∇θV, p=qoi.p)
    E_∇θV = Expectation(θ, E_qoi; kwargs...)

    # compute outer expectation
    hh(x, γ) = ∇θh(x, γ) + qoi.p.β * qoi.h(x, γ) * (qoi.p.∇θV(x, γ) - E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoi.p)
    return Expectation(θ, hh_qoi; kwargs...)

end



