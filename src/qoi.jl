import Base: @kwdef
abstract type QoI end

"""
@kwdef mutable struct GibbsQoI <: Expectation

Computes the quantity of interest (QoI) as the expected value of a function h over Gibbs measure p given input parameters θ, e. g. E_p[h(θ)]

# Arguments
- `h :: Function`   : function to evaluate expectation, receiving inputs (x,θ)
- `p :: Gibbs`      : Gibbs (invariant) distribution, receiving inputs (x,θ)

"""
@kwdef mutable struct GibbsQoI <: QoI
    h :: Function                   # function to evaluate expectation, receiving inputs (x,θ)
    p :: Gibbs                      # Gibbs (invariant) distribution, receiving inputs (x,θ)
end


"""
function Expectation(θ, qoi::GibbsQoI; kwargs...)
    θ :: Union{Real, Vector{<:Real}}    # parameters to evaluate expectation
    qoi :: GibbsQoI                     # GibbsQoI object containing h and p
    kwargs                              # keyword arguments 
    
""" 
function Expectation(θ, qoi::GibbsQoI; kwargs...)
    # fix parameters
    try
        global qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))
    catch
        global qoim = GibbsQoI(h = qoi.h, p=Gibbs(qoi.p, θ=θ))
    end

    A = Dict(kwargs) # arguments
    
    if haskey(A, :ξ) & haskey(A, :w)                                        # quadrature integration
        return Expectation_(qoim, A[:ξ], A[:w])

    elseif haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)             # MCMC sampling
        return Expectation_(qoim, A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :xsamp)                                # importance sampling
        return Expectation_(qoim, A[:g], A[:xsamp])

    elseif haskey(A, :g) & haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)
        return Expectation_(qoim, A[:g], A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :n)                                                          
        return Expectation_(qoim, A[:g], A[:n]) 

    else
        println("ERROR: key word arguments missing or invalid")

    end
end


function Expectation_(qoi::GibbsQoI, ξ::Vector{<:Real}, w::Vector{<:Real})   # quadrature integration
    Z = normconst(qoi.p, ξ, w)
    h̃(x) = qoi.h(x) * updf(qoi.p, x) / Z
    return sum(w .* h̃.(ξ))
end


function Expectation_(qoi::QoI, n::Int, sampler::Sampler, ρ0::Distribution)  # MCMC sampling
    x = rand(qoi.p, n, sampler, ρ0) 
    return sum(qoi.h.(x)) / length(x)
end


function Expectation_(qoi::GibbsQoI, g::Distribution, xsamp::Vector)         # importance sampling
    x = xsamp
    try 
        w(x) = updf(qoi.p, x) / updf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    catch 
        w(x) = updf(qoi.p, x) / pdf(g, x)
        return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
    end
end


function Expectation_(qoi::GibbsQoI, g::Distribution, n::Int, sampler::Sampler, ρ0::Distribution) # importance sampling
    x = rand(g, n, sampler, ρ0) 
    w(x) = updf(qoi.p, x) / updf(g, x)
    return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
end


function Expectation_(qoi::GibbsQoI, g::Distribution, n::Int)                # importance sampling
    x = rand(g, n)
    w(x) = updf(qoi.p, x) / pdf(g, x)
    return sum( qoi.h.(x) .* w.(x) ) / sum(w.(x))
end




"""
@kwdef mutable struct GeneralQoI <: Expectation
    h :: Function                   # function to evaluate expectation, receiving inputs (x,θ)
    p :: Distribution               # probability distribution, receiving inputs (x)
end
    
Computes the quantity of interest (QoI) as the expected value of a function h over a general measure p given input parameters θ, e. g. E_p[h(θ)]
""" 
@kwdef mutable struct GeneralQoI <: QoI
    h :: Function                   # function to evaluate expectation, receiving inputs (x,θ)
    p :: Distribution               # probability distribution, receiving inputs (x)
end


"""
function Expectation(θ, qoi::GeneralQoI; kwargs...)
    θ :: Union{Real, Vector{<:Real}}    # parameters to evaluate expectation
    qoi :: GeneralQoI                   # GeneralQoI object containing h and p
    
""" 
function Expectation(θ, qoi::GeneralQoI; kwargs...)
    try
        qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)
    catch
        qoim = GeneralQoI(h = qoi.h, p=qoi.p)
    end


    A = Dict(kwargs)

    if haskey(A, :ξ) & haskey(A, :w)                                        # quadrature integration
        return Expectation_(qoim, A[:ξ], A[:w])

    elseif haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)             # MCMC sampling
        return Expectation_(qoim, A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :xsamp)                                # importance sampling
        return Expectation_(qoim, A[:g], A[:xsamp])

    elseif haskey(A, :g) & haskey(A, :n) & haskey(A, :sampler) & haskey(A, :ρ0)
        return Expectation_(qoim, A[:g], A[:n], A[:sampler], A[:ρ0])

    elseif haskey(A, :g) & haskey(A, :n)                                                          
        return Expectation_(qoim, A[:g], A[:n]) 

    elseif haskey(A, :n)                                                    # Monte Carlo sampling
        return Expectation_(qoim, A[:n])

    else
        println("ERROR: key word arguments missing or invalid")

    end
end


function Expectation_(qoi::GeneralQoI, ξ::Vector{<:Real}, w::Vector{<:Real}) # quadrature integration
    h̃(x) = qoi.h(x) * pdf(qoi.p, x)
    return sum(w * h̃.(ξ))
end


function Expectation_(qoi::GeneralQoI, g::Distribution, xsamp::Vector)         # importance sampling
    x = xsamp
    try 
        w(x) = updf(qoi.p, x) / updf(g, x)
    catch 
        w(x) = updf(qoi.p, x) / pdf(g, x)
    end
    return sum( qoi.h.(x) .* w.(x) )
end


function Expectation_(qoi::GeneralQoI, g::Distribution, n::Int, sampler::Sampler, ρ0::Distribution) # importance sampling
    x = rand(g, n, sampler, ρ0) 
    w(x) = pdf(qoi.p, x) / updf(g, x)
    return sum( qoi.h.(x) .* w.(x) )
end


function Expectation_(qoi::GeneralQoI, g::Distribution, n::Int)                # importance sampling
    x = rand(g, n)
    w(x) = pdf(qoi.p, x) / pdf(g, x)
    return sum( qoi.h.(x) .* w.(x) )
end


function Expectation_(qoi::GeneralQoI, n::Int)                               # Monte Carlo sampling
    x = rand(qoi.p, n) 
    return sum(qoi.h.(x)) / length(x)
end



"""
function GradExpectation(θ, qoi::GibbsQoI; kwargs...)
    θ :: Union{Real, Vector{<:Real}}    # parameters to evaluate expectation
    qoi :: GibbsQoI                     # GibbsQoI object containing h and p
end
    
""" 
function GradExpectation(θ, qoi::GibbsQoI; kwargs...)
    try # h function of (x,θ)
        ∇θh(x) = ForwardDiff.gradient(θ -> qoi.h(x,θ), θ)
        qoim = GibbsQoI(h = x -> qoi.h(x, θ), p=Gibbs(qoi.p, θ=θ))
    catch # h function of x only
        ∇θh(x) = 0
        qoim = GibbsQoI(h = qoi.h, p=Gibbs(qoi.p, θ=θ))
    end
    
    # compute inner expectation E_p[∇θV]
    E_qoi = GibbsQoI(h=qoim.p.∇θV, p=qoim.p)
    E_∇θV = Expectation(θ, E_qoi; kwargs)

    # compute outer expectation
    hh(x) = ∇θh(x) + qoim.p.β * qoim.h(x) * (qoim.p.∇θV(x) - E_∇θV)
    hh_qoi = GibbsQoI(h=hh, p=qoim.p)
    return Expectation(θ, hh_qoi; kwargs)

end



function GradExpectation(θ, qoi::GeneralQoI; kwargs...)
    try # h function of (x,θ)
        ∇θh(x) = ForwardDiff.gradient(θ -> qoi.h(x,θ), θ)
        qoim = GeneralQoI(h = x -> qoi.h(x, θ), p=qoi.p)
    catch # h function of x only
        ∇θh(x) = 0
        qoim = GeneralQoI(h = qoi.h, p=qoi.p)
    end

    # compute inner expectation E_p[∇θV]
    E_qoi = GeneralQoI(h=qoim.p.∇θV, p=qoim.p)
    E_∇θV = Expectation(θ, E_qoi; kwargs)

    # compute outer expectation
    hh(x) = ∇θh(x) + qoim.p.β * qoim.h(x) * (qoim.p.∇θV(x) - E_∇θV)
    hh_qoi = GeneralQoI(h=hh, p=qoim.p)
    return Expectation(θ, hh_qoi; kwargs)

end