### define Gibbs distribution in Distribution.jl
import Distributions: pdf, logpdf, gradlogpdf, sampler, insupport
import Base: minimum, maximum, rand
import StatsBase: params

"""
mutable struct Gibbs{T<:Real} <: Distribution{Univariate,Continuous}
    V :: Function                                   # potential function with inputs (x,θ)
    ∇xV :: Function                                 # gradient of potential wrt x
    ∇θV :: Function                                 # gradient of potential wrt θ
    β :: T                                          # diffusion/temperature constant
    θ :: Union{T, Vector{T}, UndefInitializer}      # potential function parameters
        
Creates an instance of the Gibbs distribution, proportional to exp(V(x; β, θ)).
It is required to instantiate Gibbs twice: first with the potential function and its derivatives, second with parameters of the distribution.
"""
mutable struct Gibbs{T<:Real} <: Distribution{Univariate,Continuous}
    V :: Function                                   # potential function with inputs (x,θ)
    ∇xV :: Function                                 # gradient of potential wrt x
    ∇θV :: Function                                 # gradient of potential wrt θ
    β :: Union{T, UndefInitializer}                 # diffusion/temperature constant
    θ :: Union{T, Vector{T}, UndefInitializer}      # potential function parameters


    # first inner constructor: instantiates new Gibbs object with undefined parameters
    function Gibbs{T}(V::Function, ∇xV::Function, ∇θV::Function) where {T<:Real}
        # create new instance
        return new(V, ∇xV, ∇θV, undef, undef)

    end

    # second inner constructor: defines parameters of Gibbs object
    function Gibbs{T}(d::Gibbs{T}, β::T, θ::Union{T, Vector{T}}; check_args=true) where {T<:Real}
        # check arguments 
        check_args && Distributions.@check_args(Gibbs, β > 0)

        # create new instance
        V = x -> d.V(x, θ)
        ∇xV = x -> d.∇xV(x, θ)
        ∇θV = x -> d.∇θV(x, θ)
        d̃ = new(V, ∇xV, ∇θV, β, θ)
        return d̃
    end

end


# outer constructors: supply param type
Gibbs(V::Function, ∇xV::Function, ∇θV::Function) = Gibbs{Float64}(V, ∇xV, ∇θV)
Gibbs(d::Gibbs{Float64}, β::Float64, θ::Union{Float64, Vector{Float64}}; check_args=true) = Gibbs{Float64}(d, β, θ, check_args=check_args)

# convert all params to type Float64
Gibbs(d::Gibbs{Float64}, β::Real, θ::Real) = Gibbs(d, float(β), float(θ))
Gibbs(d::Gibbs{Float64}, β::Real, θ::Vector{<:Real}) = Gibbs(d, float(β), float.(θ))


## extend functions
# 0 - helper function
params(d::Gibbs) = (d.β, d.θ)

# 1 - random sampler
rand(d::Gibbs, n::Int, mcmc::Sampler, x0) = sample(d, mcmc, x0=x0, nsamp=n)

# 2 - unnormalized pdf
updf(d::Gibbs, x) = exp(d.β * d.V(x))

# 3 - normalization constant (partition function)
# function normconst(d::Gibbs; nquad=false)
#     ...
# end

# 4 - pdf
pdf(d::Gibbs, x) = exp(d.β * d.V(x)) ./ normconst(d)

# 5 - log unnormalized pdf
logupdf(d::Gibbs, x) = d.β * d.V(x)

# 6 - logpdf
logpdf(d::Gibbs, x) = d.β * d.V(x) - log(normconst(d))

# 7 - gradlogpdf (wrt x)
gradlogpdf(d::Gibbs, x::Real) = d.β * d.∇xV(x)
gradlogpdf(d::Gibbs, x::Vector) = d.β * d.∇xV(x)

# 8 - minimum
minimum(d::Gibbs) = -Inf

# 9 - maximum
maximum(d::Gibbs) = Inf
