### define Gibbs distribution in Distribution.jl
import Distributions: pdf, logpdf, gradlogpdf, sampler, insupport
import Base: minimum, maximum, rand, @kwdef
import StatsBase: params


@kwdef mutable struct Gibbs <: Distribution{Univariate,Continuous}
    V :: Function                                           # potential function with inputs (x,θ)
    ∇xV :: Function                                         # gradient of potential wrt x
    ∇θV :: Function                                         # gradient of potential wrt θ
    β :: Union{Real, Nothing} = nothing                     # diffusion/temperature constant
    θ :: Union{Real, Vector{<:Real}, Nothing} = nothing     # potential function parameters


    # inner constructor 1: all arguments defined 
    function Gibbs(V::Function, ∇xV::Function, ∇θV::Function, β::Real, θ::Union{Real, Vector{<:Real}}, check_args=true)
        # check arguments 
        check_args && Distributions.@check_args(Gibbs, β > 0)

        # redefine functions taking θ as argument 
        d = new()
        d.V = x -> V(x, θ)
        d.∇xV = x -> ∇xV(x, θ)
        d.∇θV = x -> ∇θV(x, θ)
        d.β = β
        d.θ = θ
        return d

    end

    # inner constructor 2: defines all parameters except θ
    function Gibbs(V::Function, ∇xV::Function, ∇θV::Function, β::Real, θ::Nothing)
        return new(V, ∇xV, ∇θV, β, θ)

    end

    # inner constructor 3: defines functions only
    function Gibbs(V::Function, ∇xV::Function, ∇θV::Function, β::Nothing, θ::Nothing)
        return new(V, ∇xV, ∇θV, β, θ)

    end

end

# outer constructor: creates new Gibbs object with defined parameters
function Gibbs(d::Gibbs; β::Real=d.β, θ::Union{Real, Vector{<:Real}, Nothing}=nothing)
    return Gibbs(V=d.V, ∇xV=d.∇xV, ∇θV=d.∇θV, β=β, θ=θ)
end


# 0 - helper function
params(d::Gibbs) = (d.β, d.θ)

# 1 - random sampler
rand(d::Gibbs, n::Int, sampler::Sampler, ρ0::Distribution) = sample(d.V, d.∇xV, sampler, n, rand(ρ0))

# 2 - unnormalized pdf
updf(d::Gibbs, x) = exp(d.β * d.V(x))

# 3 - normalization constant (partition function)
# function normconst(d::Gibbs, xdomain::Matrix{<:Real}; nquad=:false)
#     xdim = size(xdomain, 1)

#     if nquad != false | xdim <= 3

#         for k = 1:xdim
#             ξk, wk = gausslegendre(nquad, xdomain[k,1], xdomain[k,2])

#         xdim <= 3 # use Gauss quadrature integration
#         ξ, w = gausslegendre(nquad, xdomain[1], xdomain[2])
#         for j = 1:xdim


#     elseif xdim # use importance sampling integration
        
#         nquad == 0 # determine nquad 
#         nquad = length(d.θ)


#     end
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
