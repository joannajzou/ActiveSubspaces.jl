### define Gibbs distribution in Distribution.jl
import Distributions: pdf, logpdf, gradlogpdf, sampler, insupport
import Base: minimum, maximum, rand, @kwdef
import StatsBase: params


@kwdef mutable struct Gibbs <: Distribution{Union{Univariate,Multivariate}, Continuous}
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
function rand(d::Gibbs, n::Int, sampler::Sampler, ρ0::Distribution; burn=1000)
    nsim = Int(burn + n)
    lπ = x -> logupdf(d, x)
    gradlπ = x -> gradlogpdf(d, x)
    xsamp = sample(lπ, gradlπ, sampler, nsim, rand(ρ0))
    return xsamp[(burn+1):end]
end

function rand(d::Gibbs, n::Int, sampler::Sampler, x0::Vector; burn=1000)
    nsim = Int(burn + n)
    lπ = x -> logupdf(d, x)
    gradlπ = x -> gradlogpdf(d, x)
    xsamp = sample(lπ, gradlπ, sampler, nsim, x0)
    return xsamp[(burn+1):end]
end


# 2 - pdf
updf(d::Gibbs, x) = exp(-d.β * d.V(x))
pdf(d::Gibbs, x, normint::Integrator) = updf(d, x) ./ normconst(d, normint)

# 3 - normalization constant (partition function)
normconst(d::Gibbs, normint::QuadIntegrator) = sum(normint.w .* updf.((d,), normint.ξ))

# 4 - log unnormalized pdf
logupdf(d::Gibbs, x) = -d.β * d.V(x)
logpdf(d::Gibbs, x, normint::Integrator) = logupdf(d, x) - log(normconst(d, normint))

# 5 - gradlogpdf (wrt x)
gradlogpdf(d::Gibbs, x) = -d.β * d.∇xV(x)

# 6 - minimum
minimum(d::Gibbs) = -Inf

# 7 - maximum
maximum(d::Gibbs) = Inf
