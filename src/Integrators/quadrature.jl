abstract type QuadIntegrator <: Integrator end


"""
struct GaussQuadrature <: QuadIntegrator

Contains parameters for 1D or 2D Gauss quadrature integration.

# Arguments
- `ξ :: Union{Vector{<:Real}, Vector{<:Vector}}`  : quadrature points
- `w :: Union{Vector{<:Real}, Vector{<:Vector}}`  : quadrature weights
- `d :: Real`                                     : dimension
"""
struct GaussQuadrature <: QuadIntegrator
    ξ :: Union{Vector{<:Real}, Vector{<:Vector}, Vector} 
    w :: Vector{<:Real}
end

# simple grid integration
GaussQuadrature(ξ::Union{Vector{<:Real}}) = GaussQuadrature(ξ, ones(length(ξ)))
GaussQuadrature(ξ::Union{Vector{<:Vector}}) = GaussQuadrature(ξ, ones(length(ξ[1])))


"""
function gaussquad(nquad::Integer, gaussbasis::Function, args...; limits::Vector=[-1,1])

Computes 1D Gauss quadrature points with a change of domain to limits=[ll, ul].

# Arguments
- `nquad::Integer`          : number of quadrature points
- `gaussbasis::Function`    : polynomial basis function from FastGaussQuadrature (gausslegendre, gaussjacobi, etc.)
- `args...`                 : additional arguments for the gaussbasis() function
- `limits::Vector`          : defines lower and upper limits for rescaling

# Outputs
- `ξ :: Vector{<:Real}`     : quadrature points ranging [ll, ul]
- `w :: Vector{<:Real}`     : quadrature weights
"""
function gaussquad(nquad::Integer, gaussbasis::Function, args...; limits=[-1,1])
    ξ, w = gaussbasis(nquad, args...)
    ξ, w = rescale(ξ, w, limits)
    return ξ, w
end


"""
function gaussquad_2D(nquad::Integer, gaussbasis::Function, args...; limits::Vector=[-1,1])

Computes 2D (symmetric) Gauss quadrature points with a change of domain to limits=[ll, ul].

# Arguments
- `nquad::Integer`          : number of quadrature points
- `gaussbasis::Function`    : polynomial basis function from FastGaussQuadrature (gausslegendre, gaussjacobi, etc.)
- `args...`                 : additional arguments for the gaussbasis() function
- `limits::Vector`          : defines lower and upper limits for rescaling

# Outputs
- `ξ :: Vector{<:Real}`     : quadrature points ranging [ll, ul]
- `w :: Vector{<:Real}`     : quadrature weights
"""
function gaussquad_2D(nquad::Integer, gaussbasis::Function, args...; limits=[[-1,1],[-1,1]])
    ξ, w = gaussbasis(nquad, args...)
    ξ1, w1 = rescale(ξ, w, limits[1])
    ξ2, w2 = rescale(ξ, w, limits[2])
    ξz = [[ξ1[i], ξ2[j]] for i = 1:nquad, j = 1:nquad][:]
    wz = [w1[i] * w2[j] for i = 1:nquad, j = 1:nquad][:]
    return ξz, wz
end


# change of interval from [-1,1] to [ll,ul]
function rescale(ξ, w, limits::Vector{<:Real})
    ll, ul = limits # lower and upper limits
    ξz = (ul-ll) .* ξ / 2 .+ (ll+ul) / 2
    wz = (ul-ll)/2 * w
    return ξz, wz
end

