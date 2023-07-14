abstract type QuadIntegrator <: Integrator end

"""
struct GridIntegration <: QuadIntegrator

Contains parameters for 1D Gauss quadrature integration.

# Arguments
- `x :: Vector{<:Real}`       : x points at which to evaluate the pdf
"""
struct GridIntegration <: QuadIntegrator
    x :: Vector{<:Real}
end

"""
struct GaussQuadrature <: QuadIntegrator

Contains parameters for 1D Gauss quadrature integration.

# Arguments
- `ξ :: Vector{<:Real}`       : quadrature points
- `w :: Vector{<:Real}`       : quadrature weights
"""
struct GaussQuadrature <: QuadIntegrator
    ξ :: Vector{<:Real}
    w :: Vector{<:Real}
end


# computes Gauss Legendre quadrature points with change of domain
import FastGaussQuadrature: gausslegendre

"""
function gausslegendre(npts::Int64, ll::Real, ul::Real)

Computes 1D Gauss-Legendre quadrature points with a change of domain to [ll, ul].

# Arguments
- `npts :: Int64`           : number of quadrature points
- `ll :: Real`              : lower limit
- `ul :: Real`              : upper limit

# Outputs
- `ξ :: Vector{<:Real}`     : quadrature points ranging [ll, ul]
- `w :: Vector{<:Real}`     : quadrature weights
"""
function gausslegendre(npts::Int64, ll::Real, ul::Real)
    ξ, w = gausslegendre(npts) 
    ξz = (ul-ll) .* ξ / 2 .+ (ll+ul) / 2 # change of interval
    wz = (ul-ll)/2 * w
    return ξz, wz
end




