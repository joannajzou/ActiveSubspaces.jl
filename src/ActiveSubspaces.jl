module ActiveSubspaces

# put all packages here
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Distributions
using FastGaussQuadrature
using ForwardDiff
using AdvancedHMC
using JLD

import FastGaussQuadrature: gausslegendre


# call on all src scripts here
include("examplefile.jl")
include("sampling.jl")
include("gibbs.jl")
# include("qoi.jl")


# computes Gauss Legendre quadrature points with change of domain
function gausslegendre(npts, ll, ul)
    ξ, w = gausslegendre(npts) 
    ξz = (ul-ll) .* ξ / 2 .+ (ll+ul) / 2 # change of interval
    wz = (ul-ll)/2 * w
    return ξz, wz
end


# put all exports here
export f, dfx, gausslegendre
export Gibbs, Gibbs!, params, updf, pdf, logupdf, logpdf, gradlogpdf
export HMC, NUTS, sample

end
