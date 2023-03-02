"""

ActiveSubspaces

Author: Joanna Zou <jjzou@mit.edu>
Version: 0.1.0
Year: 2023
Notes: A Julia library for implementing active subspaces for QoI-based dimension reduction in stochastic systems characterized by an invariant measure.

"""
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
include("qoi.jl")


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
export GibbsQoI, GeneralQoI, Expectation, GradExpectation

end
