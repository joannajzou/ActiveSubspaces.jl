"""

ActiveSubspaces

Author: Joanna Zou <jjzou@mit.edu>
Version: 0.1.0
Year: 2023
Notes: A Julia library for implementing active subspaces for QoI-based dimension reduction in stochastic systems characterized by an invariant measure.

"""
module ActiveSubspaces

# load packages
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Distributions
using FastGaussQuadrature
using ForwardDiff
using AdvancedHMC
using JLD


# call on src scripts
include("Sampling/sampling.jl")

include("Distributions/distributions.jl")

include("Integrators/integrators.jl")

include("QoI/qoi.jl")

include("Subspaces/subspaces.jl")

include("Diagnostics/diagnostics.jl")




end
