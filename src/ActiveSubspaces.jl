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
using StatsFuns
using Distributions
using FastGaussQuadrature
using ForwardDiff
using AdvancedHMC
using PotentialLearning
using InvertedIndices
# using JLD


# functions for standardizing input variables 
include("DataProcessing/dataprocessing.jl")

# functions for MCMC sampling from distributions
include("Sampling/sampling.jl")

# functions for defining integration method and parameters
include("Integrators/integrators.jl")

# functions for defining generic distributions
include("Distributions/distributions.jl")

# functions for computing expectations (QoIs)
include("QoI/qoi.jl")

# functions for adaptive importance sampling
include("AdaptiveIS/adaptiveis.jl")

# functions for computing the active subspace
include("DimensionReduction/dimensionreduction.jl")

# functions for computing error/diagnostic metrics
include("Diagnostics/diagnostics.jl")



end
