module ActiveSubspaces

# put all packages here
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Distributions
using FastGaussQuadrature
using ForwardDiff
using JLD


# call on all src scripts here

include("examplefile.jl")
# include("sampling.jl")
include("gibbs.jl")
# include("qoi.jl")


# put all exports here
export f, dfx
export Gibbs, params, updf, pdf, logupdf, logpdf, gradlogpdf

end
