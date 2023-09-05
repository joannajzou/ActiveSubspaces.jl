# load packages
using ActiveSubspaces
using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using JuLIP, ACE1, ACE1pack
using Distributions
using JLD

# define ACE basis
nbody = 5
deg = 4
simdir = "ACE_MD_$(nbody)body_$(deg)deg/"

ace = ACE(species = [:Hf, :O],          # species
          body_order = nbody,           # n-body
          polynomial_degree = deg,      # degree of polynomials
          wL = 1.0,                     # Defaults, See ACE.jl documentation 
          csp = 1.0,                    # Defaults, See ACE.jl documentation 
          r0 = 1.0,                     # minimum distance between atoms
          rcutoff = 5.0)                # cutoff radius 