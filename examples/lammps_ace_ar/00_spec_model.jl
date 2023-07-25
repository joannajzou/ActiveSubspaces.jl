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
nbody = 2
deg = 8
simdir = "ACE_MD_$(nbody)body_$(deg)deg_2/"

ace = ACE(species = [:Ar],         # species
          body_order = nbody,      # 2-body
          polynomial_degree = deg, # 8 degree polynomials
          wL = 1.0,                # Defaults, See ACE.jl documentation 
          csp = 1.0,               # Defaults, See ACE.jl documentation 
          r0 = 1.0,                # minimum distance between atoms
          rcutoff = 4.0)           # cutoff radius 