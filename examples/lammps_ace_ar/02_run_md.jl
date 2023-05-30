using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using ACE1, ACE1pack, JuLIP
using Distributions
using JLD

include("define_md_LAMMPS_ace.jl")

# load posterior distribution
πβ = JLD.load("coeff_posterior.jld")["π"]

# Define ACE basis
ace = ACE(species = [:Ar],         # species
          body_order = 2,          # 2-body
          polynomial_degree = 8,   # 8 degree polynomials
          wL = 1.0,                # Defaults, See ACE.jl documentation 
          csp = 1.0,               # Defaults, See ACE.jl documentation 
          r0 = 1.0,                # minimum distance between atoms
          rcutoff = 4.0)           # cutoff radius 


# compute energy with MD #############################################################################
nsamp = 5                # number of coefficient samples

# Run MD for different coefficients and different seeds
Tend = Int(0.5E6)       # number of steps
dT = 50                 # time step
Temp = 0.65*120         # temperature

# iterate over coefficient samples
for coeff = 18:1000 # nsamp            
    println("coeff $coeff")

    # create directory
    save_dir = "ACE_MD/coeff_$coeff/"
    run(`mkdir -p $save_dir`)

    # sample coefficients
    β = rand(πβ)
    # β = πβ.μ
    β_to_file(β, ace, save_dir)

    # for seed = 1:100
    # @time energy_minimization(save_dir; maxeval=50000)
    @time run_md(Tend, save_dir, save_dir; seed = 1, dT = dT, dt = 0.0001, Temp = Temp)
    run(`python3 to_extxyz.py ACE_MD/coeff_$(coeff)/`)
    # end
end

