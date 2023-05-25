### potential function and gradients of 1D-state 2D-parameter double well model
using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD

include("model_param1.jl")
include("qoi_meanenergy.jl")
include("mc_utils.jl")



# define integrators #########################################################################################

# QuadIntegrator
ngrid = 20                                      # number of θ quadrature points in 1D
ξθ, wθ = gausslegendre(ngrid, θrng[1], θrng[2]) # 2D quad points over θ
ξx, wx = gausslegendre(200, -10, 10)            # 1D quad points over x
GQint = GaussQuadrature(ξx, wx)

# MCIntegrator
nMC = 10000                                     # number of MC/MCMC samples
nuts = NUTS(1e-1)                               # Sampler struct
MCint = MCMC(nMC, nuts, ρx0)

# ISIntegrator
πu = Uniform(-5,5)                              # importance sampling: uniform distribution
βarr = [0.01, 0.2, 1.0]                         # importance sampling: temp parameter for Gibbs biasing dist.
nβ = length(βarr)

# θ samples for MC integration 
nsamp_arr = [10, 50, 100]
nrepl = 1                                       # number of replications of sampling



# compute reference covariance matrix ######################################################################
Cref = compute_covmatrix_ref((ξθ, wθ), q, ρθ, GQint)



# run MC trials #################################################################################
for j = 1:nrepl
    println("================= REPLICATION nrepl = $j =================")

    # sample parameters
    nsamptot = nsamp_arr[end] # largest number of samples
    θsamp = [rand(ρθ) for i = 1:nsamptot]   

   
    # MC estimate ------------------------------------------------------------------------
    t = @elapsed ∇Qmc = compute_gradQ(θsamp, q, MCint; gradh=∇h) 
    CMC = compute_covmatrix(∇Qmc, nsamp_arr)
    println("Vanilla MC: $t sec.")


    # importance sampling - uniform ------------------------------------------------------
    xsamp = rand(πu, nMC)
    ISint = ISSamples(πu, xsamp)
    t = @elapsed ∇Qis_u, metrics_u = compute_gradQ(θsamp, q, ISint; gradh=∇h)
    CIS_u = compute_covmatrix(∇Qis_u, nsamp_arr)
    println("IS MC Uniform: $t sec.")


    # importance sampling - Gibbs -------------------------------------------------------
    # initialize
    CIS_g = Dict{Float64, Dict}()
    metrics_g = Dict{Float64, Dict}()
    
    # iterate over β
    for βi in βarr
        πg = Gibbs(πgibbs0, β=βi, θ=[3,3])
        xsamp = rand(πg, nMC, nuts, ρx0) 
        ISint = ISSamples(πg, xsamp)
        t = @elapsed ∇Qis_gi, metrics_gi = compute_gradQ(θsamp, q, ISint; gradh=∇h)
        CIS_g[βi] = compute_covmatrix(∇Qis_gi, nsamp_arr)
        metrics_g[βi] = metrics_gi
        println("IS MC Gibbs (β=$βi): $t sec.")
    end


    # save data ###############################################################################################

    JLD.save("data1/DW1D_MC_nsamp=$(nsamptot)_repl=$j.jld",
        "nx", nMC,
        "C", CMC,
    )

    JLD.save("data1/DW1D_ISU_nsamp=$(nsamptot)_repl=$j.jld",
        "nx", nMC,
        "π_bias", πu,
        "C", CIS_u,
        "metrics", metrics_u,
    )

    JLD.save("data1/DW1D_ISG_nsamp=$(nsamptot)_repl=$j.jld",
        "nx", nMC,
        "βarr", βarr,
        "C", CIS_g,
        "metrics", metrics_g,
    )

end

# JLD.save("data/DW1D_Ref.jld",
#         "ngrid", ngrid,
#         "Cref", Cref,
#     )