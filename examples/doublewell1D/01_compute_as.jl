### potential function and gradients of 1D-state 2D-parameter double well model
using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD

# select model
modnum = 1
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("mc_utils.jl")


# specify sample sizes and number of replications for simulation ##################################
nsamp_arr = [1000, 2000, 4000, 8000, 16000]     # sample sizes
nrepl = 1                                       # number of replications per sample size
        


# compute reference covariance matrix ############################################################
Cref = compute_covmatrix_ref((ξθ, wθ), q, ρθ, GQint)

JLD.save("data$modnum/DW1D_Ref.jld",
        "ngrid", ngrid,
        "Cref", Cref,
    )


# run MC trials ##################################################################################
try
    path = "data$modnum/repl$j"
    run(`mkdir -p $path`)
catch
end

for j = 1:nrepl
    println("================= REPLICATION nrepl = $j =================")

    # sample parameters
    nsamptot = nsamp_arr[end] # largest number of samples
    θsamp = [rand(ρθ) for i = 1:nsamptot]   
    θsamp = remove_outliers(θsamp)
    nsamp_arr[end] = length(θsamp)

    # MC estimate ------------------------------------------------------------------------
    t = @elapsed ∇Qmc = compute_gradQ(θsamp, q, MCint; gradh=∇h) 
    CMC = compute_covmatrix(∇Qmc, nsamp_arr)
    println("Vanilla MC: $t sec.")

    JLD.save("data$modnum/repl$j/DW1D_MC_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "C", CMC,
    )


    # importance sampling - uniform ------------------------------------------------------
    xsamp = rand(πu, nMC)
    ISint = ISSamples(πu, xsamp)
    t = @elapsed ∇Qis_u, metrics_u = compute_gradQ(θsamp, q, ISint; gradh=∇h)
    CIS_u = compute_covmatrix(∇Qis_u, nsamp_arr)
    println("IS MC Uniform: $t sec.")

    JLD.save("data$modnum/repl$j/DW1D_ISU_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "π_bias", πu,
        "C", CIS_u,
        "metrics", metrics_u,
    )


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


    JLD.save("data$modnum/repl$j/DW1D_ISG_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "βarr", βarr,
        "C", CIS_g,
        "metrics", metrics_g,
    )

end

