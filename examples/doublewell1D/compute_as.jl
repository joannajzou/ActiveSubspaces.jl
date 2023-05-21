### potential function and gradients of 1D-state 2D-parameter double well model
using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD

include("model_param1.jl")
include("qoi_meanenergy.jl")


function init_IS_arrays(nsamp)
    ∇Q_arr = Vector{Vector{Float64}}(undef, nsamp)
    ∇h_arr = Vector{Vector}(undef, nsamp)
    w_arr = Vector{Vector{Float64}}(undef, nsamp)
    return ∇Q_arr, ∇h_arr, w_arr
end


function compute_covmatrix_IS(q::GibbsQoI, d::Distribution, xsamp::Vector{Float64}, θsamp::Vector{Vector{Float64}}; gradh::Function=nothing)
    nMC = length(xsamp)
    nsamp = length(θsamp)
    ∇Qis, his, wis = init_IS_arrays(nsamp)
    integrator = ISSamples(d, xsamp)
    for k = 1:nsamp
        ∇Qis[k], his[k], wis[k] = grad_expectation(θsamp[k], q, integrator, gradh=gradh)
    end

    w̃is = [[his[i][j] * wis[i][j] for j = 1:nMC] for i = 1:nsamp]
    Cis = compute_covmatrix(∇Qis)
    wvar = [ISWeightVariance(wi) for wi in wis]
    wESS = [ISWeightESS(wi)/nMC for wi in wis]
    wdiag = [ISWeightDiagnostic(wi) for wi in wis]
    w̃ESS = [ISWeightESS(wi)/nMC for wi in w̃is]
    return Cis, wvar, wESS, wdiag, w̃ESS
end


# define integrators #########################################################################################

# QuadIntegrator
ngrid = 200                  # number of θ quadrature points in 1D
ξθ, wθ = gausslegendre(ngrid, θrng[1], θrng[2]) # 2D quad points over θ
ξx, wx = gausslegendre(200, -10, 10) # 1D quad points over x
GQint = GaussQuadrature(ξx, wx)

# MCIntegrator
nMC = 10000                 # number of MC/MCMC samples
nuts = NUTS(1e-1)           # Sampler struct
MCint = MCMC(nMC, nuts, ρx0)

# ISIntegrator
πu = Uniform(-5,5)          # importance sampling: uniform distribution
βarr = [0.01, 0.2, 1.0]     # importance sampling: temp parameter for Gibbs biasing dist.
nβ = length(βarr)

# θ samples for MC integration 
nsamp_arr = [1000, 2000, 4000, 8000] # 1500 
nrepl = 1                   # number of replications of sampling



# compute reference covariance matrix ######################################################################

# Cref = zeros(2,2)
# for i = 1:ngrid
#     for j = 1:ngrid
#         println("($i, $j)")
#         ξij = [ξθ[i], ξθ[j]]
#         ∇Qij = grad_expectation(ξij, q, GQint)
#         Cref .+= ∇Qij*∇Qij' * pdf(ρθ, ξij) * wθ[i] * wθ[j]
#     end
# end



# run MC trials #################################################################################

for n = 1:length(nsamp_arr)
    nsamp = nsamp_arr[n]
    println("================= NSAMP = $nsamp =================")

    # initialize
    CMC = Vector{Matrix{Float64}}(undef, nrepl)
    CIS_u = Vector{Matrix{Float64}}(undef, nrepl)
    metrics_u = init_metrics_dict(nrepl)
    CIS_g = Dict{Float64, Vector{Matrix{Float64}}}()
    metrics_g = Dict{Float64, Dict}()
    for i = 1:nβ
        βi = βarr[i]
        CIS_g[βi] = Vector{Matrix{Float64}}(undef, nrepl)
        metrics_g[βi] = init_metrics_dict(nrepl)
    end


    # iterate over replications
    for j = 1:nrepl
        println("-------- iteration nrepl = $j --------")

        # sample parameters
        θsamp = [rand(ρθ) for i = 1:nsamp]         # draw independent samples from prior density


        # MC estimate ------------------------------------------------------------------------
        t = @elapsed begin
            ∇Qmc = map(θ -> grad_expectation(θ, q, MCint, gradh=∇h), θsamp)
            CMC[j] = compute_covmatrix(∇Qmc)
        end
        println("Vanilla MC: $t sec.")


        # importance sampling - uniform ------------------------------------------------------
        xsamp = rand(πu, nMC)
        t = @elapsed begin
            CIS_u[j], metrics_u["wvar"][j], metrics_u["wESS"][j], metrics_u["wdiag"][j], metrics_u["w̃ESS"][j] = compute_covmatrix_IS(q, πu, xsamp, θsamp, gradh=∇h)
        end
        metrics_u["θsamp"][j] = θsamp
        println("IS MC Uniform: $t sec.")


        # importance sampling - Gibbs -------------------------------------------------------
        for i = 1:nβ
            βi = βarr[i]    
            ∇Qis_i, his_i, wis_i = init_IS_arrays(nsamp)
            πg = Gibbs(πgibbs0, β=βi, θ=[3,3])
            xsamp = rand(πg, nMC, nuts, ρx0) 
            t = @elapsed begin
                CIS_g[βi][j], metrics_g[βi]["wvar"][j], metrics_g[βi]["wESS"][j], metrics_g[βi]["wdiag"][j], metrics_g[βi]["w̃ESS"][j] = compute_covmatrix_IS(q, πg, xsamp, θsamp, gradh=∇h)
            end
            metrics_g[βi]["θsamp"][j] = θsamp
            println("IS MC Gibbs (β=$βi): $t sec.")
        end

    end


    # save data ###############################################################################################

    JLD.save("data/DW1D_MC_2_nsamp={$nsamp}_nrepl={$nrepl}.jld",
        "n_θsamp", nsamp,
        "n_xsamp", nMC,
        "C", CMC,
    )

    JLD.save("data/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld",
        "n_θsamp", nsamp,
        "n_xsamp", nMC,
        "π_bias", πu,
        "C", CIS_u,
        "metrics", metrics_u,
    )

    JLD.save("data/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld",
        "n_θsamp", nsamp,
        "n_xsamp", nMC,
        "βarr", βarr,
        "C", CIS_g,
        "metrics", metrics_g,
    )

end

# JLD.save("data/DW1D_Ref.jld",
#         "ngrid", ngrid,
#         "Cref", Cref,
#     )