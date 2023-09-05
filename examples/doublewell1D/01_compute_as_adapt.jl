using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using PotentialLearning


# select model
modnum = 4
include("model_param$(modnum).jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("mc_utils.jl")
include("adaptive_is_utils.jl")

# specify sample sizes and number of replications for simulation ##################################
nsamp_arr = [625, 1250, 2500, 5000] # 4000, 8000]            # sample sizes
nrepl = 5                                      # number of replications per sample size
        
println(nsamp_arr)
println("nrepl = $nrepl")


# compute reference covariance matrix ############################################################
# Cref = compute_covmatrix_ref((ξθ, wθ), q, ρθ, GQint)

# JLD.save("data$modnum/DW1D_Ref.jld",
#         "ngrid", ngrid,
#         "Cref", Cref,
#     )


# run MC trials ##################################################################################

for j = 1:nrepl

    try
        path = "data$modnum/repl$j"
        run(`mkdir -p $path`)
    catch
    end

    println("================= REPLICATION nrepl = $j =================")

    # sample parameters
    nsamptot = nsamp_arr[end] # largest number of samples

    θsamp = [rand(ρθ) for i = 1:nsamptot]   

    # or load
    # θsamp = JLD.load("data$modnum/repl$j/DW1D_ISM_nsamp=$nsamptot.jld")["metrics"]["θsamp"]

    # reference gradient: by quadrature --------------------------------------------------
    
    ∇Qgq = compute_gradQ(θsamp, q, GQint; gradh=∇h) 
    JLD.save("data$modnum/repl$j/DW1D_Quad_grad_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "∇Q", ∇Qgq,
    )

    # MC estimate ------------------------------------------------------------------------
    
    # compute
    t = @elapsed ∇Qmc = compute_gradQ(θsamp, q, MCint; gradh=∇h) 
    CMC = compute_covmatrix(∇Qmc, nsamp_arr)
    println("Vanilla MC: $t sec.")

    # save
    JLD.save("data$modnum/repl$j/DW1D_MC_cov_nsamp=$(nsamptot).jld",
        "θsamp", θsamp,
        "C", CMC,
    )

    JLD.save("data$modnum/repl$j/DW1D_MC_grad_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "∇Q", ∇Qmc,
    )


    # importance sampling - Gibbs -------------------------------------------------------
    
    # # initialize
    # πg = Gibbs(πgibbs2, θ=ρθ.μ)
    # xsamp = rand(πg, nMC, nuts, ρx0; burn=2000) 
    # ISint = ISSamples(πg, xsamp)

    # # # compute
    # t = @elapsed ∇Qis_g, metrics_g = compute_gradQ(θsamp, q, ISint; gradh=∇h)
    # CIS_g = compute_covmatrix(∇Qis_g, nsamp_arr)
    # println("IS MC Gibbs: $t sec.")

    # # save
    # JLD.save("data$modnum/repl$j/DW1D_ISG_nsamp=$(nsamptot).jld",
    #     "nx", nMC,
    #     "∇Q", ∇Qis_g,
    #     "C", CIS_g,
    #     "metrics", metrics_g,
    # )


    # importance sampling - mixture Gibbs ----------------------------------------------

    # select locations of proposal PDFs
    nλ = 20
    λsamp = reduce(vcat, ([ρθ.μ], [rand(ρθ) for i = 1:nλ-1]))
    
    ### unadapted ###
    mm = MixtureModel(λsamp, πgibbs2)

    # sample from proposal PDF
    xsamp = rand(mm, nMC, nuts, ρx0; burn=2000)
    # plot_pdf_with_sample_hist(mm, xplot, xsamp)
    # plot_mcmc_trace(Matrix(xsamp[:,:]'))
    ISint = ISSamples(mm, xsamp, GQint)

    # compute
    t = @elapsed ∇Qis_u, metrics_u = compute_gradQ(θsamp, q, ISint; gradh=∇h)
    CIS_u = compute_covmatrix(∇Qis_u, nsamp_arr)
    println("IS MC Mixture (unadapted): $t sec.")
    
    # save
    JLD.save("data$modnum/repl$j/DW1D_ISM_cov_nsamp=$(nsamptot).jld",
        "mm_cent", λsamp,
        "mm_wts", 1/nλ .* ones(nλ),
        "C", CIS_u,
        "metrics", metrics_u,
    )

    JLD.save("data$modnum/repl$j/DW1D_ISM_grad_nsamp=$(nsamptot).jld",
        "nx", nMC,    
        "∇Q", ∇Qis_u,
    )


    ### variable weights ###
    samps = [rand(Gibbs(πgibbs2, θ=λ) , nMC, nuts, ρx0; burn=2000) for λ in λsamp]
    knl = RBF(Euclidean(inv(Σθ)); ℓ=1)
    ISmix = ISMixSamples(mm, nMC, knl, samps, GQint)

    # compute
    t = @elapsed ∇Qis_w, metrics_w = compute_gradQ(θsamp, q, ISmix; gradh=∇h)
    CIS_w = compute_covmatrix(∇Qis_w, nsamp_arr)
    println("IS MC Mixture (adapt. wts.): $t sec.")

     # save
    JLD.save("data$modnum/repl$j/DW1D_ISMw_cov_nsamp=$(nsamptot).jld",
        "mm_cent", λsamp,
        "C", CIS_w,
        "metrics", metrics_w,
    )

    JLD.save("data$modnum/repl$j/DW1D_ISMw_grad_nsamp=$(nsamptot).jld",
        "nx", nMC,   
        "∇Q", ∇Qis_w,
    )
    

    ### adapted centers ###
    E_qoi = GibbsQoI(h=q.p.∇θV, p=q.p)
    E_∇θV = expectation(ρθ.μ, E_qoi, GQint)
    hgrad(x, γ) = ∇θV(x, γ) -  V(x, γ) * (∇θV(x, γ) - E_∇θV)
    qgrad = GibbsQoI(h=hgrad, p=q.p)
    λset, _ = adapt_mixture_biasing_dist(λsamp, qgrad, πgibbs2; niter=2)
    mm = MixtureModel(λset[end], πgibbs2)


    # sample from proposal PDF
    xsamp = rand(mm, nMC, nuts, ρx0; burn=2000)
    ISint = ISSamples(mm, xsamp, GQint)

    # compute
    t = @elapsed ∇Qis_a, metrics_a = compute_gradQ(θsamp, q, ISint; gradh=∇h)
    CIS_a = compute_covmatrix(∇Qis_a, nsamp_arr)
    println("IS MC Mixture (adapted): $t sec.")
    
    # save
    JLD.save("data$modnum/repl$j/DW1D_ISMa_cov_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "mm_cent", λset,
        "C", CIS_a,
        "metrics", metrics_a,
    )

    JLD.save("data$modnum/repl$j/DW1D_ISMa_grad_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "∇Q", ∇Qis_a,
    )



    ### adapted centers with variable weights ###
    samps = [rand(Gibbs(πgibbs2, θ=λ) , nMC, nuts, ρx0; burn=2000) for λ in λset[end]]
    ISmix = ISMixSamples(mm, λsamp, nMC, knl, samps, GQint)

    # compute
    t = @elapsed ∇Qis_aw, metrics_aw = compute_gradQ(θsamp, q, ISmix; gradh=∇h)
    CIS_aw = compute_covmatrix(∇Qis_aw, nsamp_arr)
    println("IS MC Mixture (adapted cent. & wt.): $t sec.")
    
    # save
    JLD.save("data$modnum/repl$j/DW1D_ISMaw_cov_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "mm_cent", λset,
        "C", CIS_aw,
        "metrics", metrics_aw,
    )

    JLD.save("data$modnum/repl$j/DW1D_ISMaw_grad_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "∇Q", ∇Qis_aw,
    )

end


