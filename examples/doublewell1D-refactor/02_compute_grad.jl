## 02 - script for compute gradients of the QoI

# specify sample size and number of replications ------------------------------------------
nsamp = 10 # 10000               # sample size
nrepl = 1 # 5                   # number of replications per sample size
ncomp = 5                   # number of components in mixture biasing distribution


# specify sample size and number of replications ------------------------------------------

for j = 1:nrepl

    # create directory
    path = "data/repl$j"
    try
        run(`mkdir -p $path`)
    catch
    end

    println("================= REPLICATION $j =================")
    
    # sample parameters
    θsamp = [rand(ρθ) for i = 1:nsamp]  
    JLD.save("$(path)/parameters_nsamp=$(nsamp).jld", "θsamp", θsamp)


    # reference: quadrature estimate -----------------------------------------------------
    # QuadIntegrator
    ngrid = 200                                     # number of θ quadrature points in 1D
    ξx, wx = gausslegendre(ngrid, -9, 11)           # 1D quad points over x
    GQint = GaussQuadrature(ξx, wx)
    
    ∇Qgq = compute_gradvar(θsamp, q, GQint) 
    JLD.save("$(path)/gradQ_quad_nsamp=$(nsamp).jld", "∇Q", ∇Qgq)


    # Monte Carlo estimate ---------------------------------------------------------------
    # MCIntegrator
    nMC = 10000                                     # number of MC/MCMC samples
    nuts = NUTS(0.02)                               # Sampler struct
    MCint = MCMC(nMC, nuts, ρx0)
    
    t = @elapsed ∇Qmc = compute_gradvar(θsamp, q, MCint) 
    println("Vanilla MC: $t sec.")
    JLD.save("$(path)/gradQ_MC_nsamp=$(nsamp).jld", "∇Q", ∇Qmc)


    # importance sampling - mixture Gibbs ------------------------------------------------
    λsamp = reduce(vcat, ([ρθ.μ], [rand(ρθ) for i = 1:ncomp-1]))
    βsamp = reduce(vcat, (0.2, 0.8*ones(ncomp-1)) )
    comps = [Gibbs(πgibbs0, β=βi, θ=λi) for (βi,λi) in zip(βsamp, λsamp)]
    mm = MixtureModel(comps)

    # compute "high fidelity" estimate for components (by Monte Carlo)
    ∇Qhf = compute_gradvar(λsamp, q, MCint) 
    JLD.save("$(path)/gradQ_MC_HF_nsamp=$(ncomp).jld", "∇Q", ∇Qhf)

    # sample from mixture
    xsamp = rand(mm, nMC, nuts, ρx0; burn=2000)
    ISint = ISSamples(mm, xsamp, GQint)

    # compute
    t = @elapsed ∇Qis_u, metrics_u = compute_gradvar(θsamp, q, ISint)
    println("IS MC Mixture (unadapted): $t sec.")

    JLD.save("$(path)/gradQ_ISM_nsamp=$(nsamp).jld",
        "∇Q", ∇Qis_u,
        "metrics", metrics_u,
        "λ", λsamp,
        "β", βsamp,
        )

    
    # importance sampling - mixture Gibbs (adapted centers) ------------------------------
    λset = adapt_mixture_biasing_dist(λsamp, qgrad, πgibbs1, nuts, ρx0; niter=3)
    comps = [Gibbs(πgibbs0, β=βi, θ=λi) for (βi,λi) in zip(βsamp, λset[end])]
    mm = MixtureModel(comps)

    # sample from mixture
    xsamp = rand(mm, nMC, nuts, ρx0; burn=2000)
    ISint = ISSamples(mm, xsamp, GQint)

    # compute
    t = @elapsed ∇Qis_a, metrics_a = compute_gradvar(θsamp, q, ISint; gradh=∇h)
    println("IS MC Mixture (adapted): $t sec.")

    JLD.save("$(path)/gradQ_ISM_nsamp=$(nsamp).jld",
        "∇Q", ∇Qis_a,
        "metrics", metrics_a,
        "λ", λset[end],
        "β", βsamp,
        )


    # importance sampling - mixture Gibbs (adapted centers and weights) -----------------




    

    


