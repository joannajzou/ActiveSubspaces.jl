using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD

# select model
modnum = 3
include("model_param$(modnum).jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("mc_utils.jl")


# specify sample sizes and number of replications for simulation ##################################
nsamp_arr = [1000, 2000, 4000, 8000]            # sample sizes
nrepl = 5                                       # number of replications per sample size
        


# compute reference covariance matrix ############################################################
# Cref = compute_covmatrix_ref((ξθ, wθ), q, ρθ, GQint)

# JLD.save("data$modnum/DW1D_Ref.jld",
#         "ngrid", ngrid,
#         "Cref", Cref,
#     )


# run MC trials ##################################################################################

for j = 2:nrepl

    try
        path = "data$modnum/repl$j"
        run(`mkdir -p $path`)
    catch
    end

    println("================= REPLICATION nrepl = $j =================")

    # sample parameters
    nsamptot = nsamp_arr[end] # largest number of samples

    θsamp = [rand(ρθ) for i = 1:nsamptot]   
    # θsamp = remove_outliers(θsamp)
    # nsamp_arr[end] = length(θsamp)

    # or load
    # θsamp = JLD.load("data$modnum/repl$j/DW1D_ISG_nsamp=$nsamptot.jld")["metrics"][0.5]["θsamp"]

    # MC estimate ------------------------------------------------------------------------
    t = @elapsed ∇Qmc = compute_gradQ(θsamp, q, MCint; gradh=∇h) 
    CMC = compute_covmatrix(∇Qmc, nsamp_arr)
    println("Vanilla MC: $t sec.")

    JLD.save("data$modnum/repl$j/DW1D_MC_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "C", CMC,
    )


    # importance sampling - uniform ------------------------------------------------------
    # xsamp = rand(πu, nMC)
    # ISint = ISSamples(πu, xsamp)
    # t = @elapsed ∇Qis_u, metrics_u = compute_gradQ(θsamp, q, ISint; gradh=∇h)
    # CIS_u = compute_covmatrix(∇Qis_u, nsamp_arr)
    # println("IS MC Uniform: $t sec.")

    # JLD.save("data$modnum/repl$j/DW1D_ISU_nsamp=$(nsamptot).jld",
    #     "nx", nMC,
    #     "π_bias", πu,
    #     "C", CIS_u,
    #     "metrics", metrics_u,
    # )


    # importance sampling - Gibbs -------------------------------------------------------
    # initialize
    CIS_g = Dict{Float64, Dict}()
    metrics_g = Dict{Float64, Dict}()
    
    # iterate over β
    for βi in βarr
        πg = Gibbs(πgibbs0, β=βi, θ=ρθ.μ)
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


    # importance sampling - mixture Gibbs ----------------------------------------------
    CIS = Dict{Float64, Dict}()
    metrics = Dict{Float64, Dict}()
    
    ncent = 10
    # select centers of proposal PDFs
    data = JLD.load("data$modnum/mixturedist.jld") 
    πc = [Gibbs(πgibbs0, β=data["temp"], θ=c) for c in data["centers"]]
    mm = MixtureModel(πc, data["weights"])

    # centers = reduce(vcat, ([ρθ.μ], [rand(ρθ) for i = 1:ncent-1]))

    # dc = Int(ceil(sqrt(ncent)))
    # ξc, wc = gausslegendre(dc, θrng[1], θrng[2])
    # centers = [[ξc[i], ξc[j]] for i = 1:dc, j = 1:dc]
    # if ncent < dc^2
    #     centers = [centers[1,2], centers[2,1], centers[2,3], centers[3,2], centers[2,2]]
    # else
    #     centers = centers[:]
    # end

    # define mixture model
    # πg = Gibbs(πgibbs0, β=βarr[2], θ=ρθ.μ)
    # πc = [Gibbs(πgibbs0, β=βarr[2], θ=c) for c in centers]
    # mm = MixtureModel(πc)

    # sample from proposal PDF
    xsamp = rand(mm, 10000, nuts, ρx0)
    # JLD.save("data$modnum/mixturedist.jld",
    #         "centers", centers,
    #         "temp", βarr[2],
    #         "weights", probs(mm),
    #         "xsamp", xsamp)

    # check samples
    # f = plot_pdf_with_sample_hist(mm, xplot, xsamp; ξx=ξx, wx=wx)

    # fig = Figure(resolution = (700, 600))
    # ax = Axis(fig[1, 1], xlabel="x", ylabel="pdf(x)", title="Mixture biasing distribution")
    # [lines!(ax, xplot, updf.((d,), xplot) ./ normconst(d, ξx, wx), color=:red, linestyle=:dash) for d in components(mm)]
    # lines!(ax, xplot, updf.((πg,), xplot) ./ normconst(πg, ξx, wx), color=:blue, linewidth=2, label="mean")
    # lines!(ax, xplot, updf.((mm,), xplot) ./ normconst(mm, ξx, wx), color=:black, linewidth=2, label="mixture")
    # hist!(ax, xsamp, color=(:blue, 0.2), normalization=:pdf, bins=100, label="samples")
    # axislegend(ax)
    # fig

    # # plot contours 
    # θ1_rng = Vector(LinRange(2.5, 5.5, 31))
    # θ2_rng = Vector(LinRange(2.5, 5.5, 31))
    # P_plot = [pdf(ρθ, [θi, θj]) for θi in θ1_rng, θj in θ2_rng]
    # fig = Figure(resolution = (600, 600))
    # ax = Axis(fig[1, 1], xlabel="θ₁", ylabel="θ₂", title="Parameter space")
    # contour!(ax, θ1_rng, θ2_rng, P_plot,levels=15)
    # [scatter!(ax, c[1], c[2], color=:red) for c in centers]


    ISint = ISSamples(mm, xsamp)

    t = @elapsed ∇Qis_i, metrics_i = compute_gradQ(θsamp, q, ISint; gradh=∇h)
    CIS[ncent] = compute_covmatrix(∇Qis_i, nsamp_arr)
    metrics[ncent] = metrics_i
    println("IS MC Mixture (ncent=$ncent): $t sec.")
    
    JLD.save("data$modnum/repl$j/DW1D_ISM_nsamp=$(nsamptot).jld",
        "nx", nMC,
        "ncent", ncent,
        "C", CIS,
        "metrics", metrics,
    )


end


