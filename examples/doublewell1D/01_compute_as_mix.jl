using ActiveSubspaces
using Distributions
using LinearAlgebra
using StatsBase
using JLD

# select model
modnum = 2
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("mc_utils.jl")
# include("plotting_utils.jl")


# specify sample sizes and number of replications for simulation ##################################
nsamp_arr = [1000, 2000, 4000, 8000, 16000]     # sample sizes
nrepl = 1                                       # number of replications per sample size
        


# run MC trials ##################################################################################
try
    path = "data$modnum/repl$j"
    run(`mkdir -p $path`)
catch
end

for j = 1:nrepl
    println("================= REPLICATION nrepl = $j =================")

    # load parameters
    nsamptot = nsamp_arr[end] # largest number of samples
    # θsamp = [rand(ρθ) for i = 1:nsamptot]   
    # θsamp = remove_outliers(θsamp)
    # nsamp_arr[end] = length(θsamp)
    θsamp = JLD.load("data$modnum/repl$j/DW1D_ISU_nsamp=16000.jld")["metrics"]["θsamp"]


    
    # importance sampling - mixture Gibbs ----------------------------------------------
    # initialize
    CIS = Dict{Float64, Dict}()
    metrics = Dict{Float64, Dict}()
    
    ncent_arr = [5] # , 9, 16]
    for ncent in ncent_arr
        # select centers of proposal PDFs
        dc = Int(ceil(sqrt(ncent)))
        ξc, wc = gausslegendre(dc, θrng[1], θrng[2])
        centers = [[ξc[i], ξc[j]] for i = 1:dc, j = 1:dc]
        if ncent < dc^2
            centers = [centers[1,2], centers[2,1], centers[2,3], centers[3,2], centers[2,2]]
        else
            centers = centers[:]
        end

        # define mixture model
        πg = Gibbs(πgibbs0, β=βarr[2], θ=ρθ.μ) 

        πc = [Gibbs(πgibbs0, β=βarr[3], θ=c) for c in centers]
        mm = MixtureModel(πc)

        # sample from proposal PDF
        xsamp = rand(πg, 100000, nuts, ρx0)

        # check samples
        f = plot_pdf_with_sample_hist(πg, xplot, xsamp; ξx=ξx, wx=wx)
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
        println("IS MC Mixture (ncent=$ncent): $t sec.")ß
    
    end

  

    JLD.save("data$modnum/repl$j/DW1D_ISM_nsamp=$(nsamptot)_2.jld",
        "nx", nMC,
        "ncent_arr", ncent_arr,
        "C", CIS,
        "metrics", metrics,
    )

end

