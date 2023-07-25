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
# include("plotting_utils.jl")


# define variables ################################################################################
ntot = 320000                           # total number of samples
nsamp_arr = [10000, 20000, 40000, 80000, 160000, ntot]    # sub-samples
nrepl = 1000                             # number of replications

# HMC integrator
hmc = HMC(100, 1e-3)
integrator = MCMC(ntot, hmc, ρx0)

# demonstrate with sample from θ
θsamp = [2.185526152546304, 2.9277228897053478, 2.1914061235376288] # rand(ρθ)
qoim = GibbsQoI(h = x -> q.h(x, θsamp), p=Gibbs(q.p, θ=θsamp))
# x = rand(qoim.p, integrator.n, integrator.sampler, integrator.ρ0) 
# f = plot_pdf_with_sample_hist(qoim.p, xplot, x; ξx=ξx, wx=wx)

# initialize
se = Dict{Int64, Float64}()
mean_repl = Dict{Int64, Vector}()
se_sm = Dict{Int64, Vector}()
se_bm = Dict{Int64, Vector}()
se_obm = Dict{Int64, Vector}()
for n in nsamp_arr
    mean_repl[n] = Vector{Float64}(undef, nrepl)
    se_sm[n] = Vector{Float64}(undef, nrepl)
    se_bm[n] = Vector{Float64}(undef, nrepl)
    se_obm[n] = Vector{Float64}(undef, nrepl)
end



# begin iterations ################################################################################

for j = 1:nrepl

    t = @elapsed begin
    println("---------------- nrepl = $j ----------------")
       
    # draw samples -------------------------------------------------------------------------------
    xsamp = rand(qoim.p, integrator.n, integrator.sampler, integrator.ρ0) # total number of samples

    for (i,n) in enumerate(nsamp_arr)
        # println("nsamp: $n")
        # compute SE estimators ------------------------------------------------------------------
        se_sm[n][j] = MCSE(qoim, xsamp[1:n])
        se_bm[n][j] = MCSEbm(qoim, xsamp[1:n])
        se_obm[n][j] = MCSEobm(qoim, xsamp[1:n])

        # store mean estimate --------------------------------------------------------------------
        mean_repl[n][j] = mean(qoim.h.(xsamp[1:n]))

    end
    end 
    println("TIME: $t sec.")

end

# estimate "true" SE -----------------------------------------------------------------------------
se = [sqrt(var(mean_repl[n]) / n) for n in nsamp_arr]

# save -------------------------------------------------------------------------------------------
JLD.save("adaptive_mcmc_trial.jld",
    "θ", θsamp,
    "se", se,
    "se_sm", se_sm,
    "se_bm", se_bm,
    "se_obm", se_obm,
    )


# plot ###########################################################################################

d = JLD.load("adaptive_mcmc_trial.jld")
col=[:magenta, :orange, :cyan]

fig = Figure(resolution = (700, 600))
ax = Axis(fig[1, 1], xlabel="sample size (n)", ylabel="MCSE", title="Monte Carlo standard error", yscale=log10, xscale=log2, xticks=nsamp_arr)

scatter!(ax, nsamp_arr, d["se"], color=:red, label="empirical est.")
n = nsamp_arr[1]
# hist!(ax, d["se_sm"][n], scale_to=-0.25*n, offset=n, color=(col[1], 0.2), direction=:x, label="single mean")
hist!(ax, d["se_bm"][n], scale_to=-0.25*n, offset=n, color=(col[2], 0.4), bins=10, direction=:x, label="BM")
hist!(ax, d["se_obm"][n], scale_to=-0.25*n, offset=n, color=(col[3], 0.4), bins=10, direction=:x, label="OBM")
for n in nsamp_arr[2:end]
    # hist!(ax, d["se_sm"][n] .+ 1e-20, scale_to=-0.25*n, offset=n, color=(col[1], 0.2), direction=:x)
    hist!(ax, d["se_bm"][n] .+ 1e-20, scale_to=-0.25*n, offset=n, color=(col[2], 0.4), bins=10, direction=:x)
    hist!(ax, d["se_obm"][n] .+ 1e-20, scale_to=-0.25*n, offset=n, color=(col[3], 0.4), bins=10, direction=:x)
end
axislegend(ax, position=:lb)
fig

