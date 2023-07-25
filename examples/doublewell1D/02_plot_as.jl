using ActiveSubspaces
using Distributions
using LinearAlgebra
using StatsBase
using JLD

# select model
modnum = 3
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("plotting_utils.jl")



# compute active subspace from reference data #########################################################################

# load reference covariance matrix
Cref = JLD.load("data$modnum/DW1D_Ref.jld")["Cref"]
Cmc = JLD.load("data$modnum/repl1/DW1D_MC_nsamp=8000.jld")["C"][8000]


# compute active subspace
as = compute_as(Cref, ρθ, 1)



# plot QoI distribution ###############################################################################################

# draw random samples from active subspace
ny = 200
ysamp, θy = sample_as(ny, as)
θymat = reduce(hcat, θy)

# draw random samples of θ from parameter density
θsamp = [rand(ρθ) for i = 1:ny]   
θmat = reduce(hcat, θsamp)

# compute QoI distribution across parameter space
θ1_rng = Vector(LinRange(2.5, 5.5, 31))
θ2_rng = Vector(LinRange(2.5, 5.5, 31))
m = length(θ1_rng)
P_plot = [pdf(ρθ, [θi, θj]) for θi in θ1_rng, θj in θ2_rng]
Q_surf = [expectation([θi, θj], q, GQint) for θi in θ1_rng, θj in θ2_rng]
Q_plot = Q_surf[end:-1:1, :] # yflip

# plot 
fa = plot_parameter_space(θ1_rng, θ2_rng, 
                    Q_plot, P_plot,
                    θmat, θymat)
save("figures/QoI_space_param$modnum.png", fa)


# plot θ samples from active subspace ################################################################################

# plot colors
N = 9
colscheme = Vector{Vector{RGB}}(undef, 2)
colscheme[1] = [RGB((i-1)*0.7/N, 0.3 + (i-1)*0.7/N, 0.6 + (i-1)*0.2/N) for i = 1:N] # color scheme for model 1
colscheme[2] = [RGB(0.6 + (i-1)*0.5/N, (i-1) * 0.9/N, (i-1) * 0.7/N) for i = 1:N] # color scheme for model 2

# modes
as = compute_as(Cref, ρθ, 3)
f = plot_as_modes(as.W1, ρθ, 30, πgibbs0, xplot, ξx, wx)

figs = Vector{Figure}(undef, 3)
as = compute_as(Cref, ρθ, 3)
for i = 1:3
    μy = as.π_y.μ[i]
    σy = sqrt(as.π_y.Σ[i,i])
    ysamp = Vector(LinRange(μy - 2.5*sqrt(σy), μy + 2.5*sqrt(σy), N))
    θsamp = [as.W1[:,i] * y for y in ysamp]
    figs[i] = plot_gibbs_pdf(θsamp, πgibbs0, xplot, ξx, wx, colscheme[2], ttl="Samples from active subspace")
    # save("figures/samples_as_param$modnum.png", fd)
end

# 2-D 
as = compute_as(Cref, ρθ, 2)
ysamp, θsamp = sample_as(20, as)
fd = plot_gibbs_pdf(θsamp, πgibbs0, xplot, ξx, wx, ttl="Samples from active subspace")
save("figures/samples_as_param$modnum.png", fd)
