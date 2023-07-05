using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using CairoMakie, Colors

# select model
modnum = 2
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("plotting_utils.jl")



# compute active subspace from reference data #########################################################################

# load reference covariance matrix
Cref = JLD.load("data$modnum/DW1D_Ref.jld")["Cref"]
Cmc = JLD.load("data$modnum/repl1/DW1D_MC_nsamp=16000.jld")["C"][16000]


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



# plot PDFs #########################################################################################################

# plot colors
N = 9
colscheme = Vector{Vector{RGB}}(undef, 2)
colscheme[1] = [RGB((i-1)*0.7/N, 0.3 + (i-1)*0.7/N, 0.6 + (i-1)*0.2/N) for i = 1:N] # color scheme for model 1
colscheme[2] = [RGB(0.6 + (i-1)*0.5/N, (i-1) * 0.9/N, (i-1) * 0.7/N) for i = 1:N] # color scheme for model 2

# vary θ1
θsamp = [[i,ρθ.μ[2]] for i in LinRange(2.0, 6.0, N)]
fb = plot_gibbs_pdf_rng(θsamp, xplot, ξx, wx, colscheme[modnum], ttl="Samples with varying θ₁")
save("figures/samples_th1_param$modnum.png", fb)

# vary θ2
θsamp = [[ρθ.μ[1], i] for i in LinRange(2.0, 6.0, N)]
fc = plot_gibbs_pdf_rng(θsamp, xplot, ξx, wx, colscheme[modnum], ttl="Samples with varying θ₂")
save("figures/samples_th2_param$modnum.png", fc)

# plot θ samples from active subspace
μy = as.π_y.μ[1]
σy = sqrt(as.π_y.Σ)[1]
ysamp = Vector(LinRange(μy - 3*sqrt(σy), μy + 3*sqrt(σy), N))
θsamp = transf_to_paramspace_fix.(ysamp, (as,))
fd = plot_gibbs_pdf_rng(θsamp, xplot, ξx, wx, colscheme[modnum], ttl="Samples from active subspace")
save("figures/samples_as_param$modnum.png", fd)

# other parametric studies
# θsamp = [[i,8-i] for i = 2:0.5:6]
θsamp = [[8-i,i] for i = 2:0.5:6]
# θsamp = [[i,i] for i = 2:0.5:6]
fd = plot_gibbs_pdf_rng(θsamp, xplot, ξx, wx, colscheme[modnum])


