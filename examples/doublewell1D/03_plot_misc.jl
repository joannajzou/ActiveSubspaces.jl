using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using CairoMakie

include("model_param1.jl")
include("qoi_meanenergy.jl")
include("plotting_utils.jl")


# initialize ###############################################################################################

# QuadIntegrator
ngrid = 100                                     # number of θ quadrature points in 1D
ξθ, wθ = gausslegendre(ngrid, θrng[1], θrng[2]) # 2D quad points over θ
ξx, wx = gausslegendre(200, -10, 10)            # 1D quad points over x
GQint = GaussQuadrature(ξx, wx)

# MCIntegrator
nMC = 10000                                     # number of MC/MCMC samples
nuts = NUTS(1e-1)                               # Sampler struct
MCint = MCMC(nMC, nuts, ρx0)

# ISIntegrator
ll = -3; ul = 3
πu = Uniform(ll, ul)                              # importance sampling: uniform distribution
βarr = [0.02, 0.2, 1.0]                         # importance sampling: temp parameter for Gibbs biasing dist.
nβ = length(βarr)

# plot settings
xplot = LinRange(ll, ul, 1000)
col=[:black, :purple, :blue, :green, :orange, :red]         # set color scheme



# plot QoI ###################################################################################################

nsamptot = nsamp_arr[end] # largest number of samples
θsamp = [rand(ρθ) for i = 1:nsamptot]   
θmat = reduce(hcat, θsamp)

# plot distribution of parameters
θ1rng = LinRange(2, 6, 100); θ2rng = θ1rng              # grid across θ-domain
c_mat = [pdf(ρθ, [θi, θj]) for θi in θ1rng, θj in θ2rng]   # PDF values                                  

fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],  xlabel="θ1", ylabel="θ2",
          title="Distribution of samples (nsamp=$nsamptot)")

scatter!(ax, θmat[1,:], θmat[2,:], color=(:blue, 0.2))
# contour!(ax, θ1rng, θ2rng, c_mat, levels=0:0.1:1)
fig


# plot samples from original sampling density
xplot = LinRange(-5, 5, 1000)
fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)",
          title="Gibbs distribution of samples")
for θi in θsamp[1:10:1000]
    πg = Gibbs(πgibbs0, β=1.0, θ=θi)
    lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, ξx, wx))
end
fig


# plot samples from active subspace
# TO DO



# plot HMC samples from Gibbs distribution ###################################################################





# plot biasing distributions #################################################################################

fig = Figure(resolution = (700, 600))
ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)",
          title="Importance sampling biasing distributions")

lines!(ax, xplot, pdf.(πu, xplot), color=col[2], label="U[$ll, $ul]")
for i = 1:length(βarr)
    βi = βarr[i]
    πg = Gibbs(πgibbs0, β=βi, θ=ρθ.μ)
    lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, ξx, wx), color=col[i+2], label="Gibbs, β=$βi")
end
axislegend(ax)
fig






# plot extreme samples
xlim = [-3, 3]
xplot = LinRange(xlim[1], xlim[2], 1000)
quadpts = (ξx, wx)

θsamp = [[4.180618946996893, 0.8237016812991298],
        [4.704795333019134, 0.9902896182178393],
        [4.240282540575501, 0.6763551600914113],
        [3.156840046761258, 0.17686037626406526],
        [4.682744104308297, 0.9603268388739838],
        [5.144587203900132, 1.1636604656037772]]

θsamp = [[6.0, 2.0]]

figs = Vector{Figure}(undef, length(θsamp))
for (i, θi) in zip(1:length(θsamp), θsamp)
    figs[i] = Figure(resolution = (450, 500))
    axi = Axis(figs[i][1,1],  xlabel="x", ylabel="pdf(x)", limits=(xlim[1], xlim[2], -0.02, 0.65))
    πg = Gibbs(πgibbs0, β=1.0, θ=θi)
    xsamp = rand(πg, MCint.n, MCint.sampler, MCint.ρ0)
    lines!(axi, xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), label="$θi") 
    scatter!(axi, xsamp, updf.(πg, xsamp) ./ normconst(πg, quadpts[1], quadpts[2]), color=:black)
    axislegend(axi)
end
figs[1]


# θsamp = [[i,j] for i = 0.5:0.5:5.5, j = 0.5:0.5:5.5][:]
θsamp = [[4,i] for i = 2:0.5:6]
θsamp = [[i,4] for i = 2:0.5:6]
θsamp = [[i,8-i] for i = 2:0.5:6]
θsamp = [[8-i,i] for i = 2:0.5:6]
# θsamp = [[2, 0.1], [0.1, 2]]


f = Figure(resolution = (450, 500))
ax = Axis(f[1,1],  xlabel="x", ylabel="pdf(x)", limits=(xlim[1], xlim[2], -0.02, 0.65))
for (i, θi) in zip(1:12, θsamp) # length(θsamp)
    πg = Gibbs(πgibbs0, β=1.0, θ=θi)
    lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), label="$i") 
end
axislegend(ax)
f

μθ = 4*ones(2)
Σθ = 0.2*I(2) 
ρθ = MvNormal(μθ, Σθ) # prior on θ 

θsamp = [rand(ρθ) for i = 1:20000]