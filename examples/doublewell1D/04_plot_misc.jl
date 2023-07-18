using ActiveSubspaces
using Distributions
using LinearAlgebra
using StatsBase
using JLD
using InvertedIndices

# select model
modnum = 3
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("plotting_utils.jl")



# load data ############################################################################################### 
nsamp_arr = [1000, 2000, 4000, 8000, 16000]
nsamptot = nsamp_arr[end]
nrepl = 10



# plot parametric variation in pdf ########################################################################

# plot colors
N = 9
colscheme = [RGB((i-1)*0.7/N, 0.3 + (i-1)*0.7/N, 0.6 + (i-1)*0.2/N) for i = 1:N] # color scheme for model 1
# colscheme = [RGB(0.6 + (i-1)*0.5/N, (i-1) * 0.9/N, (i-1) * 0.7/N) for i = 1:N] # color scheme for model 2


# random samples from sampling density
θsamp = [rand(ρθ) for i = 1:100]
g = plot_gibbs_pdf(θsamp, πgibbs0, xplot, ξx, wx, ttl="Random samples from ρ(θ)")


# vary each parameter independently
figs = Vector{Figure}(undef, d)
for i = 1:d
    θrng = [ρθ.μ[i] - 3.5*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3.5*sqrt(ρθ.Σ[i,i])] # θ-domain
    θmat = reduce(hcat, [ρθ.μ for j = 1:N])'
    θmat[:,i] = log.(LinRange(exp.(θrng[1]), exp.(θrng[2]), N)) # exp.(LinRange(log.(θrng[1]), log.(θrng[2]), N))
    θsamp = [θmat[j,:] for j = 1:N]

    figs[i] = plot_gibbs_pdf(θsamp, πgibbs0, xplot, ξx, wx, colscheme, ttl="Samples with varying θ$i")
    # save("figures/samples_th$i_param$modnum.png", fb)
end


# fixing one parameter at a time
M = 20
figs2 = Vector{Figure}(undef, d)
for i = 1:d
    θmat = reduce(hcat, [ρθ.μ for j = 1:M])'
    ρmod = MvNormal(ρθ.μ[Not(i)], ρθ.Σ[Not(i),Not(i)])
    θmat[:,Not(i)] = rand(ρmod, M)'
    θsamp = [θmat[j,:] for j = 1:M]
    figs2[i] = plot_gibbs_pdf(θsamp, πgibbs0, xplot, ξx, wx, ttl="Random samples with fixed θ$i")
    # save("figures/samples_th$i_param$modnum.png", fb)
end


# other parametric studies
# θsamp = [[i,8-i] for i = 2:0.5:6]
θsamp = [[8-i,i] for i = 2:0.5:6]
# θsamp = [[i,i] for i = 2:0.5:6]
fd = plot_gibbs_pdf_rng(θsamp, xplot, ξx, wx, colscheme[modnum])



# illustration of potential energy function ###############################################################
θi = ρθ.μ
xplot = LinRange(-2, 2, 1000)
xsamp = rand(πg, 10000, nuts, ρx0)
xsamp = xsamp[1:10:1000]

with_theme(custom_theme) do
    f = Figure(resolution = (650, 600))
    ax1 = Axis(f[1,1], yaxisposition = :left, xlabel="x", ylabel="V(x; θ)", limits=(-1.8, 1.8, -1, 4), yticklabelcolor = :dodgerblue3)
    ax2 = Axis(f[1,1], yaxisposition = :right, xlabel="x", ylabel="π(x; θ)",limits=(-1.8, 1.8, 0, 1.2), yticklabelcolor = :darkorange2)
    # potential energy
    lines!(ax1, xplot, -V.(xplot, (θi,)), label="potential function")
    # pdf
    πg = Gibbs(πgibbs0, β=1.0, θ=θi)
    band!(ax2, xplot, zeros(length(xplot)), updf.(πg, xplot) ./ normconst(πg, ξx, wx), color=(:darkorange2, 0.2), label="density function") # ./ normconst(πg, quadpts[1], quadpts[2])
    # samples
    scatter!(ax1, xsamp, -V.(xsamp, (θi,)), markersize=7, label="x-samples")
    # axislegend(ax1, framevisible=false, position=:ct)
    f
    save("figures/potentialfunction.png", f)
end



# illustration of importance sampling ###################################################################

xsamp = rand(πg2, 10000, nuts, ρx0)
dist = Gibbs(πgibbs0, β=1.0, θ=[5, 4]) # ρθ.μ

with_theme(custom_theme) do
    fig2 = Figure(resolution = (600, 550))
    ax = Axis(fig2[1, 1], xlabel = "x", ylabel = "pdf(x)", limits=(-3, 3, 0, 0.65))
    x_grid = LinRange(-3, 3, 1000)

    lines!(ax, x_grid, updf.(dist, x_grid) ./ normconst(dist, ξx, wx), color=:steelblue2, label="π(x)")
    lines!(ax, x_grid, updf.(πg2, x_grid) ./ normconst(πg2, ξx, wx), color=:red, linewidth=2.5, label="g(x)")
    # scatter!(ax, xsamp[1:2:end], updf.(πg2, xsamp[1:2:end]) ./ normconst(πg2, ξx, wx), color=:steelblue3, markersize=6, label="samples")
    # hist!(ax, xsamp, color=(:steelblue3, 0.25), normalization = :pdf, bins=70, label="sample density")

    # axislegend(ax, position=:lt, framevisible = false)
    fig2
    save("figures/importancesampling2.png", fig2)
end