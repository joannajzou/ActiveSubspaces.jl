using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using CairoMakie

# select model
modnum = 1
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("plotting_utils.jl")



# load data ##############################################################################
nsamp_arr = [1000, 2000, 4000, 8000, 16000]
nsamptot = nsamp_arr[end]
nrepl = 10
d = 2



# illustrate importance sampling ###################################################################

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



# plot biasing distributions #################################################################################
xplot = LinRange(-4, 4, 1000)
col = [:skyblue1, :seagreen, :goldenrod1, :darkorange2] 
θsamp1 = [[8-i,i] for i = 2:0.5:6]
θsamp2 = [[i,i] for i = 2:0.5:6]
θsamp3 = [[4,i] for i = 2:0.5:6]
θsamp = reduce(vcat, (θsamp1, θsamp2, θsamp3))

quadpts = (ξx, wx)

with_theme(custom_theme) do

    fig = Figure(resolution = (750, 680))
    ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)", xticks=-4:2:4,
            title="Importance sampling biasing distributions")

    for i in 1:length(θsamp)
        θi = θsamp[i]
        πg = Gibbs(πgibbs0, β=1.0, θ=θi)
        if i == length(θsamp)
            lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:black, 0.2), linewidth=1, label="samples") 
        else
            lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:black, 0.1), linewidth=1) 
        end
    end

    lines!(ax, xplot, pdf.(πu, xplot), color=col[1], label="U[-3, 3]")
    for i = 1:length(βarr)
        βi = βarr[i]
        πg = Gibbs(πgibbs0, β=βi, θ=ρθ.μ)
        lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, ξx, wx), color=col[i+1], label="Gibbs, β=$βi", linewidth=3)
    end
    hlines!(ax, 0, color=(:black, 0.1))
    axislegend(ax)
    fig

    # save("figures/biasingdist.png", fig)
end



# plot potential energy function #################################################################################
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
