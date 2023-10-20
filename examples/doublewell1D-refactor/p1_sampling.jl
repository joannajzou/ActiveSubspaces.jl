using CairoMakie
using StatsBase
using Colors

include("plotting_utils.jl")


## points for computing expectation ----------------------------------------------------------------

# pick value of θ
θtrial = rand(ρθ)
πgibbs = Gibbs(πgibbs1, θ=θtrial)

# QuadIntegrator
ngrid = 200                                     # number of θ quadrature points in 1D
ξx, wx = gausslegendre(ngrid, -9, 11)           # 1D quad points over x
GQint = GaussQuadrature(ξx, wx)

# Monte Carlo 
npts = 20000                                   # number of MC/MCMC samples
nuts = NUTS(0.02)                               # Sampler struct
xMC = rand(πgibbs, npts, nuts, ρx0)

# Adaptive IS
niter = 10
ncomp = 10
λsamp = [rand(ρθ) for i = 1:ncomp]
λset_m = adapt_mixture_IS(λsamp, q, πgibbs2, nuts, ρx0; normint=GQint, niter=niter)
λset_v = adapt_mixture_IS(λsamp, q2, πgibbs2, nuts, ρx0; normint=GQint, niter=niter)



## plot densities -------------------------------------------------------------------------------------------
colscheme = reverse([RGB(0.6 + (i-1)*0.5/niter, (i-1) * 0.9/niter, (i-1) * 0.7/niter) for i = 1:niter]) 

f = Figure(resolution = (500, 500))
ax = Axis(f[1,1], xlabel="x", ylabel="pdf(x)")
# target density
hist!(ax, xMC, color=(:skyblue, 0.2), normalization=:pdf, bins=50, label="MCMC samples")
lines!(ax, xplot, pdf.((πgibbs,), xplot, (GQint,)), color=:skyblue, linewidth=2, label="target density p(x)")
# adapted biasing distributions
for k = 1:length(λset_v)
    gk = MixtureModel(λset_v[k], πgibbs2)
    lines!(ax, xplot, pdf.((gk,), xplot, (GQint,)), color=colscheme[k], linewidth=2, label="adapt. iteration $k")
end
# target integrand
lines!(ax, xplot, integrand(xplot, q2, θtrial, GQint), color=:black, linewidth=3, label="summand h(x)p(x)")
# axislegend(ax)
f


## vary each parameter independently -------------------------------------------------------------------------------------------
N = 7
θrng = [[ρθ.μ[i] - 3*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3*sqrt(ρθ.Σ[i,i])] for i = 1:d] # θ-domain
θ1samp = [[θ1, ρθ.μ[2]] for θ1 in log.(LinRange(exp.(θrng[1][1]), exp.(θrng[1][2]), N))]
θ2samp = [[ρθ.μ[1], θ2] for θ2 in log.(LinRange(exp.(θrng[2][1]), exp.(θrng[2][2]), N))]

fig1 = plot_gibbs_pdf(θ1samp, πgibbs0, xplot, GQint, col=colscheme, ttl="Samples with varying θ1")
figi1 = plot_integrand(θ1samp, V, πgibbs0, xplot, GQint, colscheme, ttl="Samples with varying θ1")
figi1v = plot_integrand(θ1samp, (x,θ) -> V(x,θ)^2, πgibbs0, xplot, GQint, colscheme, ttl="Samples with varying θ1")
fig2 = plot_gibbs_pdf(θ2samp, πgibbs0, xplot, GQint, col=colscheme, ttl="Samples with varying θ2")
figi2 = plot_integrand(θ2samp, V, πgibbs0, xplot, GQint, colscheme, ttl="Samples with varying θ2")
figi2v = plot_integrand(θ2samp, (x,θ) -> V(x,θ)^2, πgibbs0, xplot, GQint, colscheme, ttl="Samples with varying θ2")


## plot parameter contour plot -------------------------------------------------------------------------------------------
θtest = reduce(vcat, (θ1samp, θ2samp))
θ1_rng = log.(Vector(LinRange(exp.(θbd[1][1]), exp.(θbd[1][2]), 31))) # exp.(3.8) θbd[1][2]
θ2_rng = log.(Vector(LinRange(exp.(θbd[2][1]), exp.(θbd[2][2]), 31))) # exp.(3.1) θbd[2][2]
# θ1_rng = Vector(LinRange(θbd[1][1], θbd[1][2], 31))
# θ2_rng = Vector(LinRange(θbd[2][1], θbd[2][2], 31))
m = length(θ1_rng)
θ_plot = [[θi, θj] for θi in θ1_rng, θj in θ2_rng]
P_plot = [pdf(ρθ, [θi, θj]) for θi in θ1_rng, θj in θ2_rng]
Q_plot = [expectation([θi, θj], q2, GQint) for θi in θ1_rng, θj in θ2_rng]

fa = plot_parameter_space(θ1_rng, θ2_rng, 
                    Q_plot, P_plot,
                    θtest, λsamp)

fb = plot_parameter_space(θ1_rng, θ2_rng, 
                    Q_plot, P_plot,
                    λsamp, λset_v[end])



## plot adaptive IS -------------------------------------------------------------------------------------------
# draw samples from components
mm = MixtureModel(λset_v[end], πgibbs2)
samples = [rand(comp, npts, nuts, ρx0) for comp in components(mm)]

# choose discrepancy metric
knlp = RBF(Euclidean(2); ℓ=1e-3)
knl = RBF(Euclidean(1))
ksd = KernelSteinDiscrepancy(knl)

figs = Vector{Figure}(undef, length(θtest))
for (t,θt) in enumerate(θtest)
    println(t)
    at = adapt_mixture_weights(θt, q2, mm, samples, ksd)
    # at = adapt_mixture_weights(θt, λsamp, knlp)
    mmt = MixtureModel(λset_v[end], πgibbs2, at)

    with_theme(custom_theme) do
        figs[t] = Figure(resolution = (1000, 500))
        ax1 = Axis(figs[t][1,1], xlabel="x", ylabel="pdf(x)") # limits=(-2, 3, 0, 1.5))
        ax2 = Axis(figs[t][1,2], xlabel="θ1", ylabel="θ2")

        lines!(ax1, xplot, integrand(xplot, q2, θt, GQint), color=:black, linewidth=3, label="summand h(x)p(x)")
        lines!(ax1, xplot, pdf.((mmt,), xplot, (GQint,)), color=colscheme[7], linewidth=3, label="adapted mixture")

        λmat = reduce(hcat, λsamp)'
        contour!(ax2, θ1_rng, θ2_rng, P_plot, color=(:black, 0.25), linewidth=2) #  levels=0:0.1:0.8) # exp.(LinRange(log(1e-3), log(1), 8))
        scatter!(ax2, θt[1], θt[2], color=:cyan, markersize=10, label="target")
        scatter!(ax2, λmat[:,1], λmat[:,2], color=at, colormap=:Reds, markersize=10, label="ref. pts.")

        axislegend(ax1, position=:lt)
        axislegend(ax2)

        save("figures/adapt_var_ksd_$t.png", figs[t])
    end
end

function integrand(x::Vector, q::GibbsQoI, θ::Vector, normint::Integrator)
    p = Gibbs(q.p, θ=θ)
    hsamp = abs.(q.h.(x, (θ,)))
    psamp = pdf.((p,), x, (normint,))
    return hsamp .* psamp
end


