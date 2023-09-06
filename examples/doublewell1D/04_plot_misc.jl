using ActiveSubspaces
using Distributions
using LinearAlgebra
using StatsBase
using JLD
using InvertedIndices

# select model
modnum = 4
include("model_param$modnum.jl")

# select qoi
include("qoi_meanenergy.jl")

# load utils
include("plotting_utils.jl")



# load data ############################################################################################### 
nsamp_arr = [1000, 2000, 4000, 8000, 16000]
nsamptot = nsamp_arr[end]
nrepl = 10


# load biasing dist.
πg = Gibbs(πgibbs2, θ=ρθ.μ)
λu = JLD.load("data$modnum/repl$repl/DW1D_ISM_nsamp=$(nsamptot).jld")["mm_cent"]
mmu = MixtureModel(λu, πgibbs2)
λa = JLD.load("data$modnum/repl$repl/DW1D_ISaM_nsamp=$(nsamptot).jld")["mm_cent"][end]
αa = JLD.load("data$modnum/repl$repl/DW1D_ISaM_nsamp=$(nsamptot).jld")["mm_wts"][end]
mma = MixtureModel(λa, πgibbs2, αa)

biasdist = [πg, mmu, mma]


# plot parametric variation in pdf ########################################################################

# plot colors
N = 9
colscheme = [RGB((i-1)*0.7/N, 0.3 + (i-1)*0.7/N, 0.6 + (i-1)*0.2/N) for i = 1:N] # color scheme for model 1
# colscheme = [RGB(0.6 + (i-1)*0.5/N, (i-1) * 0.9/N, (i-1) * 0.7/N) for i = 1:N] # color scheme for model 2


# random samples from sampling density
θsamp = [rand(ρθ) for i = 1:100]
g = plot_gibbs_pdf(θsamp, πgibbs0, xplot, GQint, ttl="Random samples from ρ(θ)")

# vary each parameter independently
figs = Vector{Figure}(undef, d)
figsb = Vector{Figure}(undef, d)
for i = 1:d
    θrng = [ρθ.μ[i] - 3.5*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3.5*sqrt(ρθ.Σ[i,i])] # θ-domain
    θmat = reduce(hcat, [ρθ.μ for j = 1:N])'
    θmat[:,i] = log.(LinRange(exp.(θrng[1]), exp.(θrng[2]), N)) # exp.(LinRange(log.(θrng[1]), log.(θrng[2]), N))
    θsamp = [θmat[j,:] for j = 1:N]
    figs[i] = plot_gibbs_pdf(θsamp, πgibbs0, xplot, GQint, col=colscheme, ttl="Samples with varying θ$i")
    figsb[i] = plot_integrand(θsamp, V, πgibbs0, xplot, GQint, colscheme, ttl="Samples with varying θ$i")
    # save("figures/samples_th$i_param$modnum.png", fb)
end


# fixing one parameter at a time
M = 100
figs2 = Vector{Figure}(undef, d)
for i = 1:d
    θmat = reduce(hcat, [ρθ.μ for j = 1:M])'
    ρmod = MvNormal(ρθ.μ[Not(i)], ρθ.Σ[Not(i),Not(i)])
    θmat[:,Not(i)] = rand(ρmod, M)'
    θsamp = [θmat[j,:] for j = 1:M]
    figs2[i] = plot_gibbs_pdf(θsamp, πgibbs0, xplot, GQint, ttl="Random samples with fixed θ$i")
    # save("figures/samples_th$i_param$modnum.png", fb)
end


# other parametric studies
θbd = [[ρθ.μ[i] - 3.5*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3.5*sqrt(ρθ.Σ[i,i])] for i = 1:d]
θrng = reduce(hcat, [LinRange(exp(θbd[i][1]), exp(θbd[i][2]), N) for i = 1:d])
θsamp = [log.(θrng[j,:]) for j = 1:N]
# θsamp = [[i,8-i] for i = 2:0.5:6]
# θsamp = [[8-i,i] for i = 2:0.5:6]
# θsamp = [[i,j] for i = 2:0.5:6]
fd = plot_gibbs_pdf(θsamp, πgibbs0, xplot, GQint, col=colscheme)



# plot sampling quality ####################################################################################

# vary each parameter independently
θsamp = []
for i = 1:d
    θrng = [ρθ.μ[i] - 3.5*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3.5*sqrt(ρθ.Σ[i,i])] # θ-domain
    θmat = reduce(hcat, [ρθ.μ for j = 1:N])'
    θmat[:,i] = log.(LinRange(exp.(θrng[1]), exp.(θrng[2]), N)) # exp.(LinRange(log.(θrng[1]), log.(θrng[2]), N))
    append!(θsamp, [θmat[j,:] for j = 1:N])
end
nθ = d*N

figs = Vector{Figure}(undef, nθ)
figs2 = Vector{Figure}(undef, nθ)
samps = Vector{Vector{Float64}}(undef, nθ)
se_bm = Vector{Float64}(undef, nθ)

for n = 1:nθ
    πgn = Gibbs(πgibbs1, θ=θsamp[n])
    xsamp = rand(πgn, 20000, nuts, ρx0)
    figs[n] = plot_pdf_with_sample_hist(πgn, xplot, xsamp)
    figs2[n] = plot_mcmc_trace(Matrix(xsamp[:,:]'))

    qoim = GibbsQoI(h = x -> q.h(x, θsamp[n]), p=Gibbs(q.p, θ=θsamp[n]))
    se_bm[n] = MCSEbm(qoim, xsamp)
    samps[n] = xsamp
end
f = Figure()
ax = Axis(f[1,1])
scatterlines!(ax, 1:(d*N), se_bm)
f

# mixture biasing distribution
t = 14
# idx = reduce(vcat, [Vector(1:t), Vector((2*N-t):2*N)])
# idx = Vector((N-t):(N+t))
# idx = Vector((t-2):(t+2))
# idx = [1,9]
mm = MixtureModel([Gibbs(πgibbs1, θ=θn) for θn in θsamp])
# xsamp_mm,_ = rand(mm, 100000, samps)
xsamp_mm = rand(mm, 20000, nuts, ρx0)
plot_pdf_with_sample_hist(mm, xplot, xsamp_mm)



function obj_mcmc_tuning(params::Vector)
    
    ϵ = params[1]
    nMC = Int(round(params[2]))

    nuts = NUTS(ϵ)

    se_bm = Vector{Float64}(undef, nθ)
    for n = 7:12
        qoim = GibbsQoI(h = x -> q.h(x, θsamp[n]), p=Gibbs(q.p, θ=θsamp[n]))
        πgn = Gibbs(πgibbs1, θ=θsamp[n])
        xsamp = rand(πgn, nMC, nuts, ρx0)
        se_bm[n] = MCSEbm(qoim, xsamp)
    end
    return mean(se_bm)
end

@time res = optimize(obj_mcmc_tuning, [0.01, 10000], NelderMead(),
                 Optim.Options(show_trace=true, f_tol=1e-3, g_tol=1e-5, f_calls_limit=50))

# adapted parameters
λa = JLD.load("data$modnum/repl$repl/DW1D_ISaM_nsamp=$(nsamptot).jld")["mm_cent"][end]
nλ = length(λa)
figs = Vector{Figure}(undef, nλ)
figs2 = Vector{Figure}(undef, nλ)
se_bm = Vector{Float64}(undef, nλ)

for n = 1:nλ
    πgn = Gibbs(πgibbs1, θ=λa[n])
    xsamp = rand(πgn, 10000, nuts, ρx0)
    figs[n] = plot_pdf_with_sample_hist(πgn, xplot, xsamp, ξx=ξx, wx=wx)
    figs2[n] = plot_mcmc_trace(Matrix(xsamp[:,:]'))

    qoim = GibbsQoI(h = x -> q.h(x, λa[n]), p=Gibbs(q.p, θ=λa[n]))
    se_bm[n] = MCSEbm(qoim, xsamp)
end

f = Figure()
ax = Axis(f[1,1])
scatterlines!(ax, 1:nλ, se_bm)
f


# plot ϕ(x) π(x) (including summand) ######################################################################
function integrand(x::Real, θ::Vector, V::Function, π0::Gibbs, ξx::Vector, wx::Vector; β=1.0)
    πg = Gibbs(π0, β=β, θ=θ)
    return V(x, θ) * updf(πg, x) ./ normconst(πg, GQint)
end

function integrand(x::Vector{<:Real}, θ::Vector, V::Function, π0::Gibbs, ξx::Vector, wx::Vector; β=1.0)
    πg = Gibbs(π0, β=β, θ=θ)
    return V.(x, (θ,)) .* updf.((πg,), x) ./ normconst(πg, GQint)
end


with_theme(custom_theme) do
    fig = Figure(resolution = (600, 600))
    ax = Axis(fig[1, 1],  xlabel="x", ylabel="ϕ(x) π(x)")
    for (i, θi) in enumerate(θsamp)
        lines!(ax, xplot, integrand(xplot, θi, V, πgibbs0, ξx, wx))
    end
    return fig
end


# illustration of potential energy function ###############################################################
θi = ρθ.μ
πg = Gibbs(πgibbs, θ=θi)
g = plot_gibbs_pdf([θi], πgibbs0, xplot, ξx, wx, ttl="Random samples from ρ(θ)")

xplot = LinRange(-2, 3, 1000)
xsamp = rand(πg, 10000, nuts, ρx0)
xsamp = xsamp[1:10:1000]

with_theme(custom_theme) do
    f = Figure(resolution = (650, 600))
    ax1 = Axis(f[1,1], yaxisposition = :left, xlabel="x", ylabel="V(x; θ)", limits=(-1.8, 1.8, -1, 4), yticklabelcolor = :dodgerblue3)
    ax2 = Axis(f[1,1], yaxisposition = :right, xlabel="x", ylabel="π(x; θ)",limits=(-1.8, 1.8, 0, 1.2), yticklabelcolor = :darkorange2)
    # potential energy
    lines!(ax1, xplot, V.(xplot, (θi,)), label="potential function")
    # pdf
    band!(ax2, xplot, zeros(length(xplot)), updf.((πg,), xplot) ./ normconst(πg, GQint), color=(:darkorange2, 0.2), label="density function") # ./ normconst(πg, quadpts[1], quadpts[2])
    # samples
    scatter!(ax1, xsamp, V.(xsamp, (θi,)), markersize=7, label="x-samples")
    # axislegend(ax1, framevisible=false, position=:ct)
    f
    # save("figures/potentialfunction.png", f)
end



# illustration of importance sampling ###################################################################

xsamp = rand(πg2, 10000, nuts, ρx0)
dist = Gibbs(πgibbs0, β=1.0, θ=[5, 4]) # ρθ.μ

with_theme(custom_theme) do
    fig2 = Figure(resolution = (600, 550))
    ax = Axis(fig2[1, 1], xlabel = "x", ylabel = "pdf(x)", limits=(-3, 3, 0, 0.65))
    x_grid = LinRange(-3, 3, 1000)

    lines!(ax, x_grid, updf.(dist, x_grid) ./ normconst(dist, GQint), color=:steelblue2, label="π(x)")
    lines!(ax, x_grid, updf.(πg2, x_grid) ./ normconst(πg2, GQint), color=:red, linewidth=2.5, label="g(x)")
    # scatter!(ax, xsamp[1:2:end], updf.(πg2, xsamp[1:2:end]) ./ normconst(πg2, ξx, wx), color=:steelblue3, markersize=6, label="samples")
    # hist!(ax, xsamp, color=(:steelblue3, 0.25), normalization = :pdf, bins=70, label="sample density")

    # axislegend(ax, position=:lt, framevisible = false)
    fig2
    save("figures/importancesampling2.png", fig2)
end

