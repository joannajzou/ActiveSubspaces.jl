include("main.jl")
include("mc_utils.jl")
include("plotting_utils.jl")

using DelimitedFiles

# load data -----------------------------------------------------------------------

satdir = "$(simdir)ACE_MD_satori/"
# skipped ids
# idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]
# idkeep = symdiff(1:nsamp, idskip) 
# JLD.save("$(simdir)coeff_keep.jld", "idkeep", idkeep)

# coeff samples
βsamp1 = JLD.load("$(simdir)coeff_samples_1-300.jld")["θsamp"]
idskip1 = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]
idkeep1 = symdiff(1:length(βsamp1), idskip1)
# JLD.save("$(simdir)coeff_keep.jld", "idkeep", idkeep1)


βsamp2 = JLD.load("$(satdir)coeff_samples.jld")["θsamp"]
idskip2 = JLD.load("$(satdir)coeff_skip.jld")["id_skip"]
idkeep2 = symdiff(1:length(βsamp2), idskip2)
# JLD.save("$(satdir)coeff_keep.jld", "idkeep", idkeep2)

βsamp = reduce(vcat, (βsamp1[idkeep1], βsamp2[idkeep2]))
βdim = length(βsamp[1])

# load MC estimates
∇Q1 = [JLD.load("$(simdir)coeff_$j/gradQ_energyvar.jld")["∇Q"] for j in idkeep1]
∇Q2 = [JLD.load("$(satdir)coeff_$j/gradQ_energyvar.jld")["∇Q"] for j in idkeep2]
∇Qmc = reduce(vcat, (∇Q1, ∇Q2))


# compute eigenbasis ----------------------------------------------------------
asmc = compute_as(∇Qmc, ρθ, βdim)

# check convergence of eigenspectrum
n_arr = [375, 750, 1500, 3000]
n_arr = [500, 1000, 1500, 2000, 2500, 3000]
λ_arr = [compute_eigenbasis(∇Qmc[1:n_arr[i]])[2] for i in 1:length(n_arr)]
f = plot_eigenspectrum(λ_arr, labs=[string(n) for n in n_arr])
save("$(simdir)/figures/eigenspectrum.png", f)

f = plot_eigenspectrum(asmc.λ)
save("$(simdir)/figures/eigenspectrum_single.png", f)

# modes of as
ny = 50
figsa = plot_as_modes(asmc, ρθ, ny, Vector(r), B; scl=1e-1)
fa = plot_eigenvectors_row(asmc.W1)


# plot mode shapes of pairwise potential --------------------------------------------

# define 2D points for pairwise potential
r = Vector(1.0:0.01:11)
box = [[10.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
system = [FlexibleSystem([AtomsBase.Atom(:Hf, [0.0, 0.0, 0.0] * u"Å"), AtomsBase.Atom(:Hf, [ri, 0.0, 0.0] * u"Å")],
                        box * u"Å", bcs) for ri in r]
B = [sum(compute_local_descriptors(sys, ace)) for sys in system]


# plot samples from active subspace --------------------------------------------

# from original sampling density
f2 = plot_pairwise_potential(ρθ, 200, Vector(r), B; scl=1e-1)
save("$(simdir)/figures/sampling_density.png", f2)


# from active subspace
as = compute_as(∇Qmc, ρθ, 3)
ysamp = [rand(as.π_y) for i = 1:100]
θy = [as.W1*y + as.W2*as.π_z.μ for y in ysamp]
plot_pairwise_potential(θy, Vector(r), B)



# plot variation of QoI along directions --------------------------------------------

# compute Monte Carlo estimates of Q 
Qmc = compute_Q(βsamp, idkeep, q, simdir)

# compute IS estimates of Q along other dimensions
# construct IS estimator
Temp = 200; Temp_b = 500.0
Tr = Temp/Temp_b          # temperature

πg = Gibbs(πgibbs, θ=ρθ.μ, β=Tr)
Φsamp = JLD.load("$(simdir)coeff_mean_temp/energy_descriptors.jld")["Bsamp"]
ISint = ISSamples(πg, Φsamp)
MCint = MCSamples(Φsamp)

variance(ρθ.μ, q, q2, MCint)
variance2(ρθ.μ, q, q2, MCint)
variance(ρθ.μ, q, q2, ISint)

nsamp = 10000
function variance(θ::Vector{<:Real}, q::GibbsQoI, q2::GibbsQoI, integrator::ISIntegrator)
    return expectation(θ, q2, integrator)[1] - expectation(θ, q, integrator)[1]^2
end
function variance(θ::Vector{<:Real}, q::GibbsQoI, q2::GibbsQoI, integrator::Integrator)
    return expectation(θ, q2, integrator) - expectation(θ, q, integrator)^2
end
function variance2(θ::Vector{<:Real}, q::GibbsQoI, q2::GibbsQoI, integrator::Integrator)
    expec = expectation(θ, q, integrator)
    qvar = GibbsQoI(h=(x, γ) -> (q.h(x, γ) - expec)^2,
        p = q.p)
    return expectation(θ, qvar, integrator)
end

# from original density
@time begin
θsamp = [rand(ρθ) for i = 1:nsamp]
Q0 = map(θ -> expectation(θ, q, ISint)[1], θsamp)
end

# from active subspace
@time begin
as4 = compute_as(∇Qmc, ρθ, 4)
_, θysamp = sample_as(nsamp, as4)
Qas4 = map(θ -> expectation(θ, q, ISint)[1], θysamp)
end
@time begin
as5 = compute_as(∇Qmc, ρθ, 5)
_, θysamp = sample_as(nsamp, as5)
Qas5 = map(θ -> expectation(θ, q, ISint)[1], θysamp)
end

# from pca
@time begin
pca3 = compute_as(Matrix(ρθ.Σ), ρθ, 3)
_, θysamp = sample_as(nsamp, pca3)
Qpca3 = map(θ -> expectation(θ, q, ISint)[1], θysamp)
end
@time begin
pca5 = compute_as(Matrix(ρθ.Σ), ρθ, 5)
_, θysamp = sample_as(nsamp, pca5)
Qpca5 = map(θ -> expectation(θ, q, ISint)[1], θysamp)
end

JLD.save("$(simdir)Q_meanenergy_subspace_samples.jld",
    "Q0", Q0,
    "Qas3", Qas3,
    "Qas5", Qas5,
    "Qpca3", Qpca3,
    "Qpca5", Qpca5,
)


# from as modes
nsamp = 2000
Qmod = Vector{Vector{Float64}}(undef, βdim)
@time begin
for i = 1:βdim
    println("$i")
    θyi = draw_samples_as_mode(asmc.W1, i, ρθ, nsamp)
    Qmod[i] = map(θ -> expectation(θ, q, ISint)[1], θyi)
end
end

# from pca modes
pca = compute_as(Matrix(ρθ.Σ), ρθ, βdim)
Qpca = Vector{Vector{Float64}}(undef, βdim)
@time begin
for i = 1:βdim
_, θysamp = sample_as(nsamp, pca3)
Qpca3 = map(θ -> expectation(θ, q, ISint)[1], θysamp)
end
end


# from each parameter dimension
# pid = compute_as(Matrix(I(βdim)), ρθ, βdim)
# Qpid = Vector{Vector{Float64}}(undef, βdim)
# @time begin
# for i = 1:βdim
#     println("$i")
#     θyi = draw_samples_as_mode(pid.W1, i, ρθ, nsamp)
#     Qpid[i] = map(θ -> expectation(θ, q2, ISint)[1], θyi)
# end
# end


# compute subspaces
nsamp = 1000
Qas = Vector{Vector{Float64}}(undef, βdim)
Qpca = Vector{Vector{Float64}}(undef, βdim)
for i = 1:βdim
    @time begin
    println("$i")
    # active subspace
    asi = compute_as(∇Qmc, ρθ, i)
    _, θysamp = sample_as(nsamp, asi)
    Qas[i] = map(θ -> expectation(θ, q2, ISint)[1], θysamp)
    # pca
    pcai = compute_as(Matrix(ρθ.Σ), ρθ, i)
    _, θysamp = sample_as(nsamp, pcai)
    Qpca[i] = map(θ -> expectation(θ, q2, ISint)[1], θysamp)
    end
end

JLD.save("$(simdir)Q_energyvar_subspaces_ASvsPCA.jld",
    "Qas", Qas,
    "Qpca", Qpca,
    # "Qpid", Qpid
)
Q0 = JLD.load("$(simdir)Q_meanenergy_subspace_samples.jld")["Q0"]
Qas = JLD.load("$(simdir)Q_energyvar_subspaces_ASvsPCA.jld")["Qas"]
Qpca = JLD.load("$(simdir)Q_energyvar_subspaces_ASvsPCA.jld")["Qpca"]


# compute quantile error
quant0 = compute_quantiles(abs.(Q0))
quant_as = [compute_quantiles(Q) for Q in Qas]
qerr_as = [abs.((quant - quant0) ./ quant0) for quant in quant_as]
qerr_as = reduce(hcat, qerr_as)
quant_pca = [compute_quantiles(Q) for Q in Qpca]
qerr_pca = [abs.((quant - quant0) ./ quant0) for quant in quant_pca]
qerr_pca = reduce(hcat, qerr_pca)



f = Figure(resolution=(800, 400))
# ax1 = Axis(f[1,1],
#     xlabel="subspace dimension",
#     ylabel="quantile of Q",
#     title="Active subspace",
#     xticks=1:18,
#     yticks=0.1:0.1:0.9
#     )
ax2 = Axis(f[1,1][1,1],
    xlabel="subspace dimension",
    ylabel="quantile of Q",
    title="Quantile Error in Distribution of Q",
    xticks=1:18,
    yticks=0.1:0.1:0.9
    )
# jointlimits = (0.01, 0.40)
# hm1 = heatmap!(ax1, 1:18, 0.1:0.1:0.9, qerr_as') # , colorrange=jointlimits)
hm2 = heatmap!(ax2, 1:18, 0.1:0.1:0.9, qerr_pca') # , colorrange=jointlimits)

# Colorbar(f[1,1][1,2], hm1, label="rel. error")
Colorbar(f[1,1][1,2], hm2, label="rel. error")
f

fig1 = Figure(resolution=(350, 550))
ax = Axis(fig1[1,1],
    xlabel="count",
    ylabel="Q",
    title="Distribution of Q",
    yscale=log10,
    xgridcolor=:white,
    ygridcolor=:white)

histbins = exp.(LinRange(log(10^3.5), log(10^5), 80))
# histbins = exp.(LinRange(log(10^7.5), log(10^9.5), 50))

hist!(ax, abs.(Q0[2001:5000]),
    direction=:x,
    bins = histbins,
    offset=0.0,
    color=(:green, 0.3),
    # strokecolor=:green,
    # strokewidth=2,
    # strokearound=true,
    scale_to=-140,
    label="full space")
hist!(ax, abs.(Qas4[2001:end]),
    direction=:x,
    bins = histbins,
    offset=0.0,
    color=(:red, 0.3),
    # strokecolor=:red,
    # strokewidth=2,
    # strokearound=true,
    scale_to=-150,
    label="active subspace")
# hist!(ax, abs.(Qpca[5]),
#     direction=:x,
#     bins = histbins,
#     offset=0.0,
#     color=(:blue, 0.3),
#     # strokecolor=:red,
#     # strokewidth=2,
#     # strokearound=true,
#     # scale_to=1,
#     label="pca")
# [hlines!(ax, qt, linestyle=:dash, linewidth=0.5, color=:black) for qt in quant0]
axislegend(ax)
fig1


fig = Figure(resolution=(1500, 400))
ax = Axis(fig[1,1],
    xlabel="dimension",
    ylabel="Q",
    title="Distribution of Q by dimension",
    xticks=1:βdim,
    yscale=log10,
    ygridcolor=:white)

# histbins = exp.(LinRange(log(10^3.5), log(10^5), 80))
# histbins = exp.(LinRange(log(10^7.5), log(10^9.5), 50))

for i in 1:βdim
    hist!(ax, abs.(Qas[i]),
        direction=:x,
        # bins = histbins,
        offset=1*i,
        color=(:red, 0.3),
        # strokecolor=:blue,
        # strokewidth=2,
        # strokearound=true,
        scale_to=0.5)
    
    hist!(ax, abs.(Qpca[i]),
        direction=:x,
        # bins = histbins,
        offset=1*i,
        color=(:blue, 0.3),
        # strokecolor=:blue,
        # strokewidth=2,
        # strokearound=true,
        scale_to=0.5)

    # hist!(ax, abs.(Qpid[i]),
    #     direction=:x,
    #     bins = histbins,
    #     offset=1*i,
    #     color=(:green, 0.3),
    #     # strokecolor=:blue,
    #     # strokewidth=2,
    #     # strokearound=true,
    #     scale_to=0.5)
end

fig


fig = Figure(resolution=(800, 400))
ax = Axis(fig[1,1],
    xlabel="dimension",
    ylabel="var(Q)",
    title="Variance of Q by dimension",
    xticks=1:βdim,
    yscale=log10,
    ygridcolor=:white)

scatterlines!(ax, 1:βdim, 
    [var(Q) for Q in Qmod],
    color=:red,
    label="active subspace"
    )

scatterlines!(ax, 1:βdim, 
    [var(Q) for Q in Qpca],
    color=:blue,
    label="PCA"
    )
scatterlines!(ax, 1:βdim, 
    [var(Q) for Q in Qpid],
    color=:green,
    label="dim."
    )

axislegend(ax)
fig




