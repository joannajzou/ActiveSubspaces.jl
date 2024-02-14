include("main.jl")
include("mc_utils.jl")
include("plotting_utils.jl")

using DelimitedFiles

# load data -----------------------------------------------------------------------

# skipped ids
# idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]
# idkeep = symdiff(1:nsamp, idskip) 
# JLD.save("$(simdir)coeff_keep.jld", "idkeep", idkeep)

# coeff samples
βsamp = JLD.load("$(simdir)coeff_samples.jld")["βsamp"]
idkeep = JLD.load("$(simdir)coeff_keep.jld")["idkeep"]
βsamp = βsamp[idkeep]
βdim = length(μ)


# load MC estimates
∇Qmc = [JLD.load("$(simdir)coeff_$j/gradQ_meanenergy.jld")["∇Q"] for j in idkeep]


# compute eigenbasis ----------------------------------------------------------
asmc = compute_as(∇Qmc, πβ, βdim)

# check convergence of eigenspectrum
n_arr = [375, 750, 1500, 3000]
n_arr = [500, 1000, 1500, 2000, 2500, 3000]
λ_arr = [compute_eigenbasis(∇Qmc[1:n_arr[i]])[2] for i in 1:length(n_arr)]
f = plot_eigenspectrum(λ_arr, labs=[string(n) for n in n_arr])
save("$(simdir)/figures/eigenspectrum.png", f)

f = plot_eigenspectrum(asmc.λ)
save("$(simdir)/figures/eigenspectrum_single.png", f)


# plot mode shapes of pairwise potential --------------------------------------------

# define 2D points for pairwise potential
r = Vector(1.0:0.01:5)
box = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
system = [FlexibleSystem([AtomsBase.Atom(:Ar, [0.0, 0.0, 0.0] * u"Å"), AtomsBase.Atom(:Ar, [ri, 0.0, 0.0] * u"Å")],
                        box * u"Å", bcs) for ri in r]
B = [sum(compute_local_descriptors(sys, ace)) for sys in system]
ny = 50 # number of samples

# modes of as
figsa = plot_as_modes(asmc, πβ, ny, Vector(r), B; scl=1e2)
fa = plot_eigenvectors_row(asmc.W1)
save("$(simdir)/figures/as_modes.png", figsa)
save("$(simdir)/figures/as_eigenvectors.png", fa)

# pca components
_, _, Wpca = compute_eigenbasis(Matrix(πβ.Σ))
figsb = plot_as_modes(Wpca, πβ, ny, Vector(r), B; scl=1e2)
fb = plot_eigenvectors(Wpca)

# parameter study
figsc = plot_parameter_study(πβ, ny, Vector(r), B; scl=1e2)
fc = plot_eigenvectors(Matrix(I(βdim)))


# plot samples from active subspace --------------------------------------------

# from original sampling density
f2 = plot_pairwise_potential(πβ, 100, Vector(r), B; scl=1e1)
save("$(simdir)/figures/sampling_density.png", f2)


# from active subspace
as = compute_as(∇Qmc, πβ, 3)
ysamp = [rand(as.π_y) for i = 1:100]
θy = [as.W1*y + as.W2*as.π_z.μ for y in ysamp]
plot_pairwise_potential(θy, Vector(r), B)



# plot variation of QoI along directions --------------------------------------------

# compute Monte Carlo estimates of Q 
Qmc = compute_Q(βsamp, idkeep, q, simdir)

# compute IS estimates of Q along other dimensions
# construct IS estimator
Temp = 72; Temp_b = 150.0
Tr = Temp/Temp_b          # temperature
biasdir = "$(simdir)Temp_$(Temp_b)/"

πg = Gibbs(πgibbs, θ=πβ.μ, β=Tr)
Φsamp = JLD.load("$(biasdir)coeff_mean/energy_descriptors.jld")["Bsamp"]
ISint = ISSamples(πg, Φsamp)

nsamp = 2000 # 000

# from original density
@time begin
θsamp = [rand(πβ) for i = 1:nsamp]
Q0 = map(θ -> expectation(θ, q, ISint)[1], θsamp)
end

# from active subspace
@time begin
_, θysamp = sample_as(nsamp, as)
Qas = map(θ -> expectation(θ, q, ISint)[1], θysamp)
end

# from as modes
Qmod = Vector{Vector{Float64}}(undef, βdim)
@time begin
for i = 1:βdim
    println("$i")
    θyi = draw_samples_as_mode(asmc.W1, i, πβ, nsamp)
    Qmod[i] = map(θ -> expectation(θ, q, ISint)[1], θyi)
end
end
# from pca modes
pca = compute_as(Matrix(πβ.Σ), πβ, βdim)
Qpca = Vector{Vector{Float64}}(undef, βdim)
@time begin
for i = 1:βdim
    println("$i")
    θyi = draw_samples_as_mode(pca.W1, i, πβ, nsamp)
    Qpca[i] = map(θ -> expectation(θ, q, ISint)[1], θyi)
end
end

# from each parameter dimension
pid = compute_as(Matrix(I(βdim)), πβ, βdim)
Qpid = Vector{Vector{Float64}}(undef, βdim)
@time begin
for i = 1:βdim
    println("$i")
    θyi = draw_samples_as_mode(pid.W1, i, πβ, nsamp)
    Qpid[i] = map(θ -> expectation(θ, q, ISint)[1], θyi)
end
end



fig1 = Figure(resolution=(300, 550))
ax = Axis(fig1[1,1],
    xlabel="count",
    ylabel="Q",
    title="Distribution of Q",
    xgridcolor=:white,
    ygridcolor=:white)

histbins = LinRange(0.107, 0.110, 100)

hist!(ax, abs.(Q0),
    direction=:x,
    bins = histbins,
    offset=0.0,
    color=(:green, 0.3),
    # strokecolor=:green,
    # strokewidth=2,
    # strokearound=true,
    label="full space")
hist!(ax, abs.(Qas),
    direction=:x,
    bins = histbins,
    offset=0.0,
    color=(:red, 0.3),
    # strokecolor=:red,
    # strokewidth=2,
    # strokearound=true,
    label="active subspace")
axislegend(ax)
fig1


fig = Figure(resolution=(800, 550))
ax = Axis(fig[1,1],
    xlabel="dimension",
    ylabel="Q",
    title="Distribution of Q by dimension",
    xticks=1:8,
    ygridcolor=:white)

histbins = LinRange(0.107, 0.110, 100)

for i in 1:βdim
hist!(ax, abs.(Qmod[i]),
    direction=:x,
    bins = histbins,
    offset=1*i,
    color=(:red, 0.3),
    # strokecolor=:blue,
    # strokewidth=2,
    # strokearound=true,
    scale_to=0.5)
  
hist!(ax, abs.(Qpca[i]),
    direction=:x,
    bins = histbins,
    offset=1*i,
    color=(:blue, 0.3),
    # strokecolor=:blue,
    # strokewidth=2,
    # strokearound=true,
    scale_to=0.5)

hist!(ax, abs.(Qpid[i]),
    direction=:x,
    bins = histbins,
    offset=1*i,
    color=(:green, 0.3),
    # strokecolor=:blue,
    # strokewidth=2,
    # strokearound=true,
    scale_to=0.5)
end

fig