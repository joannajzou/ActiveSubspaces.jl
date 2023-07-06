include("00_spec_model.jl")
include("qoi_meanenergy.jl")
include("mc_utils.jl")
include("molecular_plot_utils.jl")
include("plotting_utils.jl")

using DelimitedFiles


# load data ##################################################################################

# coeff sampling distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)
βdim = length(μ)

# coeff samples
βsamp = JLD.load("$(simdir)coeff_samples.jld")["βsamp"]
nsamp = length(βsamp)

# skipped ids
idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]
idkeep = symdiff(1:nsamp, idskip)
βsamp_2 = βsamp[idkeep]

# load MC estimates
∇Qmc = [JLD.load("$(simdir)coeff_$j/gradQ_meanenergy.jld")["∇Q"] for j in idkeep]

# load IS estimates
∇Qis = JLD.load("$(simdir)gradQ_meanenergy_IS_all_84.jld")["∇Q"]
metrics = JLD.load("$(simdir)metrics_84.jld")["metrics"]



# compute active subspace ##################################################################
asmc = compute_as(∇Qmc, ρθ, βdim)

asis = compute_as(∇Qis, ρθ, βdim)



# compare subspaces ########################################################################
with_theme(custom_theme) do
    f1 = plot_eigenspectrum(asmc.λ)
end
f1 = plot_eigenspectrum([asmc.λ, asis.λ]; predlab=["MC", "IS"])
# f1 = plot_eigenspectrum([asmc.λ ./ maximum(asmc.λ), asis.λ ./ maximum(asis.λ) ]; predlab=["MC", "IS"])
f2, wsd = plot_wsd(asmc.C, asis.C)
f3, cossim = plot_cossim(asmc.C, asis.C)


# check IS diagnostics #####################################################################
# remove NaN values
id1 = findall(x -> isnan(x), metrics["wvar"])
id2 = findall(x -> isnan(x), metrics["wESS"])
id = union(id1, id2)
metrics_2 = Dict{String, Vector}()
for k in collect(keys(metrics))
    metrics_2[k] = metrics[k][Not(id)]
end

# transform samples into 2D
_, asis.λ, Wis = select_eigendirections(asis.C, 2)
θsamp = metrics_2["θsamp"]
ysamp = (Wis',) .* θsamp
ymat = reduce(hcat, ysamp)

met_types = ["wvar", "wESS"]
met_ttl = ["Var(w)", "ESS(w)"]
nmet = 2

# plot diagnostic value vs. parameter value
fig = Figure(resolution=(550*nmet, 500))
ax = Vector{Axis}(undef, nmet)
for (i, met) in zip(1:nmet, met_types)
    ax[i] = Axis(fig[1,i], xlabel="y_1", ylabel="y_2", title=met_ttl[i])
    sc = scatter!(ax[i], ymat[1,:], ymat[2,:], markersize=7, color=metrics_2[met]) # colormap=Reverse(:Blues)) 
    # Colorbar(fig[1,i][1,2], sc)
end
fig

# plot samples with high/low diagnostic values #####################################################################
# Define 2D points
r = 1.0:0.01:5 # 0.99:0.01:2.5
box = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
system = [FlexibleSystem([AtomsBase.Atom(:Ar, [0.0, 0.0, 0.0] * u"Å"), AtomsBase.Atom(:Ar, [ri, 0.0, 0.0] * u"Å")],
                        box * u"Å", bcs) for ri in r]


B = [sum(compute_local_descriptors(sys, ace)) for sys in system]
energies_pred = [Bi' * βi for Bi in B, βi in βsamp]
energies_mean = [Bi' * πβ.μ for Bi in B]

fig = Figure(resolution=(1100, 500))
ax = Vector{Axis}(undef, nmet)
for (i, met) in zip(1:nmet, met_types)
    ax[i] = Axis(fig[1,i], xlabel="r (Å)", ylabel="E (eV)", title=met_ttl[i])
    sortid = sortperm(metrics[met], rev=true) # descending order 

    hi_err_id = sortid[1:50]
    lo_err_id = sortid[end-50:end-1]
    [lines!(ax[i], r, energies_pred[:,j], color=(:red, 0.2)) for j in hi_err_id]
    [lines!(ax[i], r, energies_pred[:,j], color=(:Green, 0.2)) for j in lo_err_id]
    lines!(ax[i], r, energies_mean, linewidth=2, color=:black) 
end
fig


# plot samples which didn't run #####################################################################
βskip = βsamp[idskip]
energies_skip = [Bi' * βi for Bi in B, βi in βskip]

fig = Figure(resolution=(550, 500))
ax = Axis(fig[1,1], xlabel="r (Å)", ylabel="E (eV)")
[lines!(ax, r, energies_skip[:,j], color=(:grey, 0.2)) for j in 1:length(idskip)]
# [lines!(ax, r, energies_pred[:,j], color=(:cyan, 0.2)) for j in 1:length(idkeep)]
fig


# plot samples from the active subspace #####################################################################
rdim = 8
Wmc1, Wmc2, πmc_y, πmc_z = compute_as(Wmc, rdim, πβ)
# Wis1, Wis2, πis_y, πis_z = compute_as(Wis, rdim, πβ)

ny = 100
fa = sample_as(Wmc, rdim, πβ, ny, B)
fb = sample_as(Wis, rdim, πβ, ny, B)
f = sample_as((Wmc, Wis), rdim, πβ, ny, B, ("AS from MC", "AS from IS"))


# plot mode shapes of pairwise potential ############################################
ny = 50
with_theme(custom_theme) do
    figs = plot_as_modes(Wmc, πβ, ny, Vector(r), B)
end
figs = plot_parameter_study(πβ, ny, Vector(r), B)

# pca
_, λpca, Wpca = select_eigendirections(Matrix(πβ.Σ), βdim)
figs = plot_as_modes(Wpca, πβ, ny, Vector(r), B)

# plot eigenvectors
fig = Figure(resolution=(600,1000))
ax = Axis(fig[1,1], xlabel="dim.", ylabel="eigenvector i", xticks = 1:βdim, yticks=(2.5:2.5:2.5*βdim, string.(βdim:-1:1)), title="Eigenvectors")
for i = 1:βdim
    scatterlines!(ax, 1:βdim, Wmc[:,i] .+ 2.5*(βdim+1-i), color=RGB(0, i/βdim, 0), label="ϕ_$i")
end
# axislegend(ax, position=:lt)
fig

# plot eigenvectors
plot_eigenvectors(Wmc)
plot_eigenvectors(Wpca)
plot_eigenvectors(Matrix(W))



## compute QoI with active subspace ##########################################

# define IS integrator
nsamp = 10000
πg = Gibbs(πgibbs, θ=πβ.μ)
Φsamp = JLD.load("$(simdir)coeff_nom/global_energy_descriptors.jld")["Φsamp"]
ISint = ISSamples(πg, Φsamp)

πβ2 = MvNormal(πβ.μ, πβ.Σ ./ 1e5)

# original sampling density
θsamp = [rand(πβ2) for i = 1:n]
Q0 = zeros(nsamp)
@elapsed begin
for k = 1:nsamp
    Q0[k], _, _ = expectation(θsamp[k], q, ISint)
end
end

# from active subspace
θy1 = draw_samples_as_mode(Wmc, 1, πβ2, nsamp)
Qy1 = zeros(nsamp)
for k = 1:nsamp
    Qy1[k], _, _ = expectation(θy1[k], q, ISint)
end

θy8 = draw_samples_as_mode(Wmc, 8, πβ2, nsamp)
Qy8 = zeros(nsamp)
for k = 1:nsamp
    Qy8[k], _, _ = expectation(θy8[k], q, ISint)
end

# plot hist
with_theme(custom_theme) do
    fig = Figure(resolution=(800, 550))
    # ax = Axis(fig[1,1], ylabel="ESS/n", xlabel="descriptor dimension", xticks=1:βdim, title="Histogram of ESS/n")
    # [hist!(ax, ESS[i,:], scale_to=-0.75, color=(:red, 0.5), offset=i, bins=10, direction=:x) for i = 1:βdim]
    ax = Axis(fig[1,1], xlabel="log(Q)", ylabel="count", title="Distribution of Q")
    hist!(ax, log.(abs.(Q0)), bins=LinRange(log(0.0001), log(3000), 25), color=(:blue, 0.2), label="original sampling density")
    hist!(ax, log.(abs.(Qy1)), bins=LinRange(log(0.0001), log(3000), 25), color=(:red, 0.2), label="active subspace")
    hist!(ax, log.(abs.(Qy8)), bins=LinRange(log(0.0001), log(3000), 25), color=(:green, 0.2), label="inactive subspace")
    axislegend(ax, position=:rt)
    fig
end


index = reduce(vcat, (ones(nsamp), 2*ones(nsamp), 3*ones(nsamp)))
Qall = reduce(vcat, (Q0, Qy1, Qy8))

fig = Figure(resolution=(800, 500))
# ax = Axis(fig[1,1], ylabel="ESS/n", xlabel="descriptor dimension", xticks=1:βdim, title="Histogram of ESS/n")
# [hist!(ax, ESS[i,:], scale_to=-0.75, color=(:red, 0.5), offset=i, bins=10, direction=:x) for i = 1:βdim]
ax = Axis(fig[1,1], ylabel="Q", yscale=log10, title="Distribution of Q")
boxplot!(ax, index, Qall .+ 1e-10)
fig








function draw_samples_as_mode(W::Matrix, ind::Int64, πβ::Distribution, ny::Int64)
    dims = Vector(1:size(W,1))
    W1 = W[:,ind:ind] # type as matrix
    W2 = W[:, dims[Not(ind)]]

        # sampling density of inactive variable z
    μz = W2' * πβ.μ
    Σz = Hermitian(W2' * πβ.Σ * W2) # + 1e-10 * I(βdim-rdim)
    π_z = MvNormal(μz, Σz)

    # compute sampling density
    μy = W1' * πβ.μ
    Σy = Hermitian(W1' * πβ.Σ * W1)
    π_y = MvNormal(μy, Σy)
    ysamp = [rand(π_y) for i = 1:ny]
    θy = [W1*y + W2*π_z.μ for y in ysamp]

    return θy
end




# plot Masmc.C diagnostics ############################################################
ds = load_data("$(simdir)coeff_nom/data.xyz", ExtXYZ(u"eV", u"Å"))
e_descr = compute_local_descriptors(ds, ace)
Φsamp = sum.(get_values.(e_descr))
Φ = reduce(hcat, Φsamp)
# [ustrip(get_positions(d)) for d in ds]
Masmc.C_traceplot(Φ)
Masmc.C_autocorrplot(Φ)

ESS = compute_masmc.C_ess(βsamp_2[301:400], idkeep[301:400], simdir)
JLD.save("ESS_301-400.jld", "ESS", ESS)

ess1 = JLD.load("ESS_1-100.jld")["ESS"]
ess2 = JLD.load("ESS_101-200.jld")["ESS"]
ess3 = JLD.load("ESS_201-300.jld")["ESS"]
ess4 = JLD.load("ESS_301-400.jld")["ESS"]

ESS = reduce(hcat, [ess1, ess2, ess3, ess4])

# plot ESS hist
fig = Figure(resolution=(800, 500))
# ax = Axis(fig[1,1], ylabel="ESS/n", xlabel="descriptor dimension", xticks=1:βdim, title="Histogram of ESS/n")
# [hist!(ax, ESS[i,:], scale_to=-0.75, color=(:red, 0.5), offset=i, bins=10, direction=:x) for i = 1:βdim]
ax = Axis(fig[1,1], xlabel="ESS/n", ylabel="count", title="Histogram of mean(ESS/n)")
hist!(ax, mean(abs.(ESS), dims=1)[:], bins=20)
fig



# plot projection onto active subspace ##################################################

fig = Figure(resolution=(600,600))
ax = Axis(fig[1,1], ylabel="||Pr * ei ||₂", xlabel="ei", xticks=1:10, limits=(0, 10, -0.05, 1.05))

for tr = 1:βdim
    Wtrunc = Wmc[:, 1:tr]
    Proj = Wtrunc * Wtrunc'

    infl = zeros(βdim)
    for i = 1:βdim
        ei = zeros(βdim); ei[i] = 1
        infl[i] = norm(Proj * ei)
    end

    scatterlines!(ax, 1:βdim, infl, color=RGB(0, tr/βdim, 0), label="t = $tr")
end

axislegend(ax, position=:rb)
fig









########## ##### ##### ##### ##### ##### ##### ##### 



function compute_masmc.C_ess(βsamp::Vector, ids::Vector, simdir::String)
    nβ = length(βsamp[1])
    ESS = Matrix{Float64}(undef, (nβ, length(βsamp)))

    for (i, id) in zip(1:length(βsamp), ids)
        println("sample $id")
        ds = load_data("$(simdir)coeff_$id/data.xyz", ExtXYZ(u"eV", u"Å"))
        e_descr = compute_local_descriptors(ds, ace)
        Φsamp = sum.(get_values.(e_descr))
        Φ = reduce(hcat, Φsamp)
        n = length(Φsamp)
        autocorr = [1 + 2*sum(autocor(Φ[j,:], 0:1000)) for j = 1:nβ]
        ESS[:,i] = 1 ./ autocorr
    end
    return minimum(ESS)
end


writedlm( "$(simdir)Cmatrix.csv",  asmc.C, ',')

