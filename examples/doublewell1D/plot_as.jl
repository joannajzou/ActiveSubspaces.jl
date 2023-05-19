using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using CairoMakie

include("doublewell1D.jl")

function init_eig_arrays(d, nsamp_arr)
    λarr = Matrix{Float64}(undef, (d, length(nsamp_arr)))
    Warr = Vector{Matrix{Float64}}(undef, length(nsamp_arr))
    return λarr, Warr
end

function compute_val(nsamp_arr::Vector{Int64}, λref::Vector{Float64}, Cref::Matrix{Float64}, λmc::Matrix{Float64}, Cmc::Dict)
    val = Dict{String, Vector{Float64}}()
    val["λ1_err"] = [(λmc[1,i] - λref[1]) / λref[1] for i = 1:length(nsamp_arr)]
    val["λ2_err"] = [(λmc[2,i] - λref[2]) / λref[2] for i = 1:length(nsamp_arr)]
    val["SS_err"] = [ForstnerDistance(Cref, Cmc[nsamp]) for nsamp in nsamp_arr]
    return val
end

# load data ##############################################################################
Cref = JLD.load("data/DW1D_Ref.jld")["Cref"]

nsamp_arr = [1500, 3000, 6000, 12000]
nrepl_arr = [10, 1, 1, 1]

Cmc = Dict{Int64, Matrix{Float64}}()
Cis_u = Dict{Int64, Matrix{Float64}}()
Cis_g1 = Dict{Int64, Matrix{Float64}}()
Cis_g2 = Dict{Int64, Matrix{Float64}}()
Cis_g3 = Dict{Int64, Matrix{Float64}}()

for i = 1:length(nsamp_arr)
    nsamp = nsamp_arr[i]
    nrepl = nrepl_arr[i]

    Cmc[nsamp] = JLD.load("data/DW1D_MC_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][1]
    Cis_u[nsamp]= JLD.load("data/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][1]
    Cis_g1[nsamp]= JLD.load("data/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][0.01][1]
    Cis_g2[nsamp]= JLD.load("data/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][0.2][1]
    Cis_g3[nsamp]= JLD.load("data/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["C"][1.0][1]
end

nsamp = 3000; nrepl = 1
metrics_u = JLD.load("data/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
metrics_g = JLD.load("data/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]

πu = JLD.load("data/DW1D_ISU_nsamp={$nsamp}_nrepl={$nrepl}.jld")["π_bias"]
ll = πu.a; ul = πu.b
βarr = JLD.load("data/DW1D_ISG_nsamp={$nsamp}_nrepl={$nrepl}.jld")["βarr"]
d = 2



# compute eigendecomposition #####################################################
println(" ====== Evaluate error ======")

# reference
_, λref, Wref = select_eigendirections(Cref, d)

# MC estimates
λmc, Wmc = init_eig_arrays(d, nsamp_arr)
λu, Wu = init_eig_arrays(d, nsamp_arr)
λg1, Wg1 = init_eig_arrays(d, nsamp_arr)
λg2, Wg2 = init_eig_arrays(d, nsamp_arr)
λg3, Wg3 = init_eig_arrays(d, nsamp_arr)

for i = 1:length(nsamp_arr)
    nsamp = nsamp_arr[i]
    _, λmc[:,i], Wmc[i] = select_eigendirections(Cmc[nsamp], d)
    _, λu[:,i], Wu[i] = select_eigendirections(Cis_u[nsamp], d)
    _, λg1[:,i], Wg1[i] = select_eigendirections(Cis_g1[nsamp], d)
    _, λg2[:,i], Wg2[i] = select_eigendirections(Cis_g2[nsamp], d)
    _, λg3[:,i], Wg3[i] = select_eigendirections(Cis_g3[nsamp], d)
end



# compute error metrics ##########################################################
val_mc = compute_val(nsamp_arr, λref, Cref, λmc, Cmc)
val_u = compute_val(nsamp_arr, λref, Cref, λu, Cis_u)
val_g1 = compute_val(nsamp_arr, λref, Cref, λg1, Cis_g1)
val_g2 = compute_val(nsamp_arr, λref, Cref, λg1, Cis_g2)
val_g3 = compute_val(nsamp_arr, λref, Cref, λg1, Cis_g3)


# plot error metrics #############################################################

function plot_val_metric(val_type::String, nsamp_arr::Vector{Int64}, val_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String)
    fig = Figure(resolution=(800, 500))
    ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel=ylab,
        title=ttl, xscale=log10, yscale=log10)   
    for (val_i, lab_i) in zip(val_tup, lab_tup)
        scatterlines!(ax, nsamp_arr, abs.(val_i[val_type]), label=lab_i)
    end
    axislegend(ax)
    fig
end

f1 = plot_val_metric("λ1_err", nsamp_arr,
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "RE(λ - λ̂)", 
                "Rel. error in eigenvalue λ1(C)"
)
f2 = plot_val_metric("λ2_err", nsamp_arr,
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "RE(λ - λ̂)", 
                "Rel. error in eigenvalue λ2(C)"
)
f3 = plot_val_metric("SS_err", nsamp_arr,
                (val_mc, val_u, val_g1, val_g2, val_g3),
                ("MC","IS, U[-5, 5]", "IS, Gibbs (β=0.01)", "IS, Gibbs (β=0.2)", "IS, Gibbs (β=1.0)"),
                "d(C, Ĉ)", 
                "Forstner distance from reference covariance matrix"
)



# plots ##################################################################################
col=[:black, :purple, :blue, :green, :orange, :red]         # set color scheme
xplot = LinRange(ll, ul, 1000)


# plot biasing distributions
fig = Figure(resolution = (700, 600))
ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)",
          title="Importance sampling biasing distributions")

lines!(ax, xplot, UniformPDF.(xplot), color=col[2], label="U[$ll, $ul]")
for i = 1:nβ
    βi = β_arr[i]
    dist = Gibbs(βi, θ_fix, V, ∇xV, ∇θV, wz, ξz)
    lines!(ax, xplot, pdf.(dist, xplot) ./ normconst(dist), color=col[i+2], label="Gibbs, β=$βi")
end
axislegend(ax)
fig


# plot distribution of samples
θ1rng = LinRange(0.5, 5.5, 100); θ2rng = θ1rng              # grid across θ-domain
c_mat = [pdf(π_θ, [θi, θj]) for θi in θ1rng, θj in θ2rng]   # PDF values                                  

fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],  xlabel="θ1", ylabel="θ2",
          title="Distribution of samples (nsamp=$nsamp)")

scatter!(θmat[1,:], θmat[2,:], color=(:blue, 0.05))
contour!(ax, θ1rng, θ2rng, c_mat, levels=0:0.1:1)
fig


# plot Gibbs distribution of samples
fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)",
          title="Gibbs distribution of samples")
for θi in θsamp
    dist = Gibbs(1.0, θi, V, ∇xV, ∇θV, wz, ξz)
    lines!(ax, xplot, pdf.(dist, xplot) ./ normconst(dist), color=(:steelblue, 1))
end
fig


# boxplot of L2 norm error in ∇Q
fig = Figure(resolution=(1000, 600))
ax = Axis(fig[1,1], ylabel="||∇Qref - ∇Q||₂",
          xticks=(1:5, ["MC", "IS, U[$ll, $ul]", "IS, π(β=0.01)", "IS, π(β=0.2)", "IS, π(β=1.0)"]),
          title="Norm error in ∇Q", yscale=log10)

boxplot!(ax, 1 * ones(nsamp), errMC, color=(col[1], 0.5))
boxplot!(ax, 2 * ones(nsamp), erru, color=(col[2], 0.5))
for i = 1:nβ
    βi = β_arr[i]
    boxplot!(ax, (i+2) * ones(nsamp), errIS_arr[i], color=(col[i+2], 0.5))
end
fig


# mean norm error in eigenvalues
fig = Figure(resolution=(800, 1000))
ax = Vector{Axis}(undef, 2)
for k = 1:2 # two eigenvalues
    ax[k] = Axis(fig[k,1], xlabel="number of samples (n)", ylabel="RE(λ - λ̂)",
          title="Rel. error in eigenvalue λ$k(C)", xscale=log10, yscale=log10)      
    scatterlines!(ax[k], nboot, [abs.(λMC[j][k]) for j = 1:length(nboot)], color=col[1], label="MC")
    scatterlines!(ax[k], nboot, [abs.(λu[j][k]) for j = 1:length(nboot)], color=col[2], label="IS, U[$ll, $ul]")

    for i = 1:nβ
        βi = β_arr[i]
        scatterlines!(ax[k], nboot, [abs.(λIS[i][j][k]) for j = 1:length(nboot)], color=col[i+2], label="IS, π(β=$βi)")
    end
end
axislegend(ax[1], position=:lb, framevisible=false)
fig


# mean Forstner distance
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel="mean d(C, Ĉ)",
          title="Mean Forstner distance of covariance matrix C (over Nboot=$J)",
          xscale=log10, yscale=log10)

scatterlines!(ax, nboot, dMC, color=col[1], label="MC")
scatterlines!(ax, nboot, du, color=col[2], label="IS, U[$ll, $ul]")
for i = 1:nβ
    βi = β_arr[i]
    scatterlines!(ax, nboot, dIS[i], color=col[i+2], label="IS, π(β=$βi)")
end
axislegend(ax, position=(0.1, 0.6), framevisible=false)
fig


# variance in Forstner distance
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel="var(d(C, Ĉ))",
          title="Variance in Forstner distance of covariance matrix C (over Nboot=$J)",
          xscale=log10, yscale=log10)

scatterlines!(ax, nboot, vMC, color=col[1], label="MC")
scatterlines!(ax, nboot, vu, color=col[2], label="IS, U[$ll, $ul]")
for i = 1:nβ
    βi = β_arr[i]
    scatterlines!(ax, nboot, vIS[i], color=col[i+2], label="IS, π(β=$βi)")
end
axislegend(ax, position=(0.9, 0.75), framevisible=false)
fig



# append uniform dist. to array
β_arr = reduce(vcat, (0, β_arr))
wvar_arr = reduce(vcat, ([wvar_u], wvar_arr))
wess_arr = reduce(vcat, ([wess_u], wess_arr))
hess_arr = reduce(vcat, ([hess_u], hess_arr))

# distribution of log variance of IS weights
logwvar_arr = [log.(wvar) for wvar in wvar_arr]
logwvar_lab = ["log(Var(w))", "Log variance of IS weights", "log(Var(w)) vs θ1", "log(Var(w)) vs θ2"]
fig1 = scatter3D_plot(β_arr, logwvar_arr, logwvar_lab, logscl=false)

# distribution of ESS
wess_lab = ["ESS/n", "ESS (% of total sample size)", "ESS/n vs θ1", "ESS/n vs θ2"];
fig2 = scatter3D_plot(β_arr, wess_arr, wess_lab, n=nMC)

# distribution of modified ESS
hess_lab = ["mod. ESS/n", "Mod. ESS (% of total sample size)", "Mod. ESS/n vs θ1", "Mod. ESS/n vs θ2"];
fig3 = scatter3D_plot(β_arr, hess_arr, hess_lab, n=nMC, logscl=true)


# plot realizations of Gibbs distribution 
# fig1b = Gibbdist_plot(β_arr, logwvar_arr, "Gibbs dist. of θ samples corresp. to low variance of IS wts.", rev=false, nsamp=100)
# fig2b = Gibbdist_plot(β_arr, wess_arr, "Gibbs dist. of θ samples corresp. to high ESS/n", rev=true, nsamp=100)
# fig3b = Gibbdist_plot(β_arr, hess_arr, "Gibbs dist. of θ samples corresp. to high mod. ESS/n", rev=true, nsamp=100)

th = exp(-17)
fig1b = Gibbdist_plot(β_arr, wvar_arr, "Gibbs dist. of θ samples corresp. to low variance of IS wts. (<= $th)", rev=false, thresh=th)
th = Int(0.4*nMC)
fig2b = Gibbdist_plot(β_arr, wess_arr, "Gibbs dist. of θ samples corresp. to low ESS (> $th)", rev=true, thresh=th)
th = Int(round(exp(-7)*nMC))
fig3b = Gibbdist_plot(β_arr, hess_arr, "Gibbs dist. of θ samples corresp. to low mod. ESS (> $th)", rev=true, thresh=th)


th = round(exp(-12), digits=6)
fig1c = Gibbdist_plot(β_arr, wvar_arr, "Gibbs dist. of θ samples corresp. to high variance of IS wts. (> $th)", rev=true, thresh=th)
th = Int(0.4*nMC)
fig2c = Gibbdist_plot(β_arr, wess_arr, "Gibbs dist. of θ samples corresp. to low ESS (<= $th)", rev=false, thresh=th)
th = Int(round(exp(-7)*nMC))
fig3c = Gibbdist_plot(β_arr, hess_arr, "Gibbs dist. of θ samples corresp. to low mod. ESS (<= $th)", rev=false, thresh=th)



## plot variation with q
nMC = 10000                                 # number of MC/MCMC samples
eps = 1e-1                                  # step size 
nuts = NUTS(eps)    

Qmc(θ) = Q(θ, nuts, nsamp=nMC)