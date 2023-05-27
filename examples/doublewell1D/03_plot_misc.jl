
# misc. plots ###################################################################
col=[:black, :purple, :blue, :green, :orange, :red]         # set color scheme
xplot = LinRange(ll, ul, 1000)


# plot biasing distributions
fig = Figure(resolution = (700, 600))
ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)",
          title="Importance sampling biasing distributions")

lines!(ax, xplot, pdf.(πu, xplot), color=col[2], label="U[$ll, $ul]")
for i = 1:length(βarr)
    βi = βarr[i]
    πg = Gibbs(πgibbs0, β=βi, θ=[3,3])
    lines!(ax, xplot, updf.(πg, xplot) ./ normconst(πg, ξx, wx), color=col[i+2], label="Gibbs, β=$βi")
end
axislegend(ax)
fig


# plot distribution of samples
θ1rng = LinRange(0.5, 5.5, 100); θ2rng = θ1rng              # grid across θ-domain
c_mat = [pdf(ρθ, [θi, θj]) for θi in θ1rng, θj in θ2rng]   # PDF values                                  

fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1],  xlabel="θ1", ylabel="θ2",
          title="Distribution of samples (nsamp=$nsamp)")

scatter!(θmat[1,:], θmat[2,:], color=(:blue, 0.1))
contour!(ax, θ1rng, θ2rng, c_mat, levels=0:0.1:1)
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




## plot variation with q
nMC = 10000                                 # number of MC/MCMC samples
eps = 1e-1                                  # step size 
nuts = NUTS(eps)    

Qmc(θ) = Q(θ, nuts, nsamp=nMC)