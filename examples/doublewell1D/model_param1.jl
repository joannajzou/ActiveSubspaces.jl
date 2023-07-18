### symmetric double well potential based on quartic model

# specify potential function and its gradients ###############################################################
V(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
∇xV(x, θ) = - θ[1] * x + θ[2] * x.^3
∇θV(x, θ) = [-x^2 / 2, x^4 / 4]

# instantiate Gibbs object
πgibbs0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
πgibbs = Gibbs(πgibbs0, β=1.0)

# define sampling density for θ
d = 2
μθ = 4*ones(d) 
Σθ = 0.2*I(d) 
ρθ = MvNormal(μθ, Σθ)                           # prior on θ 

# define x-domain
ρx0 = Uniform(-2, 2)                            # sampling density for initial state x0
ll = -3; ul = 3                                 # lower and upper limits
xplot = Vector(LinRange(ll, ul, 1000))          # plot domain


# define integrators #########################################################################################

# QuadIntegrator
ngrid = 100                                     # number of θ quadrature points in 1D
ξx, wx = gausslegendre(200, -10, 10)            # 1D quad points over x
GQint = GaussQuadrature(ξx, wx)

ξθ = Vector{Vector{Float64}}(undef, d)          # 2D quad points over θ
wθ = Vector{Vector{Float64}}(undef, d)
for i = 1:d
    θrng = [ρθ.μ[i] - 3.5*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3.5*sqrt(ρθ.Σ[i,i])]
    ξθ[i], wθ[i] = gausslegendre(ngrid, θrng[1], θrng[2]) # 2D quad points over θ
end

# MCIntegrator
nMC = 10000                                     # number of MC/MCMC samples
nuts = NUTS(1e-1)                               # Sampler struct
MCint = MCMC(nMC, nuts, ρx0)

# ISIntegrator
πu = Uniform(ll, ul)                            # importance sampling: uniform distribution
βarr = [0.02, 0.2, 1.0]                         # importance sampling: temp parameter for Gibbs biasing dist.
πg1 = Gibbs(πgibbs0, β=βarr[1], θ=ρθ.μ)
πg2 = Gibbs(πgibbs0, β=βarr[2], θ=ρθ.μ)
πg3 = Gibbs(πgibbs0, β=βarr[3], θ=ρθ.μ)
nβ = length(βarr)