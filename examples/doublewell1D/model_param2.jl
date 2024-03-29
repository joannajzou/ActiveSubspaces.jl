### symmetric double well potential based on explicit parameterization of well depth and distancec
# ref: https://arxiv.org/abs/1209.2521

# specify potential function and its gradients ###############################################################

# rescaling parameters
a = 0.2
b = 0.2

V(x, θ) = (a*θ[1] / (b*θ[2])^2) * ((x)^2 - (b*θ[2]) )^2 # with neg sign
∇xV(x, θ) = 4 * (a*θ[1] / (b*θ[2])^2) * ((x)^2 - (b*θ[2])) * (x)
∇θV(x, θ) = [ (a / (b*θ[2])^2) * ((x)^2 - (b*θ[2]) )^2,
          - 2*a*b*θ[1] * ( ((x)^2 - (b*θ[2]))^2 / (b*θ[2])^3 + ((x)^2 - (b*θ[2])) / (b*θ[2])^2 ) ]

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
nβ = length(βarr)