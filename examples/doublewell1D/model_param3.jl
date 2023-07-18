### asymmetric double well potential with additional cubic term
# ref: https://link.springer.com/article/10.1007/s10910-022-01328-9

# specify potential function and its gradients ###############################################################

V(x, θ) = exp(θ[1])*x^2 - exp(θ[2])*x^3 + exp(θ[3])*x^4
∇xV(x, θ) = 2*exp(θ[1])*x - 3*exp(θ[2])*x^2 + 4*exp(θ[3])*x^4
∇θV(x, θ) = [x^2, -x^3, x^4]


# instantiate Gibbs object
πgibbs0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
πgibbs = Gibbs(πgibbs0, β=1.0)

# define sampling density for θ
d = 3
μθ = [log(9), log(17.75), log(9)] # [log(16), log(24), log(9)]
Σθ = 5e-4*I(d)
ρθ = MvNormal(μθ, Σθ)                           # prior on θ 

# define x-domain
ρx0 = Uniform(-0.2, 1.2)                        # sampling density for initial state x0
ll = -2; ul = 3                                 # lower and upper limits
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
βarr = [0.005, 0.5, 1.0]                         # importance sampling: temp parameter for Gibbs biasing dist.
nβ = length(βarr)

