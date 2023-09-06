### asymmetric double well potential with additional cubic term - 2D version
# ref: https://link.springer.com/article/10.1007/s10910-022-01328-9

# specify potential function and its gradients ###############################################################

V(x, θ) = 9*x^2 - exp(θ[1])*x^3 + exp(θ[2])*x^4
∇xV(x, θ) = 18*x - 3*exp(θ[1])*x^2 + 4*exp(θ[2])*x^4
∇θV(x, θ) = [-exp(θ[1]) * x^3, exp(θ[2]) * x^4]


# instantiate Gibbs object
πgibbs0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
πgibbs1 = Gibbs(πgibbs0, β=1.0)
πgibbs2 = Gibbs(πgibbs0, β=0.5)

# define sampling density for θ
d = 2
μθ = [log(18), log(9)] # [log(16), log(24), log(9)]
Σθ = 1e-3 * I(d) # Diagonal([2e-3, 5e-3]) 
ρθ = MvNormal(μθ, Σθ)                           # prior on θ 

# define x-domain
ρx0 = Uniform(0, 1.5)                        # sampling density for initial state x0
ll = -2; ul = 3                                 # lower and upper limits
xplot = Vector(LinRange(ll, ul, 1000))          # plot domain


# define integrators #########################################################################################

# QuadIntegrator
ngrid = 200                                     # number of θ quadrature points in 1D
ξx, wx = gausslegendre(200, -9, 11)            # 1D quad points over x
GQint = GaussQuadrature(ξx, wx)

θbd = Vector{Vector{Float64}}(undef, d)
ξθ = Vector{Vector{Float64}}(undef, d)          # 2D quad points over θ
wθ = Vector{Vector{Float64}}(undef, d)
for i = 1:d
    θbd[i] = [ρθ.μ[i] - 3.5*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3.5*sqrt(ρθ.Σ[i,i])]
    ξθ[i], wθ[i] = gausslegendre(ngrid, θbd[i][1], θbd[i][2]) # 2D quad points over θ
end

# MCIntegrator
nMC = 20000                                     # number of MC/MCMC samples
nuts = NUTS(0.02)                               # Sampler struct
MCint = MCMC(nMC, nuts, ρx0)

# ISIntegrator
πu = Uniform(ll, ul)                            # importance sampling: uniform distribution
βarr = [0.5]                         # importance sampling: temp parameter for Gibbs biasing dist.
nβ = length(βarr)


# modify pdf and logpdf functions ##########################################################################
import Distributions: pdf, logpdf
import ActiveSubspaces: Gibbs

function pdf(dist::MixtureModel{Univariate, Continuous, Gibbs}, x)
    return pdf(dist, x, GQint)
end
function logpdf(dist::MixtureModel{Univariate, Continuous, Gibbs}, x)
    return logpdf(dist, x, GQint)
end
# function pdf.(dist::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, xsamp)
#     return pdf.((dist,), xsamp, (GQint,))
# end
# function logpdf.(dist::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}, xsamp)
#     logpdf.((dist,), xsamp, (GQint,))
# end
