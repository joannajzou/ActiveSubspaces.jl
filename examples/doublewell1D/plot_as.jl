using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD
using CairoMakie


# initialize ##################################################################################################

# specify potential function and its gradients
V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
∇θV(x, θ) = [x^2 / 2, -x^4 / 4]

# instantiate Gibbs object
πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)

# define QoI as mean energy V
q = GibbsQoI(h=V, p=πgibbs)
q = GibbsQoI(h=(x,γ) -> V(x,γ), p=πgibbs)

# define sampling density
μθ = 3*ones(2)
Σθ = 0.5*I(2) 
ρθ = MvNormal(μθ, Σθ) # prior on θ 
ρx0 = Uniform(-2, 2) # for initial state x0

# define constants
nsamp = 10                  # number of θ samples
ngrid = 100                 # number of θ quadrature points in 1D
nMC = 10000                 # number of MC/MCMC samples
nuts = NUTS(1e-1)           # Sampler struct

# θ samples for MC integration 
θsamp = [rand(ρθ) for i = 1:nsamp]
θsamp = JLD.load("θsamp.jld")["θsamp"]

# quadrature points for reference solution
θrng = [μθ[1] - 3.5*sqrt(Σθ[1,1]), μθ[1] + 3.5*sqrt(Σθ[1,1])]
ξθ, wθ = gausslegendre(ngrid, θrng[1], θrng[2]) # 2D quad points over θ
ξx, wx = gausslegendre(200, -10, 10) # 1D quad points over x

# importance sampling: temp parameter for Gibbs biasing dist.
βarr = [0.01, 2.0, 1.0]
nβ = length(βarr)



# load data ##########################################################################################
D = JLD.load("data/DW1D_Trial1_n=$nsamp.jld")


# compute active subspace ############################################################################
