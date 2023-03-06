### potential function and gradients of 1D-state 2D-parameter double well model
using ActiveSubspaces
using Distributions
using LinearAlgebra


# initialize ##################################################################################################

# specify potential function and its gradients
V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
∇θV(x, θ) = [x^2 / 2, -x^4 / 4]

# instantiate Gibbs object
πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)

# define QoI as mean energy V
q = GibbsQoI(h=(x,γ) -> V(x,γ), p=πgibbs)

# define sampling density
μθ = 3*ones(2)
Σθ = 0.5*I(2) 
ρθ = MvNormal(μθ, Σθ) # prior on θ 
ρx0 = Uniform(-2, 2) # for initial state x0

# define constants
nsamp = 1000                # number of θ samples
ngrid = 30                  # number of θ quadrature points in 1D
nMC = 10000                 # number of MC/MCMC samples
nuts = NUTS(1e-1)           # Sampler struct

# θ samples for MC integration 
θsamp = [rand(ρθ) for i = 1:nsamp]
θsamp = JLD.load("θsamp.jld")["θsamp"]

# quadrature points for reference solution
θrng = [μθ[1] - 3.5*sqrt(Σθ[1,1]), μθ[1] + 3.5*sqrt(Σθ[1,1])]
ξθ, wθ = gausslegendre(ngrid, θrng[1], θrng[2]) # 2D quad points over θ
ξx, wx = gausslegendre(200, -10, 10) # 1D quad points over x


# compute active subspace ###################################################################################

# compute reference by quadrature integration
Cref = zeros(2,2)
for i = 1:ngrid
    for j = 1:ngrid
        println("($i, $j)")
        ξij = [ξθ[i], ξθ[j]]
        ∇Qij = grad_expectation(ξij, q; ξ=ξx, w=wx)
        Cref += ∇Qij*∇Qij' * pdf(ρθ, ξij) * wθ[i] * wθ[j]
    end
end
λref, ϕref = select_eigendirections(Cref, 2)


# compute with Monte Carlo 
∇Qmc = map(θ -> grad_expectation(θ, q; n=nMC, sampler=nuts, ρ0=ρx0), θsamp)
λmc, ϕmc = select_eigendirections(∇Qmc, 2)


# compute with importance sampling
# Gibbs biasing distribution
πg = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=0.2, θ=[3,3])
xsamp = rand(πg, nMC, nuts, ρx0) 
∇Qis_g = map(θ -> grad_expectation(θ, q; g=πg, xsamp=xsamp), θsamp)
λis_g, ϕis_g = select_eigendirections(∇Qis_g, 2)

# uniform biasing distribution
πu = Uniform(-5,5)
xsamp = rand(πu, nMC)
∇Qis_u = map(θ -> grad_expectation(θ, q; g=πu, xsamp=xsamp), θsamp)
λis_u, ϕis_u = select_eigendirections(∇Qis_u, 2)
