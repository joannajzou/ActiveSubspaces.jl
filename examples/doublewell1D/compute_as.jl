### potential function and gradients of 1D-state 2D-parameter double well model
using ActiveSubspaces
using Distributions
using LinearAlgebra
using JLD


function initialize_IS_arrays(nsamp)
    ∇Q_arr = Vector{Vector{Float64}}(undef, nsamp)
    ∇h_arr = Vector{Vector}(undef, nsamp)
    w_arr = Vector{Vector{Float64}}(undef, nsamp)
    return ∇Q_arr, ∇h_arr, w_arr
end


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



# compute gradient of QoI #################################################################################

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

# compute with Monte Carlo 
∇Qmc = map(θ -> grad_expectation(θ, q; n=nMC, sampler=nuts, ρ0=ρx0), θsamp)


# compute with importance sampling
# uniform biasing distribution
∇Qis_u, his_u, wis_u = initialize_IS_arrays(nsamp)
πu = Uniform(-5,5)
xsamp = rand(πu, nMC)
for k = 1:nsamp
    ∇Qis_u[k], his_u[k], wis_u[k] = grad_expectation(θsamp[k], q; g=πu, n=nMC)  # xsamp=xsamp
end


# Gibbs biasing distribution
∇Qis_g = Dict{Float64, Vector}() 
his_g = Dict{Float64, Vector}() 
wis_g = Dict{Float64, Vector}() 

for i = 1:nβ
    βi = βarr[i]    
    ∇Qis_g[βi], his_g[βi], wis_g[βi] = initialize_IS_arrays(nsamp)
    πg = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=βi, θ=[3,3])
    xsamp = rand(πg, nMC, nuts, ρx0) 

    for k = 1:nsamp
        ∇Qis_g[βi][k], his_g[βi][k], wis_g[βi][k] = grad_expectation(θsamp[k], q; g=πg, xsamp=xsamp)
    end
end



# save data ###############################################################################################

JLD.save("data/DW1D_Trial1_n=$nsamp.jld",
    "Cref", Cref,
    "θsamp", θsamp,
    "∇Qmc", ∇Qmc,
    "∇Qis_u", ∇Qis_u,
    "his_u", his_u,
    "wis_u", wis_u,
    "∇Qis_g", ∇Qis_g,
    "his_g", his_g,
    "wis_g", wis_g,
)


# λref, ϕref = select_eigendirections(Cref, 2)
# λmc, ϕmc = select_eigendirections(∇Qmc, 2)
# λis_g, ϕis_g = select_eigendirections(∇Qis_g, 2)
# λis_u, ϕis_u = select_eigendirections(∇Qis_u, 2)