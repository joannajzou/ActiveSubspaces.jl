# specify potential function and its gradients
V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
∇θV(x, θ) = [x^2 / 2, -x^4 / 4]

# instantiate Gibbs object
πgibbs0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
πgibbs = Gibbs(πgibbs0, β=1.0)

# define sampling density
μθ = 3*ones(2)
Σθ = 0.5*I(2) 
ρθ = MvNormal(μθ, Σθ) # prior on θ 
θrng = [μθ[1] - 3.5*sqrt(Σθ[1,1]), μθ[1] + 3.5*sqrt(Σθ[1,1])]

ρx0 = Uniform(-2, 2) # for initial state x0