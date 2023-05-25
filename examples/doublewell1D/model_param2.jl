# specify potential function and its gradients
V(x, θ) = (θ[1] / θ[2]^4) * ((x - 1)^2 - θ[2]^2 )^2
∇xV(x, θ) = (4 * θ[1] / θ[2]^4) * ((x - 1)^2 - θ[2]^2) * (x - 1)
∇θV(x, θ) = [ (1 / θ[2]^4) * ((x - 1)^2 - θ[2]^2 )^2,
          - 4 * θ[1] * ( ((x - 1)^2 - θ[2]^2)^2 / θ[2]^5 + ((x - 1)^2 - θ[2]^2) / θ[2]^3 ) ]

# instantiate Gibbs object
πgibbs0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
πgibbs = Gibbs(πgibbs0, β=1.0)

# define sampling density
μθ = [1; 0.5] # 3*ones(2)
Σθ = [0.5 0; 0 0.05] # 0.5*I(2) 
ρθ = MvNormal(μθ, Σθ) # prior on θ 
θrng = [μθ[1] - 3.5*sqrt(Σθ[1,1]), μθ[1] + 3.5*sqrt(Σθ[1,1])]

ρx0 = Uniform(-2, 2) # for initial state x0