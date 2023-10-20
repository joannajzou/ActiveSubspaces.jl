## 01 - script for specifying the potential energy model and QoI

# potential energy function --------------------------------------------------------

# asymmetric double well potential with additional cubic term - 2D version
# ref: https://link.springer.com/article/10.1007/s10910-022-01328-9
V(x, θ) = 9*x^2 - exp(θ[1])*x^3 + exp(θ[2])*x^4
∇xV(x, θ) = 18*x - 3*exp(θ[1])*x^2 + 4*exp(θ[2])*x^4
∇θV(x, θ) = [-exp(θ[1]) * x^3, exp(θ[2]) * x^4]


# instantiate Gibbs object --------------------------------------------------------
πgibbs0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
πgibbs1 = Gibbs(πgibbs0, β=1.0)
πgibbs2 = Gibbs(πgibbs0, β=0.5)


# parameter sampling density ------------------------------------------------------
d = 2
μθ = [log(18), log(9)] # [log(16), log(24), log(9)]
Σθ = 1e-3 * I(d) # Diagonal([2e-3, 5e-3]) 
ρθ = MvNormal(μθ, Σθ)

# 2D quadrature points over θ-space
θbd = Vector{Vector{Float64}}(undef, d)
ξθ = Vector{Vector{Float64}}(undef, d)          # 2D quad points over θ
wθ = Vector{Vector{Float64}}(undef, d)
for i = 1:d
    θbd[i] = [ρθ.μ[i] - 3.5*sqrt(ρθ.Σ[i,i]), ρθ.μ[i] + 3.5*sqrt(ρθ.Σ[i,i])]
    ξθ[i], wθ[i] = gausslegendre(100, θbd[i][1], θbd[i][2]) # 2D quad points over θ
end


# quantity of interest (QoI) -----------------------------------------------------

# QoI: mean energy
h(x, θ) = V(x, θ)
∇h(x, θ) = ∇θV(x, θ)
q = GibbsQoI(h=V, ∇h=∇h, p=πgibbs1)

# QoI: 2nd moment of energy
h(x, θ) = V(x, θ)^2
∇h(x, θ) = 2*V(x, θ)*∇θV(x, θ)
q2 = GibbsQoI(h=h, ∇h=∇h, p=πgibbs1)

# define approximate gradient of the QoI (for adaptive IS)
E_qoi = GibbsQoI(h=q2.p.∇θV, p=q2.p)
E_∇θV(γ) = expectation(γ, E_qoi, GQint)
hgrad(x, γ) = q2.∇h(x, γ) -  q2.p.β * q2.h(x, γ) * (∇θV(x, γ) - E_∇θV(γ))
qgrad = GibbsQoI(h=hgrad, p=q.p)


# initial state density -----------------------------------------------------------
ρx0 = Uniform(0, 1.5)                           # sampling density for initial state x0
ll = -2; ul = 3                                 # lower and upper limits
xplot = Vector(LinRange(ll, ul, 1000))          # domain for plotting



