# define QoI as mean energy V
h(x, θ) = V(x, θ)
∇h(x, θ) = ∇θV(x, θ)


# define QoI
q = GibbsQoI(h=h, p=πgibbs1)
