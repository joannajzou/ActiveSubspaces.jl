# define QoI as mean energy V
function h(x, θ, integrator::Integrator)
    V(x, θ)
∇h(x, θ) = ∇θV(x, θ)


# define QoI
q = GibbsQoI(h=h, p=πgibbs)
