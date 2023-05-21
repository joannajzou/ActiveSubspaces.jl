# define QoI as mean energy V
function h(x, θ, integrator::Integrator)
    # compute expected value
    EV = expectation()

∇h(x, θ) = ∇θV(x, θ)


# define QoI
q = GibbsQoI(h=h, p=πgibbs)
