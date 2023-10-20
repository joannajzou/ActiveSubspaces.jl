## compute gradient of variance QoI

# using QuadIntegrator or MCIntegrator
function compute_gradvar(θsamp::Vector{Vector{Float64}},
                        q::GibbsQoI,
                        integrator::Integrator)

    q2 = GibbsQoI(h=(x, γ) -> q.h(x, γ)^2,
                  ∇h=(x, γ) -> 2*q.h(x, γ)*q.∇h(x, γ),
                  p = q.p)

    EX = map(θ -> expectation(θ, q, integrator), θsamp)
    ∇EX = map(θ -> grad_expectation(θ, q, integrator), θsamp)
    ∇EX2 = map(θ -> grad_expectation(θ, q2, integrator), θsamp)
    
    #Var[X] = E[X^2] - E[X]^2
    return ∇EX2 - 2*EX.*∇EX
end

# using QuadIntegrator or MCIntegrator
function compute_gradvar(θsamp::Vector{Vector{Float64}},
    q::GibbsQoI,
    integrator::MCMC)

    q2 = GibbsQoI(h=(x, γ) -> q.h(x, γ)^2,
    ∇h=(x, γ) -> 2*q.h(x, γ)*q.∇h(x, γ),
    p = q.p)

    # use same set of samples for all expectations, for efficiency
    xsamp = map(θi -> rand(Gibbs(q.p, θ=θi), integrator.n, integrator.sampler, integrator.ρ0), θsamp)
    MCints = map(xi -> MCSamples(xi), xsamp)
    EX = [expectation(θ, q, int) for (θ, int) in zip(θsamp, MCints)]
    ∇EX = [grad_expectation(θ, q, int) for (θ, int) in zip(θsamp, MCints)]
    ∇EX2 = [grad_expectation(θ, q2, int) for (θ, int) in zip(θsamp, MCints)]

    #Var[X] = E[X^2] - E[X]^2
    return ∇EX2 - 2*EX.*∇EX
end


# using ISIntegrator
function compute_gradvar(θsamp::Vector{Vector{Float64}},
                        q::GibbsQoI,
                        integrator::ISIntegrator)

    q2 = GibbsQoI(h=(x, γ) -> q.h(x, γ)^2,
                  ∇h=(x, γ) -> 2*q.h(x, γ)*q.∇h(x, γ),
                  p = q.p)

    out = map(θ -> expectation(θ, q, integrator), θsamp)
    gradout = map(θ -> grad_expectation(θ, q, integrator), θsamp)
    gradout2 = map(θ -> grad_expectation(θ, q2, integrator), θsamp)

    EX = [tup[1] for tup in out]
    ∇EX = [tup[1] for tup in gradout]
    ∇EX2 = [tup[1] for tup in gradout2]
    his = [tup[2] for tup in gradout2]
    wis = [tup[3] for tup in gradout2]

    metrics = compute_metrics(θsamp, his, wis)

    #Var[X] = E[X^2] - E[X]^2
    return ∇EX2 - 2*EX.*∇EX, metrics
end