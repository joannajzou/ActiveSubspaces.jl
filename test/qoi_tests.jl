using ActiveSubspaces
using Distributions
using LinearAlgebra
using Test

@testset "computing QoIs" begin

    # test case
    V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
    ∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
    ∇θV(x, θ) = [x^2 / 2, -x^4 / 4]

    πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    q = GibbsQoI(h=V, p=πgibbs)
    ρx0 = Uniform(-2, 2)
    θtest = [2.0, 4.0]

    # QoI with quadrature integration
    ξi, wi = gausslegendre(100, -10, 10)
    Qquad = expectation(θtest, q; ξ=ξi, w=wi)
    ∇Qquad = grad_expectation(θtest, q; ξ=ξi, w=wi)

    # QoI with MCMC sampling 
    nsamp = 10000
    nuts = NUTS(1e-2)
    Qmc = expectation(θtest, q; n=nsamp, sampler=nuts, ρ0=ρx0)
    ∇Qmc = grad_expectation(θtest, q; n=nsamp, sampler=nuts, ρ0=ρx0)

    # QoI with importance sampling
    g = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=0.2, θ=[3,3])
    Qis1 = expectation(θtest, q; g=g, n=nsamp, sampler=nuts, ρ0=ρx0)
    ∇Qis1 = grad_expectation(θtest, q; g=g, n=nsamp, sampler=nuts, ρ0=ρx0)

    πu = Uniform(-5,5)
    Qis2 = expectation(θtest, q; g=πu, n=nsamp)
    ∇Qis2 = grad_expectation(θtest, q; g=πu, n=nsamp)

    xsamp = rand(πu, nsamp)
    Qis3 = expectation(θtest, q; g=πu, xsamp=xsamp)
    ∇Qis3 = grad_expectation(θtest, q; g=πu, xsamp=xsamp)

    # test with magnitude of error 
    @test abs((Qquad - Qmc)/Qquad) <= 0.1
    @test abs((Qquad - Qis1)/Qquad) <= 0.1
    @test abs((Qquad - Qis2)/Qquad) <= 0.1
    @test abs((Qquad - Qis3)/Qquad) <= 0.1

    @test norm((∇Qquad - ∇Qmc)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis1)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis2)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis3)./∇Qquad) <= 1.0
    
    
end

 