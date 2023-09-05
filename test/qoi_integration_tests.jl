using ActiveSubspaces
using Distributions
using LinearAlgebra
using Test

@testset "computing QoIs" begin

    # test case
    V(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    ∇xV(x, θ) = - θ[1] * x + θ[2] * x.^3
    ∇θV(x, θ) = [-x^2 / 2, x^4 / 4]

    πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    q = GibbsQoI(h=V, p=πgibbs)
    ρx0 = Uniform(-2, 2)
    θtest = [2.0, 4.0]

    # QoI with quadrature integration
    ξi, wi = gausslegendre(100, -10, 10)
    GQint = GaussQuadrature(ξi, wi)
    Qquad = expectation(θtest, q, GQint)
    ∇Qquad = grad_expectation(θtest, q, GQint) # using ForwardDiff
    ∇Qquad2 = grad_expectation(θtest, q, GQint; gradh=∇θV) # using pre-defined gradient
    @test ∇Qquad ≈ ∇Qquad2

    # QoI with MCMC sampling 
    nsamp = 10000
    nuts = NUTS(1e-2)
    MCint = MCMC(nsamp, nuts, ρx0)
    Qmc = expectation(θtest, q, MCint)
    @time ∇Qmc = grad_expectation(θtest, q, MCint) # using ForwardDiff

    # QoI with importance sampling
    # draw MCMC samples
    g = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=0.2, θ=[3,3])
    ISGint = ISMCMC(g, nsamp, nuts, ρx0)
    Qis1, his1, wis1 = expectation(θtest, q, ISGint)
    @time ∇Qis1, ∇his1, ∇wis1 = grad_expectation(θtest, q, ISGint; gradh=∇θV) 

    # use fixed samples
    xsamp = rand(g, nsamp, nuts, ρx0)
    ISGint2 = ISSamples(g, xsamp)
    Qis2, his2, wis2 = expectation(θtest, q, ISGint2)
    @time ∇Qis2, ∇his2, ∇wis2 = grad_expectation(θtest, q, ISGint2; gradh=∇θV)

    # use uniform biasing disttribution
    πu = Uniform(-5,5)
    ISUint = ISMC(πu, nsamp)
    Qis3, his3, wis3 = expectation(θtest, q, ISUint)
    @time ∇Qis3, ∇his3, ∇wis3 = grad_expectation(θtest, q, ISUint)

    # use fixed samples from biasing distribution
    xsamp = rand(πu, nsamp)
    ISUint2 = ISSamples(πu, xsamp)
    Qis4, his4, wis4 = expectation(θtest, q, ISUint2)
    @time ∇Qis4, ∇his4, ∇wis4 = grad_expectation(θtest, q, ISUint2)

    # use mixture biasing distribution
    


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

 