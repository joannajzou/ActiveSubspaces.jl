using ActiveSubspaces
using Distributions
using LinearAlgebra
using FastGaussQuadrature
using Test

@testset "computing QoIs" begin

    # 1D test case
    V(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    ∇xV(x, θ) = - θ[1] * x + θ[2] * x.^3
    ∇θV(x, θ) = [-x^2 / 2, x^4 / 4]

    πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    q0 = GibbsQoI(h=V, p=πgibbs)
    q = GibbsQoI(h=V, p=πgibbs, ∇h=∇θV)
    ρx0 = Uniform(-2, 2)
    θtest = [2.0, 4.0]

    # QoI with quadrature integration
    ξi, wi = gaussquad(100, gaussjacobi, 0.5, 0.5, limits=[-10, 10])
    ξi, wi = gaussquad(100, gausslegendre, limits=[-10, 10])

    GQint = GaussQuadrature(ξi, wi)
    Qquad = expectation(θtest, q, GQint)
    ∇Qquad = grad_expectation(θtest, q0, GQint) # using ForwardDiff
    ∇Qquad2 = grad_expectation(θtest, q, GQint) # using pre-defined gradient
    @test ∇Qquad ≈ ∇Qquad2

    # QoI with MCMC sampling 
    nsamp = 10000
    nuts = NUTS(1e-2)
    MCint = MCMC(nsamp, nuts, ρx0)
    Qmc = expectation(θtest, q, MCint)
    @time ∇Qmc = grad_expectation(θtest, q, MCint)

    # QoI with importance sampling
    # draw MCMC samples
    g = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=0.2, θ=[3,3])
    ISint = ISMCMC(g, nsamp, nuts, ρx0)
    Qis1, his1, wis1 = expectation(θtest, q, ISint)
    @time ∇Qis1, ∇his1, ∇wis1 = grad_expectation(θtest, q, ISGint) 

    # use fixed samples
    xsamp = rand(g, nsamp, nuts, ρx0)
    ISint2 = ISSamples(g, xsamp)
    Qis2, his2, wis2 = expectation(θtest, q, ISint2)
    @time ∇Qis2, ∇his2, ∇wis2 = grad_expectation(θtest, q, ISint2)

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
    @test abs((Qquad - Qis4)/Qquad) <= 0.1

    @test norm((∇Qquad - ∇Qmc)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis1)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis2)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis3)./∇Qquad) <= 1.0
    @test norm((∇Qquad - ∇Qis4)./∇Qquad) <= 1.0

    
end

@testset "computing QoI with 2D quadrature" begin

    V(x, θ) = 1/6 * (4*(1-x[1]^2-x[2]^2)^2
            + 2*(x[1]^2-2)^2
            + ((x[1]+x[2])^2-1)^2
            + ((x[1]-x[2])^2-1)^2)
        
    ∇xV(x, θ) = [4/3 * x[1] *(4*x[1]^2 + 5*x[2]^2 - 5),
                4/3 * x[2] *(5*x[1]^2 + 3*x[2]^2 - 3)]

    ∇θV(x, θ) = 0 

    πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    q = GibbsQoI(h=V, p=πgibbs, ∇h=∇θV)
    ρx0 = MvNormal(zeros(2), I(2))
    θtest = [2.0, 4.0]

    ξi, wi = gaussquad_2D(25, gausslegendre, limits=[-10, 10])
    GQint = GaussQuadrature(ξi, wi)
    Qquad = expectation(θtest, q, GQint)
    ∇Qquad = grad_expectation(θtest, q, GQint)

    πg = Gibbs(πgibbs, θ=θtest)
    Z = normconst(πg, GQint)

    
    # check this section
    nsamp = 10000
    nuts = NUTS()
    g = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=[3,3])
    MCint = MCMC(nsamp, nuts, ρx0)
    Qmc = expectation(θtest, q, MCint)
    @time ∇Qmc = grad_expectation(θtest, q, MCint)

end 


 