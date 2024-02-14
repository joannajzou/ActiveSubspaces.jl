using ActiveSubspaces
using FastGaussQuadrature
using Distributions
using Test

@testset "computing discrepancies" begin

    # test case
    V(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    ∇xV(x, θ) = - θ[1] * x + θ[2] * x.^3
    ∇θV(x, θ) = [-x^2 / 2, x^4 / 4]

    # densities
    p = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=[3.0,3.0])
    q = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=0.5, θ=[1.0,5.0])

    # integration points
    ξi, wi = gaussquad(100, gausslegendre, limits=[-10, 10])
    GQint = GaussQuadrature(ξi, wi)

    nsamp = 10000
    nuts = NUTS(1e-2)
    ρx0 = Uniform(-2, 2)
    xsamp = rand(p, nsamp, nuts, ρx0)
    MCint = MCSamples(xsamp)


    # KL divergence
    kld = KLDivergence(GQint)

    @test compute_discrepancy(p, p, kld) < eps()
    @test abs.(compute_discrepancy(p, q, kld)) > 0.0

    # Fisher divergence
    fd_q = FisherDivergence(GQint)
    fd_m = FisherDivergence(MCint)

    @test compute_discrepancy(p, p, fd_q) < eps()
    @test compute_discrepancy(p, p, fd_m) < eps()

    @test abs(compute_discrepancy(p, q, fd_q)) > 0.0
    @test abs(compute_discrepancy(p, q, fd_m)) > 0.0
    @test round(compute_discrepancy(p, q, fd_q), digits=1) == round(compute_discrepancy(p, q, fd_m), digits=1)

end