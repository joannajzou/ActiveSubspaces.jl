using ActiveSubspaces
using Distributions
using LinearAlgebra
using Test

@testset "computing subspaces" begin

    # test case: dummy samples and density
    d = 10 # dimension
    nsamp = 100
    f = [rand(d) for i = 1:nsamp]
    ρθ = MvNormal(ones(d), I(d))
    
    # check covariance matrix calculation
    Cref = compute_covmatrix(f)
    Ca, λa, Wa = compute_eigenbasis(f)
    Cb, λb, Wb = compute_eigenbasis(Cref)

    @test Ca == Cref
    @test Ca == Cb
    @test λa == λb
    @test Wa == Wb

    # check subspace calculation
    tol = 1e-4
    W1a, W2a, π_y, π_z = find_subspaces(Wa, ρθ, tol, λa)
    r = size(W1a, 2)
    W1b, W2b, π_y, π_z = find_subspaces(Wa, ρθ, r)

    @test W1a == W1b
    @test W2a == W2b

    as1 = compute_as(f, ρθ, tol)
    as2 = compute_as(f, ρθ, r)

    @test as1.W1 == W1a
    @test as2.W1 == W1a

    @test length(as1.W1 * as1.π_y.μ) == d
    @test length(as1.W2 * as1.π_z.μ) == d

    # check sampling
    ysamp, θsamp = sample_as(5, as1)

    @test length(ysamp[1]) == r
    @test length(θsamp[1]) == d

end

 