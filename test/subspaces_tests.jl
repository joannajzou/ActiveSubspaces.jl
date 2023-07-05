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
    W1a, W2a = find_subspaces(λa, Wa, tol)
    r = size(W1a, 2)
    W1b, W2b = find_subspaces(λa, Wa, r)

    @test W1a == W1b
    @test W2a == W2b

    as1 = compute_as(f, ρθ, tol)
    as2 = compute_as(f, ρθ, r)
    as3 = compute_as(Cref, ρθ, r)

    @test as1.W1 == W1a
    @test as2.W1 == W1a
    @test as3.W1 == W1a

    @test length(as1.W1 * as1.π_y.μ) == d
    @test length(as1.W2 * as1.π_z.μ) == d

    # check sampling
    ysamp, θsamp = sample_as(5, as1)

    @test length(ysamp[1]) == r
    @test length(θsamp[1]) == d

end

 