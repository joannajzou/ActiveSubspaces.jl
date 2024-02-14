using ActiveSubspaces
using Distributions
using Test

@testset "Gibbs dist. (univariate)" begin

    # test case
    V(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    ∇xV(x, θ) = - θ[1] * x + θ[2] * x.^3
    ∇θV(x, θ) = [-x^2 / 2, x^4 / 4]
    
    d0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
    d1 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    d2 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=[1,1])
    d0a = Gibbs(d0, β=1.0)  
    d0b = Gibbs(d0, β=1.0, θ=[1,1])

    px0 = Uniform(-2, 2) # prior density for sampling x
    
    
    # test each deployment 
    @test d0a.θ == nothing
    @test d0.V(2.0, [1,1]) == d0b.V(2.0) == 3.0
    @test d1.V(2.0, [1,1]) == d0b.V(2.0) == 3.0
    @test d2.V(2.0) == 3.0

    # check type
    @test typeof(d0a) == Gibbs

    # check supplementary functions
    @test params(d0) == (nothing, nothing)
    @test params(d1) == params(d0a) == (1.0, nothing)
    @test params(d2) == params(d0b) == (1.0, [1,1])
    @test updf(d2, 1) == updf(d2, -1) # test for symmetry
    @test logupdf(d2, 1) == -0.75
    @test gradlogpdf(d2, 2) == -6.0
    @test hasupdf(d2)

    # check sampling 
    @test length(rand(d2, 10000, NUTS(1e-2), px0)) == 10000

    
end


@testset "Mixture Gibbs dist. (univariate)" begin

    # test case
    V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
    ∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
    ∇θV(x, θ) = [x^2 / 2, -x^4 / 4]
    
    d0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
    px0 = Uniform(-2, 2) # prior density for sampling x
    
    centers = [[1,1],[1,2],[2,2]]
    wts = [0.5, 0.3, 0.2]
    dcent = [Gibbs(d0, β=1.0, θ=c) for c in centers]
    mm = MixtureModel(dcent, wts)

    
    # check type
    @test typeof(mm) <: MixtureModel{Union{Multivariate, Univariate}, Continuous, Gibbs}

    # check supplementary functions
    @test components(mm) == dcent
    @test probs(mm) == wts
    @test updf(mm, 1) == updf(mm, -1) # test for symmetry
    @test logupdf(mm, 0) == -1.0
    @test hasupdf(mm)

    # check sampling 
    @test length(rand(mm, 10000, NUTS(1e-2), px0)) == 10000

    
end