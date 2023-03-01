using ActiveSubspaces
using Distributions
using Test

@testset "Gibbs dist. (univariate)" begin

    # test case
    V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
    ∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
    ∇θV(x, θ) = [x^2 / 2, -x^4 / 4]
    
    d0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
    d1 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    d2 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=[1,1])
    d0a = Gibbs(d0, β=1.0)  
    d0b = Gibbs(d0, β=1.0, θ=[1,1])

    px0 = Uniform(-2, 2) # prior density for sampling x
    
    
    # test each deployment 
    @test d0a.θ == nothing
    @test d0.V(2.0, [1,1]) == d0b.V(2.0) == -3.0
    @test d1.V(2.0, [1,1]) == d0b.V(2.0) == -3.0
    @test d2.V(2.0) == -3.0

    # check type
    @test typeof(d0a) == Gibbs


    # check supplementary functions
    @test params(d0) == (nothing, nothing)
    @test params(d1) == params(d0a) == (1.0, nothing)
    @test params(d2) == params(d0b) == (1.0, [1,1])
    @test updf(d2, 1) == updf(d2, -1)
    @test logupdf(d2, 1) == -0.75
    @test gradlogpdf(d2, 2) == -6.0

    # check sampling 
    @test length(rand(d2, 10000, NUTS(1e-2), px0)) == 10000

    
end

