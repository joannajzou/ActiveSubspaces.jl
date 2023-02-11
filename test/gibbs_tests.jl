@testset "Gibbs dist. (univariate)" begin

    # test case
    V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
    ∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
    ∇θV(x, θ) = [x^2 / 2, -x^4 / 4]
    
    d0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
    d1 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
    d2 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=[1,1])
    d0m = Gibbs!(d0, β=1.0)  
    d0m2 = Gibbs!(d0, β=1.0, θ=[1,1])
    d1m = Gibbs!(d1, θ=[1,1])
    
    
    # test each deployment 
    @test d0m.θ == nothing
    @test d0.V(2.0, [1,1]) == d0m2.V(2.0) == -3.0
    @test d1.V(2.0, [1,1]) == d1m.V(2.0) == -3.0
    @test d2.V(2.0) == -3.0

    # check type
    @test typeof(d1m) == Gibbs


    # check supplementary functions
    @test params(d0) == (nothing, nothing)
    @test params(d1) == params(d0m) == (1.0, nothing)
    @test params(d2) == params(d1m) == (1.0, [1,1])
    @test updf(d2, 1) == updf(d2, -1)
    @test logupdf(d2, 1) == -0.75
    @test gradlogpdf(d2, 2) == -6.0

end


# @testset "Gibbs dist. (multivariate)" begin

    

# end
