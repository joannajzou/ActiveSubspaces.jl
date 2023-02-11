@testset "Gibbs dist. (univariate)" begin

    # test case
    V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
    ∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
    ∇θV(x, θ) = [x^2 / 2, -x^4 / 4]
    d0 = Gibbs(V, ∇xV, ∇θV)
    dist = Gibbs(d0, 1.0, [1,1])

    # check types
    @test typeof(d0) == Gibbs{Float64}
    @test typeof(Gibbs(d0, 1.0, [1,1])) == Gibbs{Float64}

    # check definition of potential functions
    @test dist.V(2) == -3.0
    @test dist.V(2) == d0.V(2, [1,1])


    # check supplementary functions
    @test params(d0) = (UndefInitializer(), UndefInitializer())
    @test params(dist) == (1.0, [1.0,1.0])
    @test updf(dist, 1) == updf(dist, -1)
    @test logupdf(dist, 1) == -0.75
    @test gradlogpdf(dist, 2) == -6.0

end


# @testset "Gibbs dist. (multivariate)" begin

#     # multivariate test functions
   

#     # check types
    

# end

V(x, θ)
d0 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV)
d1 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)
d2 = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0, θ=[1,1])

d0b = Gibbs!(d0, β=1.0, θ=[1,1])
d1b = Gibbs!(d1, θ=[1,1])
d0b = Gibbs!(d0, β=1.0)