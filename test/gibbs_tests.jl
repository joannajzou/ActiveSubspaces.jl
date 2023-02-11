@testset "Gibbs dist. (univariate)" begin

    # univariate test functions
    V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
    ∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
    ∇θV(x, θ) = [x^2 / 2, -x^4 / 4]
    dist = Gibbs(1, [1,1], V, ∇xV, ∇θV)

    # check types
    @test typeof(Gibbs(1, [1,1], V, ∇xV, ∇θV)) == Gibbs{Float64}
    @test typeof(Gibbs(1.0, [1,1], V, ∇xV, ∇θV)) == Gibbs{Float64}
    @test typeof(Gibbs(1, [1.0,1.0], V, ∇xV, ∇θV)) == Gibbs{Float64}

    # check supplementary functions
    @test params(dist) == (1.0, [1.0,1.0])
    @test updf(dist, 1) == updf(dist, -1)
    @test logupdf(dist, 1) == -0.75
    @test gradlogpdf(dist, 2) == -6.0

end


@testset "Gibbs dist. (multivariate)" begin

    # multivariate test functions
   

    # check types
    

end

dist = Gibbs{Float64}(1.0, V, ∇xV, ∇θV)