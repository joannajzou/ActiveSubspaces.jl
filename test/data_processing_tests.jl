using ActiveSubspaces
using Distributions
using Test

@testset "MVN standardization tests" begin

    # test case: double well potential --------------------------------------------------
    
    # define sampling density for θ
    d = 2
    μθ = 4*ones(d) 
    Σθ = 0.2*I(d) 
    ρθ = MvNormal(μθ, Σθ)               # prior on θ 
    ρz = MvNormal(zeros(d), I(d))       # standard normal prior


    # functions wrt original variables θ
    Vθ(x, θ) = - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    ∇xVθ(x, θ) = - θ[1] * x + θ[2] * x.^3
    ∇θVθ(x, θ) = [-x^2 / 2, x^4 / 4]

    # functions wrt normalized variables z
    function Vn(x, z, ρ::MvNormal)
        θ = std_to_mvn(z, ρ)
        return - (θ[1] * x.^2) / 2 + (θ[2] * x.^4) / 4 + 1
    end
    
    function ∇xVn(x, z, ρ::MvNormal)
        θ = std_to_mvn(z, ρ)
        return - θ[1] * x + θ[2] * x.^3
    end
    
    function ∇θVn(x, z, ρ::MvNormal)
        return [- x^2 / 2, x^4 / 4]
    end
    
    
    # instantiate Gibbs objects
    πgθ = Gibbs(V=Vθ, ∇xV=∇xVθ, ∇θV=∇θVθ, β=1.0)

    Vz(x, z) = Vn(x, z, ρθ)
    ∇xVz(x, z) = ∇xVn(x, z, ρθ)
    ∇zVz(x, z) = ∇θVn(x, z, ρθ)
    πgz = Gibbs(V=Vz, ∇xV=∇xVz, ∇θV=∇zVz, β=1.0)


    # instantiate QoI
    qθ = GibbsQoI(h=Vθ, p=πgθ)
    qz = GibbsQoI(h=Vz, p=πgz)

    # define integration mode
    ξx, wx = gausslegendre(100, -10, 10)
    GQint = GaussQuadrature(ξx, wx)



    # begin tests -----------------------------------------------------------------------

    # test sample conversion
    zsamp = [rand(ρz) for i = 1:10]   # sample standard normal
    @test isapprox(zsamp, mvn_to_std.(std_to_mvn.(zsamp, (ρθ,)), (ρθ,)), rtol=1e-8)

    θsamp = [rand(ρθ) for i = 1:10]   # sample original density
    @test isapprox(θsamp, std_to_mvn.(mvn_to_std.(θsamp, (ρθ,)), (ρθ,)), rtol=1e-8)
    ztransf = mvn_to_std.(θsamp, (ρθ,))

    # test QoI calculation
    Qθ = [expectation(θ, qθ, GQint) for θ in θsamp]
    Qz = [expectation(z, qz, GQint) for z in ztransf]
    @test isapprox(Qθ, Qz)

    # test grad QoI calculation
    ∇Qθ = [grad_expectation(θ, qθ, GQint; gradh=∇θVθ) for θ in θsamp]
    ∇Qz = [grad_expectation(z, qz, GQint; gradh=∇zVz) for z in ztransf]
    @test isapprox(∇Qθ, ∇Qz)


end


# @testset "Uniform standardization tests" begin
#     # TO DO 
# end