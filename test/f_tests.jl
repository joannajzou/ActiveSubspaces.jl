@testset "ActiveSubspaces.jl" begin
    # Write your tests here.
    @test f(2,1) == 5
    @test f(3,5) == 11
end

@testset "Derivative Tests" begin
    # Write your tests here.
    @test dfx(2,1) == 2
end
