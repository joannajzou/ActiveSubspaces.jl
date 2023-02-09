using ActiveSubspaces
using Test

@testset "ActiveSubspaces.jl" begin
    # Write your tests here.
    @test f(2,1) == 5
    @test f(3,5) == 11
end
