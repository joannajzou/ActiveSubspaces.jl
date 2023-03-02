using SafeTestsets

@safetestset "Gibbs Distribution Tests" begin include("distributions_tests.jl") end
@safetestset "QoI Tests" begin include("qoi_tests.jl") end

