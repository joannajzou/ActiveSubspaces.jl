using SafeTestsets

@safetestset "Gibbs Tests" begin include("gibbs_tests.jl") end
@safetestset "QoI Tests" begin include("qoi_tests.jl") end

