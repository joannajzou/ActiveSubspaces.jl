using SafeTestsets

@safetestset "Gibbs Tests" begin include("distributions_tests.jl") end
@safetestset "QoI Tests" begin include("qoi_tests.jl") end
@safetestset "Subspaces Tests" begin include("subspaces_tests.jl") end