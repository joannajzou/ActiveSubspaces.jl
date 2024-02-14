using SafeTestsets

@safetestset "Preprocessing Tests" begin include("data_processing_tests.jl") end
@safetestset "Gibbs Tests" begin include("distributions_tests.jl") end
@safetestset "Discrepancy Tests" begin include("discrepancy_tests.jl") end
@safetestset "QoI Int. Tests" begin include("qoi_integration_tests.jl") end
@safetestset "Subspaces Tests" begin include("subspaces_tests.jl") end