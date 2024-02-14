function import_data(dir::String, IntType::String, nsamp_arr::Vector{Int64}, nrepl::Int64; nsamptot::Int64=nsamp_arr[end])
    Cdict = Dict{Int64, Vector{Matrix{Float64}}}()

    # initialize dict
    for nsamp in nsamp_arr
        Cdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)
    end
        
    for j = 1:nrepl
        Cj = JLD.load("$(dir)/repl$j/DW1D_$(IntType)_cov_nsamp=$nsamptot.jld")["C"]
        
        for nsamp in nsamp_arr
            Cdict[nsamp][j] = Cj[nsamp]
        end

    end

    return Cdict
end


function import_data(dir::String, IntType::String, nsamp_arr::Vector{Int64}, nrepl::Int64, β::Float64; nsamptot::Int64=nsamp_arr[end])
    Cdict = Dict{Int64, Vector{Matrix{Float64}}}()

    for (k,nsamp) in zip(1:length(nsamp_arr), nsamp_arr)
        Cdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)
        for j = 1:nrepl
            Cj = JLD.load("$(dir)/repl$j/DW1D_$(IntType)_nsamp=$nsamptot.jld")["C"][β]
            Ckeys = collect(keys(Cj))
            Cdict[nsamp][j] = Cj[Ckeys[k]]
        end
    end

    return Cdict
end


function compute_val(Cref::Matrix{Float64}, Cdict::Dict, rdim::Int64)
    # extract values
    nsamp_arr = sort(collect(keys(Cdict)))
    
    # compute eigenvalues and eigenvectors
    _, λref, _ = compute_eigenbasis(Cref)
    λdict, _ = compute_eigenbasis(Cdict, nsamp_arr)

    # compute validation metrics
    val = Dict{String, Dict}()
    val["λ1_err"] = Dict{Int64, Vector{Float64}}()
    val["λ2_err"] = Dict{Int64, Vector{Float64}}()
    val["Forstner"] = Dict{Int64, Vector{Float64}}()
    val["WSD"] = Dict{Int64, Vector{Float64}}()

    for nsamp in nsamp_arr
        val["λ1_err"][nsamp] = [(λi[1] - λref[1]) / λref[1] for λi in λdict[nsamp]]
        val["λ2_err"][nsamp] = [(λi[2] - λref[2]) / λref[2] for λi in λdict[nsamp]]
        val["Forstner"][nsamp] = [ForstnerDistance(Cref, Ci) for Ci in Cdict[nsamp]]
        val["WSD"][nsamp] = [WeightedSubspaceDistance(Cref, Ci, rdim) for Ci in Cdict[nsamp]]
    end

    return val

end


function concat_dicts(C_arr::Vector{<:Dict})
    Call = Dict{Int64, Vector{Matrix}}()
    nsamp_arr = collect(keys(C_arr[1]))

    for nsamp in nsamp_arr
        Call[nsamp] = [C[nsamp] for C in C_arr]
    end
    return Call
end
