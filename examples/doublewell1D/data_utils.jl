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

