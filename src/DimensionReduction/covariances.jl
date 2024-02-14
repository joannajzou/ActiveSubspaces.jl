
"""
function compute_covmatrix(dθ::Vector{T}) where T <: Vector{<:Real}

Computes covariance matrix from samples of feature vector dθ.

# Arguments
- `dθ :: Vector{Vector{<:Real}}`    : samples of feature vector

# Outputs 
- `C :: Matrix{Real}`            : empirical covariance matrix

""" 
function compute_covmatrix(dθ::Vector{T}) where T <: Vector{<:Real}
    return Matrix(Hermitian(mean(di*di' for di in dθ)))
end



"""
function compute_covmatrix(dθ1::Vector{T}, dθ2::Vector{T}) where T <: Vector{<:Real}

Computes multifidelity covariance matrix from samples of the "high-fidelity" features dθ1 and "low-fidelity" features dθ2.

# Arguments
- `dθ1 :: Vector{Vector{<:Real}}`    : samples of "high fidelity" feature vector
- `dθ2 :: Vector{Vector{<:Real}}`    : samples of "low fidelity" feature vector


# Outputs 
- `C :: Matrix{Real}`            : multifidelity covariance matrix

""" 
function compute_covmatrix(dθh::Vector{T}, dθl1::Vector{T}, dθl2::Vector{T}) where T <: Vector{Float64}
    logCh = log(Matrix(Hermitian(mean([di*di' for di in dθh]))))
    logCl1 = log(Matrix(Hermitian(mean([di*di' for di in dθl1]))))
    logCl2 = log(Matrix(Hermitian(mean([di*di' for di in dθl2]))))

    return exp(logCh + (logCl2 - logCl1))
end


"""
function compute_covmatrix(dθ::Vector{T}, nsamp_arr::Vector{Int64}) where T <: Vector{<:Real}

Computes array of covariance matrices using varying sample sizes of feature vector dθ. All elements of nsamp_arr <= length(dθ).

# Arguments
- `dθ :: Vector{T}`                         : samples of feature vector
- `nsamp_arr :: Vector{Int64}`              : subsets of samples

# Outputs 
- `C_arr :: Dict{Int64, Matrix{Float64}}`   : dictionary with keys being nsamp, values being covariance matrices

""" 
function compute_covmatrix(dθ::Vector{T}, nsamp_arr::Vector{Int64}) where T <: Vector{<:Real}
    C_arr = Dict{Int64, Matrix}()
    
    for nsamp in nsamp_arr
        C_arr[nsamp] = compute_covmatrix(dθ[1:nsamp])
    end

    return C_arr
end

function compute_covmatrix(dθh::Vector{T}, dθl1::Vector{T}, dθl2::Vector{T}, nsamp_arr::Vector{Int64}) where T <: Vector{Float64}
    C_arr = Dict{Int64, Matrix}()

    for nsamp in nsamp_arr
        C_arr[nsamp] = compute_covmatrix(dθh, dθl1, dθl2[1:nsamp])
    end

    return C_arr
end

function compute_covmatrix(dθh::Vector{T}, dθl::Vector{T}, Mh::Vector{Int64}, Ml::Vector{Int64}, budget::Vector{<:Real}) where T <: Vector{Float64}
    C_arr = Dict{Int64, Matrix}()

    for i = 1:length(Mh)
        C_arr[budget[i]] = compute_covmatrix(dθh[1:Mh[i]], dθl[1:Mh[i]], dθl[(Mh[i]+1):(Mh[i]+Ml[i])])
    end

    return C_arr
end



"""
function compute_covmatrix(dθ::Vector{T}, nboot::Int64) where T <: Vector{<:Real}

Computes a bootstrap estimate of the empirical covariance matrix.

# Arguments
- `dθ :: Vector{T}`                         : samples of feature vector
- `nsamp_arr :: Vector{Int64}`              : subsets of samples

# Outputs 
- `C_arr :: Dict{Int64, Matrix{Float64}}`   : dictionary with keys being nsamp, values being covariance matrices

""" 
# TO DO:
# function compute_covmatrix(dθ::Vector{T}, nboot::Int64) where T <: Vector{<:Real}

#     return C_arr
# end

