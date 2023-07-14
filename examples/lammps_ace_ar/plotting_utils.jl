using CairoMakie
using Colors
using StatsBase


custom_theme = Theme(
    font = "/Users/joannazou/Documents/MIT/Presentations/UNCECOMP 2023/InterFont/static/Inter-Regular.ttf",
    fontsize = 20,
    Axis = (
        xgridcolor = :white,
        ygridcolor = :white,
    )
)


function plot_energy(e_pred::Vector{Float64}, e_true)
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10

    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="E ref. [eV/atom]", ylabel="E pred. [eV/atom]")
    scatter!(ax, e_true, e_pred, markersize=7, colormap=(:viridis, 0.5))
    lines!(ax, r0:rs:r1, r0:rs:r1, color=:red)
    return fig
end


function plot_energy(e_pred::Vector{T}, e_true; predlab=1:length(e_pred)) where T <: Vector{<:Real}
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10

    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="E ref. [eV/atom]", ylabel="E pred. [eV/atom]")
   
    for k = 1:length(e_pred)
        scatter!(ax, e_true, e_pred[k], markersize=7, colormap=(:viridis, 0.5), label=predlab[k])
    end
    lines!(ax, r0:rs:r1, r0:rs:r1, color=:red)
    axislegend(ax, position=:lt)
    return fig
end


function plot_error(err_pred::Vector{Float64})
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="error [eV]", ylabel="count")
    hist!(ax, err_pred, bins=30)
    return fig
end


function plot_error(err_pred::Vector{T}; predlab=1:length(err_pred)) where T <: Vector{<:Real}
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="error [eV]", ylabel="count")

    for k = 1:length(err_pred)
        hist!(ax, err_pred[k], colormap=(:viridis, 0.2), bins=30, label=predlab[k])
    end
    axislegend(ax)
    return fig
end


function plot_eigenspectrum(λ::Vector{Float64})
    fig = Figure(resolution=(500,500))
    ax = Axis(fig[1,1],
            title="Spectrum of gradient covariance matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λi)",
            yscale=log10,
            xgridvisible=false,
            ygridvisible=false)
    scatterlines!(ax, 1:length(λ), λ)
    return fig
end


function plot_eigenspectrum(λ::Vector{T}; predlab=1:length(λ)) where T <: Vector{<:Real}
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Spectrum of matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λ_i)",
            yscale=log10)

    for k = 1:length(λ)
        scatterlines!(ax, 1:length(λ[k]), λ[k], label=predlab[k])
    end
    axislegend(ax)
    return fig
end


function plot_cosine_sim(W::Vector{T}) where T <: Matrix{<:Real}
    d = size(W[1], 2)
    cs = [cosine_sim(W[1][:,i], W[2][:,i]) for i = 1:d]
    
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Cosine similarity of eigenvectors",
            xlabel="index i",
            ylabel="cosine sim. of ϕ_i",
            xticks=(1:10:d)
            )
    
    scatterlines!(ax, 1:d, cs)
    return fig
end


function plot_wsd(Cref::Matrix{<:Real}, C::Matrix{<:Real})
    dim = size(Cref, 1)
    wsd_vec = [WeightedSubspaceDistance(Cref, C, r) for r = 1:dim]
    
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Weighted subspace distance",
            xlabel="dimension (r)",
            ylabel="WSD(r)",
            # yscale=log10
            )

    scatterlines!(ax, 1:dim, wsd_vec)
    return fig, wsd_vec
end


function plot_cossim(Cref::Matrix{<:Real}, C::Matrix{<:Real})
    dim = size(Cref, 1)
    _, _, Wref = compute_eigenbasis(Cref)
    _, _, W = compute_eigenbasis(C)
    
    cossim = [Wref[:,i]'*W[:,i] for i = 1:dim]
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Cosine similarity",
            xlabel="eigenvector (ϕi)",
            xticks = 1:dim,
            ylabel="cos. sim.",
            # yscale=log10
            )

    scatterlines!(ax, 1:dim, cossim)
    return fig, cossim
end

function plot_cossim(Cref::Matrix{<:Real}, Ctup::Tuple, labs::Tuple)
    dim = size(Cref, 1)
    _, _, Wref = compute_eigenbasis(Cref)
    
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Cosine similarity",
            xlabel="eigenvector (ϕi)",
            xticks = 1:dim,
            ylabel="cos. sim.",
            # yscale=log10
            )

    for (j,C) in enumerate(Ctup)
        _, _, W = compute_eigenbasis(C)
        cossim = [Wref[:,i]'*W[:,i] for i = 1:dim]
        scatterlines!(ax, 1:dim, cossim, label=labs[j])
    end
    axislegend(ax)
    return fig
end


function plot_mcmc_trace(postsamp::Matrix)

    d = size(postsamp, 1)
    if d % 2 != 0
        d = d + 1
        postsamp = reduce(vcat, [postsamp, zeros(size(postsamp, 2))'])
    end


    fig = Figure(resolution=(1800, 1000))
    ax = Vector{Axis}(undef, d)

    for i = 1:d
        if i < d/2 + 1
            ax[i] = Axis(fig[i,1], ylabel="state $i")
        elseif i == Int(d/2)
            ax[i] = Axis(fig[i,1], xlabel="step (k)", ylabel="state $i")
        elseif i == d 
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="step (k)", ylabel="state $i")
        else
            ax[i] = Axis(fig[i-Int(d/2),2], ylabel=" state $i")
        end
        postsamp_i = postsamp[i,:]
        lines!(ax[i], 1:length(postsamp_i), postsamp_i)
    end
    fig
end

function plot_mcmc_trace(postsamp::Vector{Vector{Float64}})
    postsamp = reduce(hcat, postsamp)
    return plot_mcmc_trace(postsamp)
end

function plot_mcmc_autocorr(postsamp::Matrix; lags=0:1000)
    fig = Figure(resolution=(1000, 750))
    ax = Axis(fig[1,1], xlabel="lag (τ)", ylabel="autocorr.")
    for i = 1:d
        autocorr_i = autocor(postsamp[i,:], lags)
        lines!(ax, 1:length(autocorr_i), autocorr_i, label="state $i")
    end
    axislegend(ax)
    fig
end

function plot_mcmc_autocorr(postsamp::Vector{Vector{Float64}})
    postsamp = reduce(hcat, postsamp)
    return plot_mcmc_autocorr(postsamp)
end


function plot_staggered_hist(vecs::Union{Tuple,Vector}, xoff::Vector, xlab::String, xticklabs::Vector, ylab::String, ttl::String; labs=:none, logscl=false)
    fig = Figure(resolution=(1200,600))

    if logscl == false
        ax = Axis(fig[1, 1], xticks=(xoff, xticklabs), ylabel=ylab, title=ttl)
    else
        # ylab = "Log " * ylab
        ax = Axis(fig[1, 1], xticks=(xoff, xticklabs), ylabel=ylab, title=ttl, yscale=log10)
        vecs = [abs.(v) .+ 1e-10 for v in vecs]
    end

    if labs == :none
        [hist!(ax, v, scale_to=-0.15, color=(:blue, 0.5), offset=xoff[i], bins=100, direction=:x) for (i,v) in enumerate(vecs)]
    else
        [hist!(ax, v, scale_to=-0.15, color=(:blue, 0.5), offset=xoff[i], bins=100, direction=:x, label=labs[i]) for (i,v) in enumerate(vecs)]
        axislegend(ax)
    end

    return fig
end


function plot_IS_diag_2D_AS(C::Matrix, π::Distribution, θsamp::Vector, met::Vector, met_ttl::String)
    # define subspace
    as = compute_as(C, π, 2)

    # convert samples
    ysamp = (as.W1',) .* θsamp
    ymat = reduce(hcat, ysamp)


    fig = Figure(resolution=(550, 500))
    ax = Axis(fig[1,1], xlabel="y1", ylabel="y2", title=met_ttl)
    sc = scatter!(ax, ymat[1,:], ymat[2,:], markersize=7, color=met) # colormap=Reverse(:Blues)) 
    Colorbar(fig[1,1][1,2], sc)

    return fig
end


function plot_IS_diag_2D_AS(C::Matrix, π::Distribution, θsamp::Vector, mettup::Tuple, met_ttl::Tuple)
    # define subspace
    as = compute_as(C, π, 2)

    # convert samples
    ysamp = (as.W1',) .* θsamp
    ymat = reduce(hcat, ysamp)

    # metric
    nmet = length(mettup)
    cmin = minimum([minimum(wi) for wi in mettup])       # minimum value for colorbar
    cmax = maximum([maximum(wi) for wi in mettup])       # maximum value for colorbar

    # plot diagnostic value vs. parameter value
    fig = Figure(resolution=(550*nmet, 500))
    ax = Vector{Axis}(undef, nmet)
    for i = 1:nmet
        ax[i] = Axis(fig[1,i], xlabel="y1", ylabel="y2", title=met_ttl[i])
        sc = scatter!(ax[i], ymat[1,:], ymat[2,:], markersize=7, color=mettup[i], colormap=Reverse(:Blues), colorrange=(cmin,cmax)) 
        if i == 2; Colorbar(fig[1,i][1,2], sc); end
    end

    return fig
end