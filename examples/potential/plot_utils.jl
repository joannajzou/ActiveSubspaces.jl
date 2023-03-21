using CairoMakie

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
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Spectrum of gradient covariance matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λ_i)",
            yscale=log10)
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