# using ActiveSubspaces
# using Distributions
# using LinearAlgebra
# using JLD
using Optim

# # select model
# modnum = 4
# include("model_param$(modnum).jl")

# # select qoi
# include("qoi_meanenergy.jl")

# # load utils
# include("mc_utils.jl")
# include("plotting_utils.jl")



# # initialization ####################################################################

# ## set algorithm parameters
# N0 = 5000
# Nk = 5000
# Nmax = 100000

# ## randomly sample initial distribution parameters
# nλ = 20
# λsamp = [rand(ρθ) for i = 1:nλ]   
# # JLD.save("data$modnum/mixture_params.jld", "λsamp", λsamp)

# # or load from file 
# λsamp = JLD.load("data$modnum/mixture_params.jld")["λsamp"]

# λmat = reduce(hcat, λsamp)


# # tempering parameters


# # define target expectations 
# θsamp = λsamp
# qois = [assign_param(q, θ) for θ in θsamp]

# # initialize vectors
# niter = 3
# λset = Vector{Vector}(undef, niter)
# αset = Vector{Vector{Float64}}(undef, niter)
# xset = []

# ## compute initial biasing dist
# λset[1] = λsamp 
# αset[1] = 1/nλ * ones(nλ)
# αk = αset[1]
# h0 = MixtureModel(λset[1], πgibbs2, αset[1])
# g0 = h0

# # draw samples
# xsamp = rand(h0, N0, nuts, ρx0)
# ISint = ISSamples(h0, xsamp)
# append!(xset, xsamp)

# Neval = N0


# # start adaptation ####################################################################

# for k = 2:niter
#     println("------- iteration $k -------")
#     # update parameters
#     λk = update_dist_params(λset[k-1], πgibbs2, qois, ISint)
#     αk = update_mixture_wts(αset[k-1], λset[k-1], πgibbs2, qois, ISint)
#     # compute new biasing dist.
#     gk = MixtureModel(λk, πgibbs2, αk)
#     # draw samples
#     xsamp = rand(gk, Nk, nuts, ρx0)
#     append!(xset, xsamp)
#     # update biasing distribution
#     Neval += Nk
#     hk = MixtureModel((h0, gk), ((Neval - Nk)/Neval, Nk/Neval))
#     # check stopping criterion
#     # println(meet_stop_crit())

#     # save values
#     λset[k] = λk
#     # αset[k] = αk
#     h0 = hk
# end


# # plot centers ###############################################################################################
# θ1_rng = log.(Vector(LinRange(exp.(θbd[1][1]), exp.(θbd[1][2]), 31)))
# θ2_rng = log.(Vector(LinRange(exp.(θbd[2][1]), exp.(θbd[2][2]), 31)))
# # θ1_rng = Vector(LinRange(θbd[1][1], θbd[1][2], 31))
# # θ2_rng = Vector(LinRange(θbd[2][1], θbd[2][2], 31))
# m = length(θ1_rng)
# P_plot = [pdf(ρθ, [θi, θj]) for θi in θ1_rng, θj in θ2_rng]
# Q_surf = [expectation([θi, θj], q, GQint) for θi in θ1_rng, θj in θ2_rng]
# Q_plot = Q_surf[end:-1:1, :] # yflip
# # Qlog_plot =  log.(Q_plot .- minimum(Q_plot) .+ 1)
# # plot 
# fa = plot_parameter_space(θ1_rng, θ2_rng, 
#                     Q_plot, P_plot,
#                     reduce(hcat, λsamp),
#                     reduce(hcat, λset[end])
# )


# functions ################################################################################################

# function update_dist_params(λsamp::Vector, πg::Gibbs, qois::Vector{GibbsQoI}, integrator::ISSamples)

#     xsamp = integrator.xsamp
#     N = length(xsamp)
#     J = length(λsamp)

#     # define functions
#     # Zg(λ) = 1/N * sum(g.(xsamp, (λ,)) ./ h.(xsamp))
#     Zg_quad(λ) = normconst(Gibbs(πg, θ=λ), GQint)
#     g(x, λ) = updf(Gibbs(πg, θ=λ), x)

#     # initilaize
#     λopt = Vector{Vector{Float64}}(undef, J)

#     for j = 1:J
#         println("adapt component $j")
#         qj = qois[j]
#         ϕj(xsamp, λ) = abs.(qj.h.(xsamp)) .* log.(g.(xsamp, (λ,)) ./ Zg_quad(λ))
#         # wt(x) = updf(qj.p, x) / updf(integrator.g, x)
#         logwt(x) = logupdf(qj.p, x) - logupdf(integrator.g, x)
        
#         # estimators
#         M = maximum(logwt.(xsamp))
#         # ent(λ) = - sum( ϕj(xsamp, λ) .* wt.(xsamp) ) / sum( wt.(xsamp) )
#         function entlog(λ)
#             E = - sum( ϕj(xsamp, λ) .* exp.(logwt.(xsamp) .- M) ) / sum( exp.(logwt.(xsamp) .- M) )
#             if length(E) > 1
#                 return sum(E) # sum of entropic vector
#             else
#                 return E
#             end
#         end
#             # gradentlog!(G, λ) = gradent!(G, λ, ϕj, πg, ISint, logwt)

#         # res = optimize(entlog, gradentlog!, λsamp[j], GradientDescent(), Optim.Options(show_trace=true, iterations=3)) # ; autodiff = :forward)
#         @time res = optimize(entlog, λsamp[j], NelderMead(),
#                  Optim.Options(f_tol=1e-3, g_tol=1e-4, f_calls_limit=100)) # ; autodiff = :forward)

#         λopt[j] = Optim.minimizer(res)
#     end

#     return λopt

# end


function update_dist_params(λsamp::Vector,
                        πg::Gibbs,
                        qois::Vector{GibbsQoI},
                        integrator::ISSamples)

    xsamp = integrator.xsamp
    N = length(xsamp)
    J = length(λsamp)

    # define functions
    # Zg(λ) = 1/N * sum(g.(xsamp, (λ,)) ./ h.(xsamp)) # Monte Carlo estimate
    Zg(λ) = normconst(Gibbs(πg, θ=λ), GQint)
    g(x, λ) = updf(Gibbs(πg, θ=λ), x)

    # initilaize
    λopt = Vector{Vector{Float64}}(undef, J)

    for j = 1:J
        println("adapt component $j")
        qj = qois[j]
        
        function ϕj(xsamp, λ)
            hsamp = qj.h.(xsamp)
            hsamp = [abs.(h) for h in hsamp]
            return hsamp .* log.(g.(xsamp, (λ,)) ./ Zg(λ))
        end

        logwt(xv) = logupdf.((qj.p,), xv) - logpdf(integrator.g, xv, GQint)
        
        # estimators
        M = maximum(logwt(xsamp))
        function entlog(λ)
            E = - sum( ϕj(xsamp, λ) .* exp.(logwt(xsamp) .- M) ) / sum( exp.(logwt(xsamp) .- M) )
            if length(E) > 1
                return sum(E) # sum of entropic vector
            else
                return E
            end
        end

        @time res = optimize(entlog, λsamp[j], NelderMead(),
                 Optim.Options(show_trace=true, f_tol=1e-4, g_tol=1e-6, f_calls_limit=100)) # ; autodiff = :forward)

        λopt[j] = Optim.minimizer(res)
    end

    return λopt

end





# function update_mixture_wts(αsamp::Vector, λsamp::Vector, πg::Gibbs, qois::Vector{GibbsQoI}, integrator::ISSamples)
#     xsamp = integrator.xsamp
#     N = length(xsamp)
#     J = length(λsamp)

#     ϕj(x,j) = qois[j].h(x)
#     πfj(j) = qois[j].p

#     comps = [Gibbs(πg, θ=λ) for λ = λsamp]
#     πgα(α) = MixtureModel(comps, α)

#     logwt(x,α,j) = 2*logupdf(πfj(j), x) - logupdf(πgα(α), x) - logupdf(integrator.g, x)
    
#     function varj(α,j)
#         α = normalize(α)
#         M = maximum(logwt.(xsamp, (α,), (j,)))
#         return sum( ϕj.(xsamp,(j,)).^2 .* exp.(logwt.(xsamp, (α,), (j,)) .- M) ) / sum( exp.(logwt.(xsamp, (α,), (j,)) .- M) )
#     end

#     varsum(α) = sum([varj(α, j) for j = 1:J])

#     @time res = optimize(varsum, αsamp, NelderMead(),
#                  Optim.Options(show_trace=true, f_tol=1e-2, g_tol=1e-2, f_calls_limit=50))

#     return normalize(Optim.minimizer(res))
# end


function update_mixture_wts(αsamp::Vector,
                        λsamp::Vector,
                        πg::Gibbs,
                        qois::Vector{GibbsQoI},
                        integrator::ISSamples)
    xsamp = integrator.xsamp
    N = length(xsamp)
    J = length(λsamp)

    ϕj(x,j) = qois[j].h(x)
    πfj(j) = qois[j].p
    # πgj(j) = Gibbs(πg, θ=λsamp[j])
    # numj(xv,j) = ϕj.(xv,(j,)) .* updf.((πfj(j),), xv) # .- βsamp[j] * updf.((πgj(j),), xv)
    function numj2(xv,j)
        numj(x) = ϕj(x,j) * updf(πfj(j), x)
        return [numj(x)'*numj(x) for x in xv]
    end

    comps = [Gibbs(πg, θ=λ) for λ in λsamp]
    πgα(α) = MixtureModel(comps, α)
    denomj(xv, α) = pdf(πgα(α), xv, GQint) .* pdf(integrator.g, xv, GQint)
    
    function varj(α,j)
        α = normalize(α)
        return sum( numj2(xsamp,j) ./ denomj(xsamp,α) )
    end

    varsum(α) = sum([varj(α, j) for j = 1:J])
 
    @time res = optimize(varsum, αsamp, NelderMead(),
                 Optim.Options(show_trace=true, f_tol=1e-3, g_tol=1e-2, f_calls_limit=1000))

    return normalize(Optim.minimizer(res))
end


function normalize(α::Vector)
    α = abs.(α)
    return α ./ sum(α)
end

function gradent!(G, λ::Vector, ϕ::Function, πg::Distribution, integrator::ISSamples, logwt::Function)
    xsamp = integrator.xsamp
    
    E_qoi = GibbsQoI(h=πg.∇θV, p=πg)
    E_∇θV, _, _ = expectation(λ, E_qoi, integrator)
    
    gradϕ(xsamp, γ) = - ϕ(xsamp, γ) * πg.β .* [(πg.∇θV(x, γ) .- E_∇θV) for x in xsamp]
    M = maximum(logwt.(xsamp))
    gradentlog(γ) = sum( gradϕ(xsamp, γ) .* exp.(logwt.(xsamp) .- M) ) / sum( exp.(logwt.(xsamp) .- M) )
    
    gradent_eval = gradentlog(λ)
    G[1] = gradent_eval[1]
    G[2] = gradent_eval[2]
end


function integrand(x::Real, θ::Vector, V::Function, π0::Gibbs, ξx::Vector, wx::Vector; β=1.0)
    πg = Gibbs(π0, β=β, θ=θ)
    return abs(V(x, θ)) * updf(πg, x) ./ normconst(πg, GQint)
end

function integrand(x::Vector{<:Real}, θ::Vector, V::Function, π0::Gibbs, ξx::Vector, wx::Vector; β=1.0)
    πg = Gibbs(π0, β=β, θ=θ)
    return abs.(V.(x, (θ,))) .* updf.((πg,), x) ./ normconst(πg, GQint)
end



#### KEEP?
# colscheme = [RGB((i-1)*0.7/niter, 0.3 + (i-1)*0.7/niter, 0.6 + (i-1)*0.2/niter) for i = 1:niter] # color scheme for model 1
# figs = Vector{Figure}(undef, nλ)
# for j = 1:nλ
#     with_theme(custom_theme) do
#         figs[j] = Figure(resolution = (600, 600))
#         ax = Axis(figs[j][1, 1],  xlabel="x", ylabel="ϕ(x) π(x)")
#         # target: integrand
#         lines!(ax, xplot, integrand(xplot, λsamp[j], V, πgibbs0, GQint), linewidth=2, color=:black, label="target")
#         # original distribution
#         g0 = Gibbs(πgibbs0, β=0.5, θ=λsamp[j])
#         lines!(ax, xplot, updf.((g0,), xplot) ./ normconst(g0, GQint), color=:red, label="original")
#         # adapted distribution
#         for k = 2:(niter-1)
#             gk = Gibbs(πgibbs0, β=0.5, θ=λset[k][j])
#             lines!(ax, xplot, updf.((gk,), xplot) ./ normconst(gk, GQint), color=colscheme[k])
#         end
#         gk = Gibbs(πgibbs0, β=0.5, θ=λset[end][j])
#         lines!(ax, xplot, updf.((gk,), xplot) ./ normconst(gk, GQint), color=colscheme[end], label="adapted")

#         axislegend(ax)
#     end
# end




# function expectation_is_stable(xsamp::Vector, ϕ::Function, f::Distribution, g::Distribution)
#     logwt(x) = if hasupdf(f) & hasupdf(g)
#         logupdf(f, x) - logupdf(g, x)
#     elseif hasupdf(f) & !hasupdf(g)
#         logupdf(f, x) - logpdf(g, x)
#     elseif !hasupdf(f) & hasupdf(g)
#         logpdf(f, x) - logupdf(g, x)
#     else
#         logpdf(f, x) - logpdf(g, x)
#     end

#     M = maximum(logwt.(xsamp))
#     return sum( ϕ(xsamp) .* exp.(logwt.(xsamp) .- M) ) / sum( exp.(logwt.(xsamp) .- M) ), ϕ(xsamp), exp.(logwt.(xsamp))
# end




function adapt_mixture_biasing_dist_with_wts(λsamp::Vector,
                    q::GibbsQoI,
                    πg::Gibbs;
                    N0=5000,
                    Nk=N0,
                    niter=10)

    nλ = length(λsamp)

    # define target expectations 
    θsamp = λsamp
    qois = [assign_param(q, θ) for θ in θsamp]
    
    # initialize vectors
    λset = Vector{Vector}(undef, niter)
    αset = Vector{Vector{Float64}}(undef, niter)
    xset = []
    
    ## compute initial biasing dist
    λset[1] = λsamp 
    αset[1] = 1/nλ * ones(nλ)
    αk = αset[1]
    h0 = MixtureModel(λset[1], πg, αset[1])
    
    # draw samples
    xsamp = rand(h0, N0, nuts, ρx0)
    ISint = ISSamples(h0, xsamp)
    append!(xset, xsamp)
    
    Neval = N0
    
    # start adaptation 
    for k = 2:niter
        println("------- iteration $k -------")
        # update parameters
        println("updating params")
        λk = update_dist_params(λset[k-1], πg, qois, ISint)
        # println("updating mixture weights")
        αk = update_mixture_wts(αset[k-1], λset[k-1], πg, qois, ISint)
        # compute new biasing dist.
        gk = MixtureModel(λk, πg, αk)
        # draw samples
        xsamp = rand(gk, Nk, nuts, ρx0)
        append!(xset, xsamp)
        # update biasing distribution
        Neval += Nk
        hk = MixtureModel((h0, gk), ((Neval - Nk)/Neval, Nk/Neval))
        # check stopping criterion
        # println(meet_stop_crit())
    
        # save values
        λset[k] = λk
        αset[k] = αk
        h0 = hk
    end

    return λset, αset
end



function adapt_mixture_biasing_dist(λsamp::Vector,
            q::GibbsQoI,
            πg::Gibbs;
            N0=5000,
            Nk=N0,
            niter=10)

    nλ = length(λsamp)

    # define target expectations 
    θsamp = λsamp
    qois = [assign_param(q, θ) for θ in θsamp]

    # initialize vectors
    λset = Vector{Vector}(undef, niter)
    αset = Vector{Vector{Float64}}(undef, niter)
    xset = []

    ## compute initial biasing dist
    λset[1] = λsamp 
    αset[1] = 1/nλ * ones(nλ)
    αk = αset[1]
    h0 = MixtureModel(λset[1], πg, αset[1])

    # draw samples
    xsamp = rand(h0, N0, nuts, ρx0)
    ISint = ISSamples(h0, xsamp)
    append!(xset, xsamp)

    Neval = N0

    # start adaptation 
    for k = 2:niter
        println("------- iteration $k -------")
        # update parameters
        println("updating params")
        λk = update_dist_params(λset[k-1], πg, qois, ISint)
        # println("updating mixture weights")
        # αk = update_mixture_wts(αset[k-1], λset[k-1], πg, qois, ISint)
        # compute new biasing dist.
        gk = MixtureModel(λk, πg, αk)
        # draw samples
        xsamp = rand(gk, Nk, nuts, ρx0)
        append!(xset, xsamp)
        # update biasing distribution
        Neval += Nk
        hk = MixtureModel((h0, gk), ((Neval - Nk)/Neval, Nk/Neval))
        # check stopping criterion
        # println(meet_stop_crit())

        # save values
        λset[k] = λk
        # αset[k] = αk
        h0 = hk
    end

    return λset # , αset
end