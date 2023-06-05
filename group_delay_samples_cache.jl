

struct group_delay_samples_cache
    x_los_adj::Vector{Float64}

    delay_samples::Vector{group_delay_samples}
end

function make_cached_samples(group_params, n_steps_per_day)
    results = Vector{group_delay_samples_cache}(undef, length(group_params))

    for i in eachindex(group_params)
        x_los_adj = collect(-3:0.05:3)
        samples_los_adj = Vector{group_delay_samples}(undef, length(x_los_adj))
        
        for j in eachindex(x_los_adj)
            samples_los_adj[j] = make_delay_samples(group_params[i], 1024, n_steps_per_day, x_los_adj[j])
        end

        results[i] = group_delay_samples_cache(x_los_adj, samples_los_adj)
    end

    return results
end


function get_cached_samples(cache, los_adj)
    findnearest(A::AbstractArray,t) = findmin(abs.(A .- t))[2]

    ix = findnearest(cache.x_los_adj, los_adj)

    return cache.delay_samples[ix]
end