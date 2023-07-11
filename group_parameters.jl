


struct group_parameters
    pr_ward_to_discharge::Float64
    pr_ward_to_ICU::Float64

    pr_ICU_to_discharge::Float64
    pr_ICU_to_postICU::Float64

    pr_postICU_to_death::Float64
  
    dist_symptomatic_to_ward::Gamma

    dist_ward_to_discharge::Gamma
    dist_ward_to_death::Gamma
    dist_ward_to_ICU::Gamma

    dist_ICU_to_death::Gamma
    dist_ICU_to_postICU::Gamma
    dist_ICU_to_discharge::Gamma
    
    dist_postICU_to_death::Gamma
    dist_postICU_to_discharge::Gamma
end

struct group_delay_samples
    symptomatic_to_ward::Vector{Int32}

    ward_to_discharge::Vector{Int32}
    ward_to_death::Vector{Int32}
    ward_to_ICU::Vector{Int32}

    ICU_to_death::Vector{Int32}
    ICU_to_postICU::Vector{Int32}
    ICU_to_discharge::Vector{Int32}
    
    postICU_to_death::Vector{Int32}
    postICU_to_discharge::Vector{Int32}
end

function sample_delay_dist(dist, n_samples, n_steps_per_day)
    return round.(Int32, rand(dist, n_samples) .* n_steps_per_day)
end

function adj_gamma(dist, adj_los)
    shape = dist.α
    scale = dist.θ

    shape_adj = exp(log(shape) + adj_los)
    #scale_adj = exp(log(scale) + adj_los / 2)

    return Gamma(shape_adj, scale)
end

function make_delay_samples(
    group_params, n_samples, n_steps_per_day,
    adj_los
)
    group_delay_samples(
        sample_delay_dist(group_params.dist_symptomatic_to_ward, n_samples, n_steps_per_day),

        sample_delay_dist(adj_gamma(group_params.dist_ward_to_discharge, adj_los) , n_samples, n_steps_per_day),
        sample_delay_dist(adj_gamma(group_params.dist_ward_to_death, adj_los) , n_samples, n_steps_per_day),
        sample_delay_dist(adj_gamma(group_params.dist_ward_to_ICU, adj_los) , n_samples, n_steps_per_day),

        sample_delay_dist(group_params.dist_ICU_to_death, n_samples, n_steps_per_day),
        sample_delay_dist(group_params.dist_ICU_to_postICU, n_samples, n_steps_per_day),
        sample_delay_dist(group_params.dist_ICU_to_discharge, n_samples, n_steps_per_day),

        sample_delay_dist(adj_gamma(group_params.dist_postICU_to_death, adj_los) , n_samples, n_steps_per_day),
        sample_delay_dist(adj_gamma(group_params.dist_postICU_to_discharge, adj_los) , n_samples, n_steps_per_day)
    )

end