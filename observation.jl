
# Returns the distribution of the observation model with corresponding mean
function observation_model(mean)
    return truncated(Laplace(mean, 0.5 + mean * 0.005), 0, Inf)
end

# ParticleFilters.jl compatible probability of observation functio
# xt - previous state, not used
# ctx - current context
# xt1 - current state
# yt1 - current observation
function pf_prob_obs(xt, ctx, xt1, yt1)
    sim_ward = get_total_ward_occupancy(xt1, ctx.t)
    sim_ICU = get_total_ICU_occupancy(xt1, ctx.t)

    true_ward = round(Int32, yt1[1])
    true_ICU = round(Int32, yt1[2])

    return pdf(observation_model(true_ward), sim_ward) * pdf(observation_model(true_ICU), sim_ICU)
end

function get_ward_outbreak_occupancy(pf_state, t)
    return sum(pf_state.epidemic.Q)
end

function get_ward_progression_occupancy(pf_state, t)
    arr_all = pf_state.arr_all

    return sum(arr_all[:, t, c_ward, s_occupancy]) + 
        sum(arr_all[:, t, c_postICU_to_death, s_occupancy]) + 
        sum(arr_all[:, t, c_postICU_to_discharge, s_occupancy]) 
end

function get_total_ward_occupancy(pf_state, t)
    return get_ward_outbreak_occupancy(pf_state, t) + get_ward_progression_occupancy(pf_state, t)
end


function get_total_ICU_occupancy(pf_state, t)
    arr_all = pf_state.arr_all

    return sum(arr_all[:, t, c_ICU, s_occupancy])
end

function get_sim_ward_progression_occupancy(pf_state, t)
    return round(Int64, rand(observation_model(get_ward_progression_occupancy(pf_state, t))))
end

function get_sim_ward_outbreak_occupancy(pf_state, t)
    return round(Int64, rand(observation_model(get_ward_outbreak_occupancy(pf_state, t))))
end

function get_sim_total_ward_occupancy(pf_state, t)
    return get_sim_ward_progression_occupancy(pf_state, t) + 
         get_sim_ward_outbreak_occupancy(pf_state, t)
end

function get_sim_total_ICU_occupancy(pf_state, t)
    return round(Int64, rand(observation_model(get_total_ICU_occupancy(pf_state, t))))
end