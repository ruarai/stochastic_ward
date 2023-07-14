
# The 'state' of each sample
struct model_state

    # The total progression simulation data array
    # With dimensions of age-group, time, compartment and 'slot' (occupancy/transitions)
    arr_all::Array{Int32, 4}

    # Adjustment factors on probability of hospitalisation and length-of-stay
    adj_pr_hosp::Float64
    adj_los::Float64

    # Parameters for ward_epidemic
    log_ward_importation_rate::Float64
    log_ward_clearance_rate::Float64

    # The state of the 'within-hospital outbreaks' component
    epidemic::ward_epidemic
end


function create_prior(
    n_steps, n_days, mean_log_ward_importation_rate
)
    adj_pr_hosp = rand(Normal(0, 0.2))
    adj_los = rand(Normal(0, 0.8))

    log_ward_importation_rate = rand(Normal(mean_log_ward_importation_rate, 0.25))
    log_ward_clearance_rate = log(1 / rand(TruncatedNormal(7, 4, 3, 14)))

    return model_state(
        zeros(def_n_age_groups, n_steps, def_n_compartments, def_n_slots),

        adj_pr_hosp, adj_los,
        log_ward_importation_rate, log_ward_clearance_rate,

        ward_epidemic(
            zeros(Int64, n_days, def_n_ward_epidemic),
            zeros(Int64, n_days, def_n_ward_epidemic),
            zeros(Int64, n_days, def_n_ward_epidemic),
        )
    )
end
