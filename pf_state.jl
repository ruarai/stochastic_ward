
# The 'state' of each particle, with this selected for by the particle filter
struct pf_state

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


    obs_c::Float64
end