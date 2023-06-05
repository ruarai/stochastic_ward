
struct pf_state
    arr_all::Array{Int32, 4}

    adj_pr_hosp::Float64
    adj_los::Float64

    log_ward_importation_rate::Float64
    log_ward_clearance_rate::Float64

    epidemic::ward_epidemic
end