
struct pf_state
    arr_all::Array{Int32, 4}

    adj_pr_hosp::Float64
    adj_los::Float64

    ward_importation_rate::Float64
    ward_clearance_rate::Float64

    case_curve::Vector{Int32}

    epidemic::ward_epidemic
end