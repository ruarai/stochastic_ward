
struct pf_context
    n_steps_per_day::Int64
    n_steps::Int64

    t::Int64

    time_varying_estimates::NamedTuple{(:pr_age_given_case, :pr_hosp, :pr_ICU), Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}}
    group_params::Vector{group_parameters}
end