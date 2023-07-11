

# Contains read-only data shared across all model runs
struct model_context
    # The number of simulation steps per day
    n_steps_per_day::Int64

    # The total number of simulation steps
    n_steps::Int64

    # The current simulation step
    t::Int64

    # Contextual data on time-varying probabilties and group-level parameters that
    # are not selected for by the particle filter
    time_varying_estimates::NamedTuple{(:pr_age_given_case, :pr_hosp, :pr_ICU), Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}}
    group_params::Vector{group_parameters}

    # Is the current time-step a forecasting step?
    is_forecast::Bool

    # Cached delay samples constructed from group_params
    delay_samples_cache::Vector{group_delay_samples_cache}

    # The array of ensemble case forecasts (and backcasts)
    # Each element i in the array is associated with particle i (and as such is not selected for)
    case_curves::Array{Array{Int32}}
end