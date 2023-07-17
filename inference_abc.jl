
using Distributions
using StatsFuns
using DataFrames
using Random

include("globals.jl")
include("group_parameters.jl")

include("ward_epidemic.jl")

include("progression.jl")

include("model_step.jl")
include("observation.jl")

include("group_delay_samples_cache.jl")

include("model_state.jl")
include("model_context.jl")


# The primary function for calling from R
function run_inference(
    n_days,
    n_steps_per_day,
    num_samples,
    thresholds,
    rejection_probability_threshold,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    mean_log_ward_importation_rate,

    true_occupancy_matrix
)
    # Julia and R do not play along nicely with these parameters. Ensure they are rounded integers.
    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)
    num_samples = round(Int64, num_samples)

    n_steps = n_days * n_steps_per_day


    # Load in the bootstrapped data
    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)

    # Create a cache of length of stay durations for the progression model
    samples_cache = make_cached_samples(group_params[1], n_steps_per_day)
    

    # Produce a set of case curves that will be associated with a sample index (and not actually selected for)
    context_case_curves = Array{Array{Int32}}(undef, num_samples)
    for i in 1:num_samples
        context_case_curves[i] = case_curves[:, sample(1:size(case_curves, 2))]
    end

    model_states = Vector{model_state}(undef, num_samples)
    n_rejected = zeros(Int64, length(thresholds))
    n_accepted = zeros(Int64, length(thresholds))

    # Data shared to all model runs
    context = model_context(
        n_steps_per_day, n_steps, n_days,
        time_varying_estimates[1], group_params[1],
        samples_cache, context_case_curves
    )

    # Arbitrary-ish rule to 'give up' when failure rate is too high
    do_give_up(accepted, rejected) = accepted + rejected > (1.0 / rejection_probability_threshold) * 4.0 && accepted < rejection_probability_threshold * (accepted + rejected)

    for i in eachindex(thresholds)
        println(thresholds[i])
        if i > 1
            println("Previous acceptance probability: ", n_accepted[i - 1] / (n_rejected[i - 1] + n_accepted[i - 1]) )
        end

        # Split into threads. Try to produce num_samples outputs
        Threads.@threads for s in 1:num_samples
            rejected = true
    
            # Keep attempting to produce a model output (until we give up)
            while rejected && !do_give_up(n_accepted[i], n_rejected[i])

                # Run the model
                state = model_process(s, context, Random.MersenneTwister(), mean_log_ward_importation_rate)

                rejected = do_reject_output(state, true_occupancy_matrix, thresholds[i], n_days, n_steps_per_day)

                if !rejected
                    model_states[s] = state
                    n_accepted[i] += 1

                    if n_accepted[i] % 100 == 0
                        println(n_accepted[i], " / ", num_samples, ", acceptance probability: ", n_accepted[i] / (n_rejected[i] + n_accepted[i]))
                    end
                else
                    n_rejected[i] += 1
                end
            end

            if do_give_up(n_accepted[i], n_rejected[i])
                break
            end
        end

        if !do_give_up(n_accepted[i], n_rejected[i])
            break
        end
    end



    parameters_output = DataFrame(
        sample = Int[],

        adj_pr_hosp = Float64[],
        adj_los = Float64[],

        log_importation_rate = Float64[],
        log_clearance_rate = Float64[],
    )

    simulations_output = DataFrame(
        sample = Int[],
        day = Int[],

        sim_ward = Int[],
        sim_ward_outbreak = Int[], 
        sim_ward_progression = Int[],
        sim_ICU = Int[]
    )

    for s in 1:num_samples
        sample_s = model_states[s]


        push!(
            parameters_output,
            Dict(
                :sample => s,

                :adj_pr_hosp => sample_s.adj_pr_hosp,
                :adj_los => sample_s.adj_los,
                :log_importation_rate => sample_s.log_ward_importation_rate,
                :log_clearance_rate => sample_s.log_ward_clearance_rate,
            )
        )

        for d in 1:n_days
            t = (d - 1) * n_steps_per_day + 1

            push!(
                simulations_output,
                Dict(
                    :sample => s,
                    :day => d,

                    :sim_ward => get_total_ward_occupancy(sample_s, t, d),
                    :sim_ward_outbreak => get_ward_outbreak_occupancy(sample_s, d),
                    :sim_ward_progression => get_ward_progression_occupancy(sample_s, t),
                    :sim_ICU => get_total_ICU_occupancy(sample_s, t),
                )
            )
        end
    end

    return (simulations = simulations_output, parameters = parameters_output)
end


function read_group_parameter_samples(group_parameters_table)

    n_samples = round(Int32, maximum(group_parameters_table.sample))
    println("Reading in $n_samples group_parameters across $def_n_age_groups age groups")

    out_samples = Vector{Vector{group_parameters}}(undef, n_samples)

    for i in 1:n_samples
        out_samples[i] = Vector{group_parameters}(undef, def_n_age_groups)
    end

    for row in eachrow(group_parameters_table)
        i_sample = round(Int32, row.sample)
        i_age_group = findfirst(x -> x == row.age_group, age_groups)

        out_samples[i_sample][i_age_group] = group_parameters(
            row.pr_ward_to_discharge,
            row.pr_ward_to_ICU,

            row.pr_ICU_to_discharge,
            row.pr_ICU_to_postICU,

            row.pr_postICU_to_death,
        
            Gamma(row.shape_onset_to_ward, row.scale_onset_to_ward),
            
            Gamma(row.shape_ward_to_discharge, row.scale_ward_to_discharge),
            Gamma(row.shape_ward_to_death, row.scale_ward_to_death),
            Gamma(row.shape_ward_to_ICU, row.scale_ward_to_ICU),
            
            Gamma(row.shape_ICU_to_death, row.scale_ICU_to_death),
            Gamma(row.shape_ICU_to_discharge, row.scale_ICU_to_discharge),
            Gamma(row.shape_ICU_to_postICU, row.scale_ICU_to_postICU),

            Gamma(row.shape_postICU_to_death, row.scale_postICU_to_death),
            Gamma(row.shape_postICU_to_discharge, row.scale_postICU_to_discharge)
        )
    end

    return out_samples
end


function read_time_varying_estimates(time_varying_estimates_table, n_days)

    n_days = round(Int32, n_days)

    n_samples = round(Int32, maximum(time_varying_estimates_table.bootstrap))
    println("Reading in $n_samples time-varying estimate samples across $def_n_age_groups age groups")

    out_samples = Vector{}(undef, n_samples)

    for i in 1:n_samples
        out_samples[i] = (
            pr_age_given_case = zeros(def_n_age_groups, n_days),
            pr_hosp = zeros(def_n_age_groups, n_days),
            pr_ICU = zeros(def_n_age_groups, n_days)
        )
    end


    for row in eachrow(time_varying_estimates_table)
        i_sample = round(Int32, row.bootstrap)
        i_age_group = findfirst(x -> x == row.age_group, age_groups)
        i_t = round(Int32, row.t)

        out_samples[i_sample].pr_age_given_case[i_age_group, i_t] = row.pr_age_given_case
        out_samples[i_sample].pr_hosp[i_age_group, i_t] = row.pr_hosp
        out_samples[i_sample].pr_ICU[i_age_group, i_t] = row.pr_ICU
    end



    return out_samples
    
end