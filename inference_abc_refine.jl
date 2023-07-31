

# The primary function for calling from R
function run_inference_refine(
    n_days,
    n_steps_per_day,

    num_samples,
    thresholds,
    rejection_probability_threshold,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    parameters_prior,

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

    # Data shared to all model runs
    context = model_context(
        n_steps_per_day, n_steps, n_days,
        time_varying_estimates[1], group_params[1],
        samples_cache, context_case_curves
    )


    model_states = Vector{model_state}(undef, num_samples)
    n_rejected = zeros(Int64, length(thresholds))
    n_accepted = zeros(Int64, length(thresholds))

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

                row_ix = rand(1:size(parameters_prior, 1))
                sigma = [
                    parameters_prior.adj_pr_hosp[row_ix],
                    parameters_prior.adj_los[row_ix],
                    parameters_prior.log_importation_rate[row_ix]
                ]

                state = model_process(s, sigma, context, Random.MersenneTwister())

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

    return simulations_output
end


function do_reject_output(state, true_occupancy_matrix, threshold, n_days, n_steps_per_day)

    rejected = false
    
    for d in 1:n_days
        t = (d - 1) * n_steps_per_day + 1
        if true_occupancy_matrix[d, 1] > -0.5

            sim_ward = get_total_ward_occupancy(state, t, d)
            sim_ICU = get_total_ICU_occupancy(state, t)

            known_ward = true_occupancy_matrix[d, 1]
            known_ICU = true_occupancy_matrix[d, 2]

            error_ward = abs(known_ward - sim_ward)
            error_ICU = abs(known_ICU - sim_ICU)

            if error_ward > max(known_ward * threshold, 2) || error_ICU > max(known_ICU * threshold * 1.5, 4)
                rejected = true

                break
            end
        end
    end

    return rejected
end