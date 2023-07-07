
# The primary function for calling the particle filter from R
function run_inference_exact(
    n_days,
    n_steps_per_day,
    num_particles,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    epsilons,

    true_occupancy_matrix
)
    # Julia and R do not play along nicely with these parameters. Ensure they are rounded integers.
    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)
    num_particles = round(Int64, num_particles)

    n_steps = n_days * n_steps_per_day


    # Load in the bootstrapped data
    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)
    
    forecast_start_day = get_forecast_start_day(true_occupancy_matrix[:, 1])

    samples_cache = make_cached_samples(group_params[1], n_steps_per_day)

    # Create the particles, initialising values from the prior distribution
    particles = [create_prior(n_steps) for i in 1:num_particles]

    # Produce a set of case curves that will be associated with a particle index (and not actually selected for)
    context_case_curves = Array{Array{Int32}}(undef, num_particles)
    for i in 1:num_particles
        context_case_curves[i] = case_curves[:, sample(1:size(case_curves, 2))]
    end

    results_table = DataFrame(
        particle = Int[],
        day = Int[],

        sim_ward = Int[],
        sim_ward_outbreak = Int[], 
        sim_ward_progression = Int[],
        sim_ICU = Int[],

        adj_pr_hosp = Float64[],
        adj_los = Float64[],

        log_importation_rate = Float64[],
        log_clearance_rate = Float64[]
    )

    # Primary simulation loop
    for d in 1:n_days
        #if d % 25 == 0
        println("Stepping at day $d")
        #end

        t = (d - 1) * n_steps_per_day + 1
        is_forecast = d >= forecast_start_day

        # Simulation context for day d
        ctx_day = pf_context(
            n_steps_per_day, n_steps, t,
            time_varying_estimates[1], group_params[1],
            is_forecast, samples_cache, context_case_curves
        )


        
        # Decide whether to step forward without resampling (when no observation is available)
        # or with resampling (where an observation is present)
        if true_occupancy_matrix[d, 1] < -0.5
            Threads.@threads for k in 1:num_particles
                particles[k] = pf_step(particles[k], ctx_day, k, MersenneTwister())
            end
        else
            next_particles = Vector{pf_state}(undef, num_particles)

            for eps in epsilons
                accepted = zeros(Bool, num_particles)

                Threads.@threads for k in 1:num_particles
                    n_try = 0
        
                    while !accepted[k] && n_try <= 500
                        n_try += 1
    
                        r = sample(1:num_particles)
            
                        next_state = pf_step(particles[r], ctx_day, r, MersenneTwister())
                        does_fit_exact = pf_prob_obs(1, ctx_day, next_state, true_occupancy_matrix[d, :], eps)
    
                        if does_fit_exact
                            next_particles[k] = next_state
                            accepted[k] = true
                        end
                    end
        
                end

                if all(accepted)
                    break
                end

            end


            particles = copy(next_particles)
        end



        # Save the results
        for p in 1:num_particles
            particle_p = particles[p]

            push!(
                results_table,
                Dict(
                    :particle => p,
                    :day => d,

                    :sim_ward => get_sim_total_ward_occupancy(particle_p, t),
                    :sim_ward_outbreak => get_sim_outbreak_occupancy(particle_p, t),
                    :sim_ward_progression => get_sim_progression_occupancy(particle_p, t),
                    :sim_ICU => get_sim_total_ICU_occupancy(particle_p, t),
                    :adj_pr_hosp => particle_p.adj_pr_hosp,
                    :adj_los => particle_p.adj_los,
                    :log_clearance_rate => particle_p.log_ward_clearance_rate,
                    :log_importation_rate => particle_p.log_ward_importation_rate
                )
            )
        end
    end

    return results_table
end