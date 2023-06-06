
using Distributions
using StatsFuns
using DataFrames
using Random

using ParticleFilters

include("globals.jl")
include("group_parameters.jl")

include("ward_epidemic.jl")

include("progression.jl")
include("progression_pf.jl")

include("particle_filter/includes.jl")

include("group_delay_samples_cache.jl")

include("pf_state.jl")
include("pf_context.jl")


# The primary function for calling the particle filter from R
function run_inference(
    n_days,
    n_steps_per_day,
    num_particles,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

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

    rng = MersenneTwister()

    # Initialise the components of the particle filter
    predict_model = ThreadedPredictModel{pf_state}(pf_step)
    reweight_model = ReweightModel(pf_prob_obs)
    resample_model = LowVarianceResamplerDebug(num_particles)
    filter = BasicParticleFilter2(predict_model, reweight_model, resample_model, num_particles, MersenneTwister())

    # Create the ParticleCollection, initialising values from the prior distribution
    particle_collection = ParticleCollection([create_prior(n_steps) for i in 1:num_particles]);

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
        log_clearance_rate = Float64[],

        obs_c = Float64[],

        weight = Float64[]
    )

    weights_memory = zeros(num_particles)

    # Primary simulation loop
    for d in 1:n_days
        if d % 25 == 0
            println("Stepping at day $d")
        end

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
            next_step = predict(filter, particle_collection, ctx_day, rng)
            particle_collection = ParticleCollection(next_step)
        else
            particle_collection = update(filter, particle_collection, ctx_day, true_occupancy_matrix[d, :])

            # Only update the weights in memory when we step with an observation
            weights_memory = get_weights(reweight_model, particle_collection, ctx_day, true_occupancy_matrix[d, :])
        end

        # Save the results
        for p in 1:num_particles
            particle_p = particle(particle_collection, p)

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
                    :log_importation_rate => particle_p.log_ward_importation_rate,
                    :obs_c => particle_p.obs_c,
                    :weight => weights_memory[p]
                )
            )
        end
    end

    return results_table
end

function create_prior(
    n_steps
)

    adj_pr_hosp = rand(Normal(0, 0.4))
    adj_los = rand(Normal(0, 0.4))

    log_ward_importation_rate = rand(Normal(-8, 1))
    log_ward_clearance_rate = log(1 / rand(TruncatedNormal(7, 4, 3, 14)))

    obs_c = rand(Normal(-2, 1))

    return pf_state(
        zeros(def_n_age_groups, n_steps, def_n_compartments, def_n_slots),

        adj_pr_hosp, adj_los,
        log_ward_importation_rate, log_ward_clearance_rate,

        ward_epidemic(
            fill(ward_steady_state_size, def_n_ward_epidemic),
            zeros(Int64, def_n_ward_epidemic),
            zeros(Int64, def_n_ward_epidemic),
        ),

        obs_c
    )
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


function get_forecast_start_day(occ_data)
    for i in reverse(2:length(occ_data))
        if occ_data[i] < -0.5 && occ_data[i - 1] > -0.5
            return i
        end
    end
    return -1
end