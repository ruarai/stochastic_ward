
using Distributions
using StatsFuns
using DataFrames
using Random

using ParticleFilters

include("globals.jl")
include("group_parameters.jl")
include("progression.jl")
include("progression_pf.jl")

include("threaded_predict_model.jl")
include("resampler.jl")

function run_inference(
    n_days,
    n_steps_per_day,
    num_particles,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    true_occupancy_matrix
)

    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)

    n_steps = n_days * n_steps_per_day

    num_particles = round(Int64, num_particles)

    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)
    

    rng = MersenneTwister()


    predict_model = ThreadedPredictModel{pf_state}(pf_step)

    reweight_model = ReweightModel(pf_prob_obs)

    resample_model = LowVarianceResamplerDebug(num_particles)

    filter = BasicParticleFilter(predict_model, reweight_model, resample_model, num_particles, MersenneTwister())


    particles = ParticleCollection(
        [create_prior(group_params, time_varying_estimates, case_curves, n_steps, n_steps_per_day) for i in 1:num_particles]
    );

    sim_ward = zeros(n_days, num_particles)
    sim_ICU = zeros(n_days, num_particles)


    selected_adj_pr_hosp = zeros(n_days, num_particles)
    selected_adj_los_scale = zeros(n_days, num_particles)
    selected_adj_los_shape = zeros(n_days, num_particles)
    selected_cases = zeros(n_days, num_particles)

    for d in 1:n_days
        if d == n_days - 28
            println("Reinitialising case curves across particles")
            particles = reinitialise_case_curves(particles, num_particles, case_curves)
        end


        if true_occupancy_matrix[d, 1] < -0.5
            println("Stepping without inference at day $d")
            next_step = predict(filter, particles, 0.0, rng)
            particles = ParticleCollection(next_step)
        else
            println("Stepping with inference at day $d")
            particles = update(filter, particles, 0.0, true_occupancy_matrix[d, :]);
        end


        for p in 1:num_particles
            particle_p = particle(particles, p)

            t = (d - 1) * n_steps_per_day + 1

            sim_ward[d, p] = get_total_ward_occupancy(particle_p.arr_all, t)
            sim_ICU[d, p] = get_total_ICU_occupancy(particle_p.arr_all, t)

            selected_adj_pr_hosp[d, p] = particle_p.adj_pr_hosp
            selected_adj_los_scale[d, p] = particle_p.adj_los_scale
            selected_adj_los_shape[d, p] = particle_p.adj_los_shape


            selected_cases[d, p] = particle_p.case_curve[d]
        end
    end



    return (
        sim_ward = sim_ward, sim_ICU = sim_ICU,
        selected_adj_pr_hosp = selected_adj_pr_hosp,
        selected_adj_los_scale,
        selected_adj_los_shape,
        selected_cases
    )

end

function create_prior(
    group_params, time_varying_estimates, case_curves,
    n_steps, n_steps_per_day
)

    group_param_sample = sample(group_params)
    time_varying_sample = sample(time_varying_estimates)

    case_curve = case_curves[:, sample(1:size(case_curves, 2))]

    adj_pr_hosp = rand(Normal(0, 0.4))
    adj_los_shape = rand(Normal(0, 0.4))
    adj_los_scale = rand(Normal(0, 0.4))

    return pf_state(
        zeros(def_n_age_groups, n_steps, def_n_compartments, def_n_slots),
        1,
        n_steps,
        n_steps_per_day,
        adj_pr_hosp,
        adj_los_scale,
        adj_los_shape,
        group_param_sample,
        time_varying_sample,
        case_curve
    )
end

function reinitialise_case_curves(particles, num_particles, case_curves)
    particle_vec = Vector{pf_state}(undef, num_particles)

    for i in 1:num_particles
        pf_state_old = particle(particles, i)

        particle_vec[i] = pf_state(
            pf_state_old.arr_all,
            pf_state_old.t,
            pf_state_old.n_steps,
            pf_state_old.n_steps_per_day,

            pf_state_old.adj_pr_hosp,
            pf_state_old.adj_los_scale,
            pf_state_old.adj_los_shape,

            pf_state_old.group_params,
            pf_state_old.time_varying_estimates,
            case_curves[:, sample(1:size(case_curves, 2))]
        )
    end

    return ParticleCollection(particle_vec)
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