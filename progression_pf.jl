

function process_age_group(
    t, arr,

    group_params,
    group_samples,
    n_steps, n_steps_per_day,

    pr_ICU,

    rng
)
    for c in 1:def_n_compartments

        # Update the current occupancy value to reflect the value at the last timestep
        # counted at arr[ix(t - 1, c, s_occupancy)]
        # plus/minus any transitions (counted by arr[ix(t, c, s_occupancy)])
        if t > 1
            arr[t, c, s_occupancy] = arr[t, c, s_occupancy] + arr[t - 1, c, s_occupancy]
        end

        # Also increment occupancy by number of inward transitions
        arr[t, c, s_occupancy] = arr[t, c, s_occupancy] + arr[t, c, s_transitions]
    end

    transition_delay(
        c_symptomatic, c_ward,
        arr[t, c_symptomatic, s_transitions],
        t,
        arr,
        group_samples.symptomatic_to_ward,
        n_steps, n_steps_per_day,

        rng
    )

    pr_not_ICU = 1 - pr_ICU
    pr_discharge_given_not_ICU = group_params.pr_ward_to_discharge / (1 - group_params.pr_ward_to_ICU)
    pr_discharge_adj = pr_discharge_given_not_ICU * pr_not_ICU;

    transition_ward_next(
        t, arr,
        pr_discharge_adj, pr_ICU,

        group_samples,
        n_steps, n_steps_per_day,
        
        rng
    )

    transition_ICU_next(
        t, arr,
        group_params,
        group_samples,
        n_steps, n_steps_per_day,

        rng
    )

    transition_delay(
        c_postICU_to_death, c_died_postICU,
        arr[t, c_postICU_to_death, s_transitions],
        t,
        arr,
        group_samples.postICU_to_death,
        n_steps, n_steps_per_day,

        rng
    )
    transition_delay(
        c_postICU_to_discharge, c_discharged_postICU,
        arr[t, c_postICU_to_discharge, s_transitions],
        t,
        arr,
        group_samples.postICU_to_discharge,
        n_steps, n_steps_per_day,

        rng
    )
end

struct pf_state
    arr_all::Array{Int32, 4}
    t::Int32
    n_steps::Int32
    n_steps_per_day::Int32

    adj_pr_hosp::Float64
    adj_los::Float64

    group_params::Vector{group_parameters}
    time_varying_estimates::NamedTuple{(:pr_age_given_case, :pr_hosp, :pr_ICU), Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}}
    case_curve::Vector{Int32}

    epidemic::ward_epidemic
end

function pf_step(xt, u, rng)
    t_start = xt.t
    arr_all = copy(xt.arr_all)
    n_steps = xt.n_steps
    n_steps_per_day = xt.n_steps_per_day

    for t in t_start:(t_start + n_steps_per_day - 1)
        for a in 1:def_n_age_groups
            group_params = xt.group_params[a]
            group_samples = make_delay_samples(group_params, 512, n_steps_per_day, xt.adj_los)

            if t % n_steps_per_day == 1
                d = ((t - 1) ÷ n_steps_per_day) + 1

                cases = xt.case_curve[d]
                pr_age_and_hosp = logistic(
                    xt.adj_pr_hosp +
                    logit.(
                        xt.time_varying_estimates.pr_age_given_case[a, d] * 
                        xt.time_varying_estimates.pr_hosp[a, d]
                    )
                )

                arr_all[a, t, c_symptomatic, s_transitions] += rand(Binomial(cases, pr_age_and_hosp))
            end


            pr_ICU = xt.time_varying_estimates.pr_ICU[a, max((t - 1) ÷ n_steps_per_day - 4, 1)]

            arr_ref = @view arr_all[a, :, :, :]

            @inbounds process_age_group(
                t,
                arr_ref,
                        
                group_params,
                group_samples,
                n_steps, n_steps_per_day,

                pr_ICU,

                rng
            )
        end
    end

    stepped_epidemic = step_ward_epidemic(xt.epidemic, xt.case_curve[((t_start - 1) ÷ n_steps_per_day) + 1])

    
    return pf_state(
        arr_all,
        t_start + n_steps_per_day,
        xt.n_steps,
        xt.n_steps_per_day,

        xt.adj_pr_hosp + rand(Normal(0, 0.02)),
        xt.adj_los + rand(Normal(0, 0.02)),

        xt.group_params,
        xt.time_varying_estimates,
        xt.case_curve,

        stepped_epidemic
    )
end



function pf_prob_obs(xt, ut, xt1, yt1)
    sim_ward = get_total_ward_occupancy(xt1, xt1.t - 1)
    sim_ICU = get_total_ICU_occupancy(xt1, xt1.t - 1)

    true_ward = yt1[1]
    true_ICU = yt1[2]
    
    return pdf(Poisson(true_ward), sim_ward) * pdf(Poisson(true_ICU), sim_ICU)
end



function get_total_ward_occupancy(pf_state, t)
    arr_all = pf_state.arr_all

    return sum(arr_all[:, t, c_ward, s_occupancy]) + 
        sum(arr_all[:, t, c_postICU_to_death, s_occupancy]) + 
        sum(arr_all[:, t, c_postICU_to_discharge, s_occupancy]) +

        pf_state.epidemic.I + pf_state.epidemic.Q
end


function get_outbreak_occupancy(pf_state, t)
    return pf_state.epidemic.I + pf_state.epidemic.Q
end

function get_total_ICU_occupancy(pf_state, t)
    arr_all = pf_state.arr_all

    return sum(arr_all[:, t, c_ICU, s_occupancy])
end