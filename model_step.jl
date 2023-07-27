


function model_process(p_ix, params, ctx, rng)
    state = create_model_state(params, ctx.n_steps, ctx.n_days)

    for a in 1:def_n_age_groups
        group_params = ctx.group_params[a]
        group_samples = get_cached_samples(ctx.delay_samples_cache[a], state.adj_los)

        for d in 1:ctx.n_days
            t_start = (d - 1) * ctx.n_steps_per_day + 1

            for t in t_start:(t_start + ctx.n_steps_per_day - 1)
                model_step_t(t, d, p_ix, a, rng, state, ctx, group_params, group_samples)
            end
        end
    end

    run_ward_epidemic!(
        state.epidemic, ctx.n_days,
        ctx.case_curves[p_ix],
        state.log_ward_importation_rate,
        state.log_ward_clearance_rate
    )

    return state
end

function model_step_t(t, d, p_ix, a, rng, state, ctx, group_params, group_samples)
    if t % ctx.n_steps_per_day == 1
        cases = ctx.case_curves[p_ix][d]
        pr_age_and_hosp = logistic(
            state.adj_pr_hosp +
            logit.(
                ctx.time_varying_estimates.pr_age_given_case[a, d] * 
                ctx.time_varying_estimates.pr_hosp[a, d]
            )
        )

        state.arr_all[a, t, c_symptomatic, s_transitions] += rand(Binomial(cases, pr_age_and_hosp))
    end


    pr_ICU = ctx.time_varying_estimates.pr_ICU[a, max((t - 1) ÷ ctx.n_steps_per_day - 4, 1)]

    arr_age_group_view = @view state.arr_all[a, :, :, :]

    @inbounds process_age_group(
        t,
        arr_age_group_view,
                
        group_params,
        group_samples,
        ctx.n_steps, ctx.n_steps_per_day,

        pr_ICU,

        rng
    )

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