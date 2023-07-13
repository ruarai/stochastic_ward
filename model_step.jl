


function model_step(state, ctx, p_ix, rng)

    t_start = ctx.t
    n_steps = ctx.n_steps
    n_steps_per_day = ctx.n_steps_per_day

    for a in 1:def_n_age_groups
        group_params = ctx.group_params[a]
        group_samples = get_cached_samples(ctx.delay_samples_cache[a], state.adj_los)


        for t in t_start:(t_start + n_steps_per_day - 1)
            if t % n_steps_per_day == 1
                d = ((t - 1) ÷ n_steps_per_day) + 1

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


            pr_ICU = ctx.time_varying_estimates.pr_ICU[a, max((t - 1) ÷ n_steps_per_day - 4, 1)]

            arr_age_group_view = @view state.arr_all[a, :, :, :]

            @inbounds process_age_group(
                t,
                arr_age_group_view,
                        
                group_params,
                group_samples,
                n_steps, n_steps_per_day,

                pr_ICU,

                rng
            )
        end
    end

    t_start_day = ((t_start - 1) ÷ n_steps_per_day) + 1
    stepped_epidemic = step_ward_epidemic(
        state.epidemic, t_start_day,
        ctx.case_curves[p_ix][t_start_day],
        state.log_ward_importation_rate,
        state.log_ward_clearance_rate
    )

    return model_state(
            state.arr_all,
    
            state.adj_pr_hosp,
            state.adj_los,
    
            state.log_ward_importation_rate,
            state.log_ward_clearance_rate,
    
            stepped_epidemic
        )
end