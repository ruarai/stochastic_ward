


function model_step(state, ctx, p_ix, rng)
    arr_all = copy(state.arr_all)

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

                arr_all[a, t, c_symptomatic, s_transitions] += rand(Binomial(cases, pr_age_and_hosp))
            end


            pr_ICU = ctx.time_varying_estimates.pr_ICU[a, max((t - 1) ÷ n_steps_per_day - 4, 1)]

            arr_age_group_view = @view arr_all[a, :, :, :]

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

    stepped_epidemic = step_ward_epidemic(
        state.epidemic,
        ctx.case_curves[p_ix][((t_start - 1) ÷ n_steps_per_day) + 1],
        state.log_ward_importation_rate,
        state.log_ward_clearance_rate
    )

    if ctx.is_forecast
        # Return the updated state
        # Do not vary parameters if we are in the forecasting period
        return model_state(
            arr_all,
    
            state.adj_pr_hosp,
            state.adj_los,
    
            state.log_ward_importation_rate,
            state.log_ward_clearance_rate,
    
            stepped_epidemic
        )
    else
        # Return the updated state
        # Adjust the time-varying parameters
        return model_state(
            arr_all,
    
            state.adj_pr_hosp + rand(Normal(0, 0.05)),
            state.adj_los + rand(Normal(0, 0.05)),
    
            state.log_ward_importation_rate + rand(Normal(0, 0.01)),
            state.log_ward_clearance_rate + rand(Normal(0, 0.01)),
    
            stepped_epidemic
        )
    end
end