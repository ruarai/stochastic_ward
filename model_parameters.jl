

struct model_parameters
    adj_pr_hosp::Float64
    adj_los::Float64

    log_ward_importation_rate::Float64
end

function create_prior(model_priors)
    adj_pr_hosp = rand(model_priors[1])
    adj_los = rand(model_priors[2])

    log_ward_importation_rate = rand(model_priors[3])

    return [adj_pr_hosp, adj_los, log_ward_importation_rate]
end

function create_model_state(params, n_steps, n_days)
    return model_state(
        zeros(def_n_age_groups, n_steps, def_n_compartments, def_n_slots),

        params[1], params[2],
        params[3], log(1/14),

        ward_epidemic(zeros(n_days))
    )
end

function perturb_parameters(params, model_perturbs)
    return params .+ rand.(model_perturbs)
end

function prior_prob(params, model_priors)
    prod(cdf.(model_priors, params))
end

function perturb_prob(params, params_t_m_1, model_perturbs)
    prod(cdf.(model_perturbs, params_t_m_1 - params))
end
