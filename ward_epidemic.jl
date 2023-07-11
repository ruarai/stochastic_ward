

struct ward_epidemic
    S::Matrix{Int64}
    I::Matrix{Int64}
    Q::Matrix{Int64}
end

function step_ward_epidemic(x, d, cases, log_importation_rate, log_clearance_rate)
    n_steps = 100
    dt = 1 / n_steps

    importation_rate_standardised = exp(log_importation_rate) / def_n_ward_epidemic


    for i in 1:def_n_ward_epidemic
        if d == 1
            S = ward_steady_state_size
            I = 0
            Q = 0
        else
            S = x.S[d - 1, i]
            I = x.I[d - 1, i]
            Q = x.Q[d - 1, i]
        end

        for t in 1:n_steps
            rate_new_I = cases * importation_rate_standardised
        
            rate_S_to_I = I * 0.9 / (I + S + 1e-10)
            rate_I_to_Q = 0.5
            rate_clearance_Q = exp(log_clearance_rate)
        
        
            S_to_I = rand(Binomial(S, expm1(rate_S_to_I * dt)))
        
            new_I = rand(Poisson(rate_new_I * dt))
        
            I_to_Q = rand(Binomial(I, expm1(rate_I_to_Q * dt)))
            clearance_Q = rand(Binomial(Q, rate_clearance_Q * dt))
        
            S = S - S_to_I
            I = I + new_I + S_to_I - I_to_Q
            Q = Q + I_to_Q - clearance_Q

            if I + Q == 0
                S = ward_steady_state_size
            end
        end

        x.S[d, i] = S
        x.I[d, i] = I
        x.Q[d, i] = Q
    end

    return ward_epidemic(x.S, x.I, x.Q)
end
