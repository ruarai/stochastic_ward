

struct ward_epidemic
    S::Vector{Int64}
    I::Vector{Int64}
    Q::Vector{Int64}
end


function step_ward_epidemic(x, cases, log_importation_rate, log_clearance_rate)
    n_steps = 100
    dt = 1 / n_steps

    importation_rate_standardised = exp(log_importation_rate) / def_n_ward_epidemic

    S = copy(x.S)
    I = copy(x.I)
    Q = copy(x.Q)

    for i in 1:def_n_ward_epidemic
        for t in 1:n_steps
            rate_new_I = (cases * importation_rate_standardised)# * (I[i] == 0 && Q[i] == 0)
        
            rate_S_to_I = I[i] * 0.9 / (I[i] + S[i] + 1e-10)
            rate_I_to_Q = 0.5
            rate_clearance_Q = exp(log_clearance_rate)
        
        
            S_to_I = rand(Binomial(S[i], expm1(rate_S_to_I * dt)))
        
            new_I = rand(Poisson(rate_new_I * dt))
        
            I_to_Q = rand(Binomial(I[i], expm1(rate_I_to_Q * dt)))
            clearance_Q = rand(Binomial(Q[i], rate_clearance_Q * dt))
        
            S[i] = S[i] - S_to_I
            I[i] = I[i] + new_I + S_to_I - I_to_Q
            Q[i] = Q[i] + I_to_Q - clearance_Q

            if I[i] + Q[i] == 0
                S[i] = ward_steady_state_size
            end
        end
    end

    return ward_epidemic(S, I, Q)
end
