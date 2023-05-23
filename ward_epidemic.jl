

struct ward_epidemic
    S::Vector{Int64}
    I::Vector{Int64}
    Q::Vector{Int64}
end


function step_ward_epidemic(x, cases, importation_rate, clearance_rate)
    n_steps = 100
    dt = 1 / n_steps

    importation_rate_standardised = importation_rate / def_n_ward_epidemic

    S = copy(x.S)
    I = copy(x.I)
    Q = copy(x.Q)

    for i in 1:def_n_ward_epidemic
        for t in 1:n_steps
            pop_size = S[i] + I[i] + Q[i]

            rate_new_S = 0.01 * (pop_size < ward_steady_state_size) * (I[i] == 0 && Q[i] == 0)
            rate_new_I = (cases * importation_rate_standardised) * (I[i] == 0 && Q[i] == 0)#1e-4
        
            rate_S_to_I = I[i] * 0.9 / (I[i] + S[i] + 1e-10)
            rate_I_to_Q = 0.5
            rate_clearance_Q = clearance_rate
        
        
            new_S = rand(Poisson(rate_new_S * dt))
            S_to_I = rand(Binomial(S[i], expm1(rate_S_to_I * dt)))
        
            new_I = rand(Poisson(rate_new_I * dt))
        
            I_to_Q = rand(Binomial(I[i], expm1(rate_I_to_Q * dt)))
            clearance_Q = rand(Binomial(Q[i], rate_clearance_Q * dt))
        
            S[i] = S[i] + new_S - S_to_I
            I[i] = I[i] + new_I + S_to_I - I_to_Q
            Q[i] = Q[i] + I_to_Q - clearance_Q
        end
    end

    return ward_epidemic(S, I, Q)
end
