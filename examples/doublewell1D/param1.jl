### potential function and gradients of 1D-state 2D-parameter double well model

V(x, θ) = (θ[1] * x.^2) / 2 - (θ[2] * x.^4) / 4 - 1 # with neg sign
∇xV(x, θ) = θ[1] * x - θ[2] * x.^3
∇θV(x, θ) = [x^2 / 2, -x^4 / 4]