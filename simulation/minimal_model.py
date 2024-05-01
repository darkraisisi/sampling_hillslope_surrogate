import numpy as np

def step(B, D, g, warm_up=0):
    dt: float = 0.5     # Time step in years
    # Define the physical parameters
    r, c, i, d, s = 2.1, 2.9, -0.7, 0.04, 0.4 
    Wo, a, Et, Eo, k, b, C = 5e-4, 4.02359478109, 0.021, 0.084, 0.05, 0.28, 1e-4
    alpha = np.log(Wo/C)/a

    def dX_dt(B,D,g):
        dB_dt_step = (1-(1-i)*np.exp(-1*D/d))*(r*B*(1-B/c))-g*B/(s+B)
        dD_dt_step = Wo*np.exp(-1*a*D)-np.exp(-1*B/b)*(Et+np.exp(-1*D/k)*(Eo-Et))-C
        return dB_dt_step, dD_dt_step
    
    def apply_change(B, delta_B, D, delta_D, clip=True):
        if clip:
            B = np.clip(B + (delta_B * dt), 0.0, c)
            D = np.clip(D + (delta_D * dt), 0.0, alpha)
        else:
            B += (delta_B * dt)
            D += (delta_D * dt)
        return B, D
    
    # Warm-up steps plus final true step.
    for i in range(warm_up+1):
        delta_B, delta_D = dX_dt(B,D,g)
        print(f"({i}) Delta B: {delta_B:.3f}, Delta D: {delta_D:.10f}")
        B, D = apply_change(B, delta_B, D, delta_D)
        print(f"({i}) New B: {B:.3f}, New D: {D:.10f}")

    return B, D, delta_B, delta_D