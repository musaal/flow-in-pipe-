import torch

def compute_flow_parameters(positions, velocities, pipe_radius):
    """
    Given particle positions and velocities, compute flow parameters.
    Returns the computed values corresponding to y^+, U_z^+, P^+, u_z^2, u_r^2, u_t^2, UzUr, p^2.
    """
    # Calculate mean velocities
    mean_vx = torch.mean(velocities[:, 0])
    mean_vy = torch.mean(velocities[:, 1])
    mean_vz = torch.mean(velocities[:, 2])

    # Calculate velocity fluctuations
    fluct_vx = velocities[:, 0] - mean_vx
    fluct_vy = velocities[:, 1] - mean_vy
    fluct_vz = velocities[:, 2] - mean_vz

    # Turbulent kinetic energy (TKE)
    TKE = 0.5 * torch.mean(fluct_vx**2 + fluct_vy**2 + fluct_vz**2)

    # Reynolds stress tau_xz
    reynolds_stress_xz = torch.mean(fluct_vx * fluct_vz)

    # Radial distances (r)
    r = torch.sqrt(positions[:, 0]**2 + positions[:, 1]**2)

    # Mean velocity radial profile
    num_bins = 20
    bins = torch.linspace(0, pipe_radius, num_bins)
    digitized = torch.bucketize(r, bins)  # Use bucketize instead of digitize

    mean_velocity_radial_profile = []
    for i in range(1, num_bins + 1):
        mean_velocity_radial_profile.append(torch.mean(velocities[digitized == i, 2]) if (digitized == i).any() else torch.tensor(0.0))

    mean_velocity_radial_profile = torch.stack(mean_velocity_radial_profile)

    # Assuming these flow parameters are computed from the data
    y_plus = r / pipe_radius
    Uz_plus = mean_velocity_radial_profile  # Just an approximation, refine as needed
    p_plus = reynolds_stress_xz  # Simplified approximation
    uz2 = fluct_vz**2
    ur2 = fluct_vy**2
    ut2 = fluct_vx**2
    UzUr = fluct_vz * fluct_vy
    p2 = fluct_vx**2  # A rough approximation for p^2

    return y_plus, Uz_plus, p_plus, uz2, ur2, ut2, UzUr, p2

def custom_loss_function(pred_positions, pred_velocities, true_flow_data, pipe_radius):
    """
    Custom loss function that compares predicted particle positions/velocities
    to the true flow data (y^+, U_z^+, P^+, etc.).
    """
    # Compute flow parameters based on predicted positions and velocities
    y_plus_pred, Uz_plus_pred, p_plus_pred, uz2_pred, ur2_pred, ut2_pred, UzUr_pred, p2_pred = compute_flow_parameters(pred_positions, pred_velocities, pipe_radius)

    # Ensure true_flow_data is in the expected format
    if isinstance(true_flow_data, dict):
        # Extract true flow parameters from the dictionary
        y_plus_true = torch.tensor(true_flow_data['y^+'].values, dtype=torch.float32)
        Uz_plus_true = torch.tensor(true_flow_data['U_z^+'].values, dtype=torch.float32)
        p_plus_true = torch.tensor(true_flow_data['P^+'].values, dtype=torch.float32)
        uz2_true = torch.tensor(true_flow_data['u_z^2'].values, dtype=torch.float32)
        ur2_true = torch.tensor(true_flow_data['u_r^2'].values, dtype=torch.float32)
        ut2_true = torch.tensor(true_flow_data['u_t^2'].values, dtype=torch.float32)
        UzUr_true = torch.tensor(true_flow_data['UzUr'].values, dtype=torch.float32)
        p2_true = torch.tensor(true_flow_data['p^2'].values, dtype=torch.float32)
        
    elif isinstance(true_flow_data, torch.Tensor):
        # Check if the tensor has the expected shape
        if true_flow_data.dim() == 2:
            # Assuming true_flow_data has the shape [num_samples, num_parameters]
            y_plus_true = true_flow_data[:, 0]
            Uz_plus_true = true_flow_data[:, 1]
            p_plus_true = true_flow_data[:, 2]
            uz2_true = true_flow_data[:, 3]
            ur2_true = true_flow_data[:, 4]
            ut2_true = true_flow_data[:, 5]
            UzUr_true = true_flow_data[:, 6]
            p2_true = true_flow_data[:, 7]
        elif true_flow_data.dim() == 1:
            # If true_flow_data is 1D, you can still convert to tensors without unsqueeze
            y_plus_true = true_flow_data[0]  # This will just be a single value
            Uz_plus_true = true_flow_data[1]
            p_plus_true = true_flow_data[2]
            uz2_true = true_flow_data[3]
            ur2_true = true_flow_data[4]
            ut2_true = true_flow_data[5]
            UzUr_true = true_flow_data[6]
            p2_true = true_flow_data[7]
        else:
            raise ValueError("true_flow_data must be a 1D or 2D tensor.")

    else:
        raise ValueError("true_flow_data must be a dictionary or a tensor.")

    # Mean squared error (MSE) for each flow parameter
    loss_y_plus = torch.mean((y_plus_pred - y_plus_true) ** 2)
    loss_Uz_plus = torch.mean((Uz_plus_pred - Uz_plus_true) ** 2)
    loss_p_plus = torch.mean((p_plus_pred - p_plus_true) ** 2)
    loss_uz2 = torch.mean((uz2_pred - uz2_true) ** 2)
    loss_ur2 = torch.mean((ur2_pred - ur2_true) ** 2)
    loss_ut2 = torch.mean((ut2_pred - ut2_true) ** 2)
    loss_UzUr = torch.mean((UzUr_pred - UzUr_true) ** 2)
    loss_p2 = torch.mean((p2_pred - p2_true) ** 2)

    # Sum up the individual losses to get the total loss
    total_loss = (loss_y_plus + loss_Uz_plus + loss_p_plus + loss_uz2 + 
                  loss_ur2 + loss_ut2 + loss_UzUr + loss_p2)

    return total_loss
