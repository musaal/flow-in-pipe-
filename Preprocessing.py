import numpy as np

# Flow parameters
pipe_radius = 1.0             # Radius of the pipe (R)
bulk_Re = 285000              # Bulk Reynolds number
friction_Re = 6019.48         # Friction Reynolds number (Re_tau)
z_slice = 0.1                 # The z-value (slice) for the gas particles
num_particles = 500000        # Number of particles
velocity_range = 10.0         # Max velocity magnitude (adjust as needed)

# Extended log-law constants (as provided)
k_von_karman = 0.387          # von Kármán constant
alpha = 2.0                   # Adjustable constant for the log-law
beta = 0                      # Adjustable constant for the log-law

# Function to generate positions in the circular cross-section of the pipe (at a single z-slice)
def generate_positions_in_circle(num_particles, pipe_radius, z_slice):
    theta = np.random.uniform(0, 2 * np.pi, num_particles)  # Angular positions
    r = np.sqrt(np.random.uniform(0, pipe_radius**2, num_particles))  # Radial positions
    
    # Convert polar coordinates to Cartesian (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(num_particles, z_slice)  # All particles are at the same z-slice
    
    positions = np.stack((x, y, z), axis=1)
    return positions

# Function to compute streamwise velocity profile using the extended log-law
def extended_log_velocity_profile(r, Re_tau, pipe_radius, alpha, beta):
    # Radial distance normalized by pipe radius (0 at center, 1 at wall)
    y_plus = Re_tau * (1 - r / pipe_radius)
    
    # Extended log-law for turbulent boundary layer (for r close to the wall)
    u_z = alpha * np.log(y_plus) + beta
    u_z = np.maximum(u_z, 0)  # Ensure no negative velocities
    return u_z

# Function to generate velocities with streamwise profile based on radius
def generate_velocities(num_particles, Re_tau, pipe_radius, alpha, beta):
    # Generate radial positions to calculate streamwise velocity
    theta = np.random.uniform(0, 2 * np.pi, num_particles)  # Angular positions
    r = np.sqrt(np.random.uniform(0, pipe_radius**2, num_particles))  # Radial positions
    
    # Streamwise velocity follows the extended log-law
    v_z = extended_log_velocity_profile(r, Re_tau, pipe_radius, alpha, beta)
    
    # Small turbulent fluctuations around the mean in x and y directions
    v_x = np.random.normal(0, 0.1 * velocity_range, num_particles)
    v_y = np.random.normal(0, 0.1 * velocity_range, num_particles)
    
    # Stack the velocity components together
    velocities = np.stack((v_x, v_y, v_z), axis=1)
    return velocities

# Generate positions and velocities for the particles in the current slice
positions = generate_positions_in_circle(num_particles, pipe_radius, z_slice)
velocities = generate_velocities(num_particles, friction_Re, pipe_radius, alpha, beta)

# Combine positions and velocities into a dataset (current slice)
dataset_current = np.hstack((positions, velocities))

# Optionally, save the dataset to a CSV file
np.savetxt("pipe_flow_slice_0.1.csv", dataset_current, delimiter=",", header="x,y,z,vx,vy,vz", comments='')

print(f"Dataset of {num_particles} particles in the slice at z={z_slice} saved to 'pipe_flow_slice_test.csv'.")


#  need to drop the z 0.1 and make a prodiction an the hole data a



def custom_loss_function(predictions, targets, velocities, positions):
    # Basic MSE Loss
    criterion = nn.MSELoss()
    loss = criterion(predictions, targets)

    # Calculate additional losses based on physical properties
    fluct_vx = velocities[:, 0] - np.mean(velocities[:, 0])
    fluct_vz = velocities[:, 2] - np.mean(velocities[:, 2])

    # Reynolds stress component (tau_xz)
    reynolds_stress_xz = np.mean(fluct_vx * fluct_vz)
    # target_reynolds_stress = calculate_target_reynolds_stress(targets) # You'll need to define this

    # Penalize differences in Reynolds stress
    reynolds_stress_loss = (reynolds_stress_xz )**2 # (reynolds_stress_xz - target_reynolds_stress)**2

    # Add the Reynolds stress loss to the total loss
    total_loss = loss + reynolds_stress_loss

    return total_loss
