import numpy
import random
import csv
import time
from scipy.optimize import root_scalar

# =========================== Physical Constants =========================== # useful constants to use
m_p_rest = 0.93827208943  # rest mass of proton (GeV/c^2)
m_n_rest = 0.93857  # rest mass of neutron (GeV/c^2)
m_pi_rest = 0.13957  # rest mass of charged pions (GeV/c^2)
m_pi_o_rest = 0.135  # rest mass of neutral pions (GeV/c^2)
m_k_rest = 0.49638  # rest mass of charged kaons (GeV/c^2)
m_mu_rest = 0.105658  # rest mass of muons (GeV/c^2)
m_e_rest = 0.000511  # rest mass of electrons / positrons (GeV/c^2)
N_A = 6.022 * (10 ** 23)  # Avogadro's number (particles/mol)
r_o = 1.2  # nuclear radius constant (fm)
dEdX_MIP = 0.002  # Ionization energy loss rate for Minimum Ionizing Particles (GeV per g/cm^2)

# =========================== Atmospheric Parameters ===========================
density_at_surface = 1.225e-3  # density of air at sea level (g/cm^3)
scale_height = 6.5e5  # scale height of Earth's atmosphere (cm)
total_altitude = 100  # Assumed starting altitude of the cosmic ray (km)
M_avg = 14.5  # average molar mass of air (g/mol)
X_o = 37.1  # radiation length of air for EM interactions (g/cm^2)
ground_depth = density_at_surface * scale_height  # Total atmospheric depth at sea level (g/cm^2)

# =========================== Hadronic Interaction Parameters ===========================
sigma_146 = [250, 230, 300, 290]  # Cross-sections for N2 collisions [pi, k, n, p] (mb)
sigma_166 = [290, 260, 340, 320]  # Cross-sections for O2 collision [pi, k, n, p] (mb)
alpha_o_for_fraction_particles_perturbations = 40  # Controls randomness in secondary particle fractions
# Average fractions of secondary particles produced in collisions [pi+, pi-, pi0, k+, k-, p, n]
fract_particles_array_proton_collision = numpy.array([0.24, 0.22, 0.30, 0.04, 0.03, 0.06, 0.06])
fract_particles_array_neutron_collision = numpy.array([0.22, 0.24, 0.30, 0.04, 0.03, 0.05, 0.07])
fract_particles_array_pion_plus_collision = numpy.array([0.32, 0.28, 0.25, 0.04, 0.03, 0.03, 0.03])
fract_particles_array_pion_minus_collision = numpy.array([0.28, 0.32, 0.25, 0.04, 0.03, 0.03, 0.03])
fract_particles_array_kaon_plus_collision = numpy.array([0.24, 0.22, 0.24, 0.10, 0.06, 0.03, 0.03])
fract_particles_array_kaon_minus_collision = numpy.array([0.22, 0.24, 0.24, 0.06, 0.10, 0.03, 0.03])
rest_mass_array = numpy.array([m_pi_rest, m_pi_rest, m_pi_o_rest, m_k_rest, m_k_rest, m_p_rest, m_n_rest])

# =========================== Simulation Variables & Placeholders ===========================
energy_threshold = 0.001  # Kinetic energy threshold for stopping particles (GeV)
X_max = 0  # Tracks the maximum depth reached in a shower
minimum_altitude = total_altitude  # Tracks the lowest altitude reached
number_of_secondary_particles = numpy.zeros(7, dtype=float)  # Reusable array for particle counts
alpha_array_particle = numpy.zeros(7)  # Reusable array for Dirichlet distribution alphas


# =========================== Helper Functions ===========================
def particle_interacts_with_nitrogen_or_oxygen():
    # Determines if interaction is with N2 (78.8%) or O2 (21.2%) based on air composition.
    return 1 if numpy.random.uniform(0, 1) < 0.788 else 0


def total_number_of_particles_scaled_to_abide_energy_conservation(E_secondaries_func, rest_mass_array_func,number_of_secondary_particles_integer_func):
    # Ensures the total rest mass of produced particles does not exceed the available energy.
    total_rest_mass_func = numpy.sum(rest_mass_array_func * number_of_secondary_particles_integer_func)
    if total_rest_mass_func > E_secondaries_func:
        # If rest mass is too high, scale down the number of particles proportionally.
        scale_factor_func = E_secondaries_func / total_rest_mass_func if total_rest_mass_func > 1e-9 else 0
        number_of_secondary_particles_integer_func = numpy.round(number_of_secondary_particles_integer_func * scale_factor_func).astype(int)
        total_rest_mass_func = numpy.sum(rest_mass_array_func * number_of_secondary_particles_integer_func)
    if total_rest_mass_func > E_secondaries_func:  # Final check after rounding
        number_of_secondary_particles_integer_func = numpy.zeros(7, dtype=int)
        total_rest_mass_func = 0
    n_total_func = numpy.sum(number_of_secondary_particles_integer_func)
    return n_total_func, total_rest_mass_func, number_of_secondary_particles_integer_func


def enforce_conservation_laws(number_of_secondary_particles_integer_func, Q_initial, B_initial, S_initial):
    # Iteratively "trades" particles to enforce conservation of Charge (Q), Baryon Number (B), and Strangeness (S).
    n_sec = number_of_secondary_particles_integer_func.copy()
    max_loops = 2000  # Number of attempts to fix conservation before failing.
    for _ in range(max_loops):
        # Calculate current quantum numbers from the particle list.
        Q_final = n_sec[0] + n_sec[3] + n_sec[5] - n_sec[1] - n_sec[4]
        B_final = n_sec[5] + n_sec[6]
        S_final = n_sec[3] - n_sec[4]
        # Calculate the differences we need to fix.
        Q_diff, B_diff, S_diff = Q_final - Q_initial, B_final - B_initial, S_final - S_initial
        if Q_diff == 0 and B_diff == 0 and S_diff == 0: return n_sec  # Success! All laws conserved.

        # Prioritized list of trades. Start with the most constrained quantum number (Strangeness).
        if S_diff > 0:  # Too positive, need to reduce S. Trade K+ for pi+.
            if n_sec[3] > 0: n_sec[3] -= 1; n_sec[0] += 1; continue
        elif S_diff < 0:  # Too negative, need to increase S. Trade K- for pi-.
            if n_sec[4] > 0: n_sec[4] -= 1; n_sec[1] += 1; continue
        # Then fix Baryon Number.
        if B_diff > 0:  # Too many baryons. Trade n for pi0, or p for pi+.
            if n_sec[6] > 0:
                n_sec[6] -= 1; n_sec[2] += 1; continue
            elif n_sec[5] > 0:
                n_sec[5] -= 1; n_sec[0] += 1; continue
        elif B_diff < 0:  # Too few baryons. Trade pi0 for n, or pi+ for p.
            if n_sec[2] > 0:
                n_sec[2] -= 1; n_sec[6] += 1; continue
            elif n_sec[0] > 0:
                n_sec[0] -= 1; n_sec[5] += 1; continue
        # Finally, fix Charge.
        if Q_diff > 0:  # Too positive. Trade pi+ for pi0, or p for n.
            if n_sec[0] > 0:
                n_sec[0] -= 1; n_sec[2] += 1; continue
            elif n_sec[5] > 0:
                n_sec[5] -= 1; n_sec[6] += 1; continue
        elif Q_diff < 0:  # Too negative. Trade pi- for pi0, or n for p.
            if n_sec[1] > 0:
                n_sec[1] -= 1; n_sec[2] += 1; continue
            elif n_sec[6] > 0:
                n_sec[6] -= 1; n_sec[5] += 1; continue
    return None  # Return None if the loop finishes without finding a solution.


def KE_distribution_among_n_total_particles(E_secondaries_func, total_rest_mass_func, n_total_func,number_of_secondary_particles_integer_func):
    # Randomly distributes the available kinetic energy among all secondary particles.
    total_KE_of_particles = E_secondaries_func - total_rest_mass_func
    if total_KE_of_particles < 0:
        total_KE_of_particles = 0  # Ensure non-negative KE.
    if n_total_func <= 0:
        return (numpy.array([]) for _ in range(7))  # Return empty arrays if no particles.
    alpha_array_KE = numpy.ones(int(n_total_func))  # Use alpha=1 for uniform random splitting.
    # Dirichlet distribution ensures all fractions sum to 1, conserving energy.
    KE_distribution_secondary = numpy.random.dirichlet(alpha_array_KE) * total_KE_of_particles
    # Slice the single energy array into separate arrays for each particle type.
    indices = numpy.cumsum(number_of_secondary_particles_integer_func).astype(int)
    KE_arrays = numpy.split(KE_distribution_secondary, indices[:-1])
    return tuple(KE_arrays)


def E_total_of_secondary_particles(KE_pi_plus_func, KE_pi_minus_func, KE_pi_o_func, KE_k_plus_func, KE_k_minus_func, KE_p_func, KE_n_func):
    # Calculates the total energy (KE + rest mass) for each secondary particle.
    return KE_pi_plus_func + m_pi_rest, KE_pi_minus_func + m_pi_rest, KE_pi_o_func + m_pi_o_rest, KE_k_plus_func + m_k_rest, KE_k_minus_func + m_k_rest, KE_p_func + m_p_rest, KE_n_func + m_n_rest


def photons_energy_split_when_pi_o_decays(E_lab_pi_o_func, E_pi_o_rest_func):
    # Simulates the 2-body decay of a pi0 into two photons, conserving energy and momentum.
    gamma = E_lab_pi_o_func / E_pi_o_rest_func if E_pi_o_rest_func > 0 else 1.0  # Lorentz factor.
    beta_gamma = numpy.sqrt(1 - 1 / gamma ** 2) if gamma > 1 else 0
    cos_theta = numpy.random.uniform(-1, 1)  # Random angle of decay in the pi0's rest frame.
    # Doppler shift formulas to get photon energies in the lab frame.
    E_forward = gamma * (E_pi_o_rest_func / 2) * (1 + beta_gamma * cos_theta)
    E_backward = gamma * (E_pi_o_rest_func / 2) * (1 - beta_gamma * cos_theta)
    return E_forward, E_backward


def check_pair_production_can_even_take_place(E_photon_func):
    # Checks if a photon has enough energy to create an electron-positron pair.
    return 1 if E_photon_func > 2 * m_e_rest else 0


def distance_from_top_of_atmosphere(X_depth_func, lambda_factor, X_o_or_lambda_int):
    # Calculates a random interaction/decay depth based on an exponential probability distribution.
    effective_lambda = lambda_factor * X_o_or_lambda_int  # The mean free path for the process.
    # Sample from P(x) = exp(-x/lambda) to get the distance traveled (delta_X).
    delta_X = -effective_lambda * numpy.log(random.uniform(1e-6, 1.0 - 1e-6)) if effective_lambda > 0 else float('inf')
    X_event = X_depth_func + delta_X  # The new atmospheric depth of the event.
    # Convert the new depth back to a geometric altitude (km).
    log_arg = X_event / (scale_height * density_at_surface) if (scale_height * density_at_surface) > 0 else 0
    altitude_in_cm = total_altitude * 1e5  # Default to top altitude.
    if X_event > 0 and log_arg > 0:
        try:
            altitude_in_cm = -scale_height * numpy.log(log_arg)
        except FloatingPointError:  # Catch errors if X_event is extremely large.
            altitude_in_cm = 0
    return X_event, altitude_in_cm / 1e5, delta_X


def Energy_distribution_in_e_plus_e_minus_pair_from_photon_it_decayed(E_photon_func):
    # Simulates how the photon's energy is split between the new electron and positron.
    y_frac = 0.5;  # Default to a 50/50 split as a failsafe.
    r = random.uniform(0, 1)
    coffs_of_y_cubic = [4, -6, 9, -7 * r]  # Coefficients for the energy sharing distribution.
    try:  # Solve the cubic equation to find a more realistic energy split.
        roots = numpy.roots(coffs_of_y_cubic)
        real_roots = [r.real for r in roots if numpy.isreal(r) and 0 < r.real < 1]
        if real_roots: y_frac = float(real_roots[0])
    except Exception:  # If the solver fails, we just use the 50/50 split.
        pass
    return (1.0 - y_frac) * E_photon_func, y_frac * E_photon_func


def Will_bremsstrahlung_happen_or_not(E_electron_or_positron_func):
    # Checks if an electron/positron's kinetic energy is above the critical energy for bremsstrahlung (~81 MeV).
    return 1 if E_electron_or_positron_func > (0.081 + m_e_rest) else 0


def Energy_distribution_electron_and_photon_or_positron_and_photon(KE_electron_or_positron):
    # Simulates how a charged particle's kinetic energy is split between itself and a new photon during bremsstrahlung.
    y = 0.5  # Default to a 50/50 split as a failsafe.
    r = random.uniform(1e-6, 1.0 - 1e-6)

    # Defines the equation for the energy sharing distribution.
    def equation(y_var):
        return 2 * numpy.log(max(y_var, 1e-9)) - 2 * max(y_var, 1e-9) + (max(y_var, 1e-9) ** 2) / 2 - (
                26.131 * r - 27.631)

    try:  # Use a numerical solver to find a realistic energy split.
        sol = root_scalar(equation, bracket=(1e-6, 1.0 - 1e-6), method='brentq')
        if sol.converged: y = sol.root
    except Exception:  # If solver fails, use the 50/50 split.
        pass
    return (1 - y) * KE_electron_or_positron, y * KE_electron_or_positron


# =========================== Main Execution Block ===========================

if __name__ == "__main__":
    try:  # Get user input for simulation parameters.
        E_cosmic_ray = float(input("Enter the cosmic ray energy for all showers (in GeV): "))
        num_showers = int(input("Enter the number of showers to simulate: "))
    except ValueError:
        print("Invalid input. Please enter numbers.")
        exit()

    # Create a unique filename for the output data.
    output_filename = f"shower_data_{int(E_cosmic_ray)}GeV_{num_showers}runs.csv"
    print(f"\nStarting simulation of {num_showers} showers at {E_cosmic_ray} GeV.")
    print(f"Aggregated data will be saved to '{output_filename}'")
    start_time = time.time()  # Start the timer.

    all_final_particles = []  # This will collect all particles from all showers.

    # Loop to run the simulation 'num_showers' times.
    for shower_id in range(1, num_showers + 1):
        print(f"\n--- Running Shower {shower_id} of {num_showers} ---")

        # Initialize the particle lists and tracking variables for each new shower.
        hydronic_particles = [{"type": "proton", "total_energy": numpy.float64(E_cosmic_ray), "atmospheric_depth": numpy.float64(0.0)}]
        non_interacting_particles = []
        X_max = 0
        minimum_altitude = total_altitude

        # --- Main Simulation Loop (for one shower) ---
        while len(hydronic_particles) > 0:  # Loop until no more active particles are left.
            particle_in_observation = hydronic_particles.pop()  # Take the last particle from the list (depth-first).
            ptype = particle_in_observation["type"]
            current_energy = particle_in_observation["total_energy"]
            current_depth = particle_in_observation["atmospheric_depth"]

            # --- HADRONIC INTERACTION BLOCK ---
            if ptype in ["proton", "neutron", "kaon+", "kaon-"]:
                # Set properties based on particle type.
                if ptype == "proton":
                    rest_mass, sigma_idx, thr, frac_arr = m_p_rest, 3, 0.3, fract_particles_array_proton_collision
                elif ptype == "neutron":
                    rest_mass, sigma_idx, thr, frac_arr = m_n_rest, 2, 0.3, fract_particles_array_neutron_collision
                elif ptype == "kaon+":
                    rest_mass, sigma_idx, thr, frac_arr = m_k_rest, 1, 0.5, fract_particles_array_kaon_plus_collision
                else:  # kaon-
                    rest_mass, sigma_idx, thr, frac_arr = m_k_rest, 1, 0.5, fract_particles_array_kaon_minus_collision

                is_charged = (ptype != "neutron")

                # Calculate the random depth of the next interaction.
                effective_sigma = (0.78 * sigma_146[sigma_idx] + 0.21 * sigma_166[sigma_idx]) * 1e-27
                lambda_int = M_avg / (effective_sigma * N_A) if effective_sigma > 0 else float('inf')
                X_interaction, _, delta_X = distance_from_top_of_atmosphere(current_depth, 1.0, lambda_int)

                # --- Ground boundary check for ALL hadrons ---
                if X_interaction >= ground_depth:
                    if is_charged:  # If charged, lose energy on the way to the ground.
                        delta_X_to_ground = max(0, ground_depth - current_depth)
                        energy_at_ground = current_energy - (dEdX_MIP * delta_X_to_ground)
                        particle_in_observation["total_energy"] = numpy.float64(max(energy_at_ground, rest_mass))
                        particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                        non_interacting_particles.append(particle_in_observation)
                    else:  # Uncharged particle (neutron) hits ground with unchanged energy.
                        particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                        non_interacting_particles.append(particle_in_observation)
                    continue  # Stop processing this particle.

                # If interaction is in the atmosphere, apply ionization loss.
                energy_after_loss = current_energy
                if is_charged:
                    energy_loss = dEdX_MIP * delta_X
                    energy_after_loss = current_energy - energy_loss
                    # Check if particle stopped from ionization before reaching the interaction point.
                    if (energy_after_loss - rest_mass) <= energy_threshold or energy_after_loss < rest_mass:
                        particle_in_observation["total_energy"] = numpy.float64(max(energy_after_loss, rest_mass))
                        particle_in_observation["atmospheric_depth"] = numpy.float64(X_interaction)
                        non_interacting_particles.append(particle_in_observation)
                        continue

                # --- Handle hadrons that are now below their interaction threshold ---
                if energy_after_loss <= thr:
                    if is_charged:
                        # Particle is too low-energy to interact, but will "coast" until it stops.
                        kinetic_energy_remaining = energy_after_loss - rest_mass
                        if kinetic_energy_remaining > energy_threshold:
                            delta_X_to_stop = kinetic_energy_remaining / dEdX_MIP if dEdX_MIP > 0 else float('inf')
                            X_stop = X_interaction + delta_X_to_stop  # It stops *after* the point of the would-be interaction.

                            if X_stop < ground_depth:  # Stops in the atmosphere.
                                particle_in_observation["total_energy"] = numpy.float64(rest_mass)
                                particle_in_observation["atmospheric_depth"] = numpy.float64(X_stop)
                                non_interacting_particles.append(particle_in_observation)
                            else:  # Hits the ground before stopping completely.
                                delta_X_to_ground = max(0, ground_depth - X_interaction)
                                energy_at_ground = energy_after_loss - (dEdX_MIP * delta_X_to_ground)
                                particle_in_observation["total_energy"] = numpy.float64(max(energy_at_ground, rest_mass))
                                particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                                non_interacting_particles.append(particle_in_observation)
                        else:  # Already effectively stopped.
                            particle_in_observation["total_energy"] = numpy.float64(rest_mass)
                            particle_in_observation["atmospheric_depth"] = numpy.float64(X_interaction)
                            non_interacting_particles.append(particle_in_observation)
                    else:  # Uncharged particle (neutron) below threshold just stops being tracked.
                        particle_in_observation["atmospheric_depth"] = numpy.float64(X_interaction)
                        non_interacting_particles.append(particle_in_observation)
                    continue  # Stop processing this particle.

                # --- If above threshold, proceed with the full interaction ---
                particle_in_observation["total_energy"] = energy_after_loss
                particle_in_observation["atmospheric_depth"] = X_interaction
                X = X_interaction

                # ... [Rest of your detailed hadronic interaction logic, which is well-established] ...
                # ... This part calculates multiplicity, secondary energy, etc. ...
                nuclei_value = particle_interacts_with_nitrogen_or_oxygen()
                A = 14 if nuclei_value == 1 else 16
                sigma_effective = sigma_146[sigma_idx] if nuclei_value == 1 else sigma_166[sigma_idx]
                R_A = r_o * A ** (1 / 3)
                V = sigma_effective * R_A / (1000 * numpy.pi * r_o ** 3)
                V = max(1, int(round(V)))
                if ptype in ["proton", "neutron"]:
                    s = m_p_rest ** 2 + rest_mass ** 2 + 2 * m_p_rest * energy_after_loss
                    n_ch_avg = 3.6 + 0.45 * numpy.log(s) + 0.12 * numpy.log(s) ** 2
                    k_particle = 1.5 + 0.15 * numpy.log(numpy.sqrt(s)); fract_E = 0.30 + 0.15 * numpy.tanh(numpy.log10(energy_after_loss / 10))
                else:
                    s = m_k_rest ** 2 + m_p_rest ** 2 + 2 * m_p_rest * energy_after_loss
                    n_ch_avg = 2.1 + 0.23 * numpy.log(s) + 0.11 * numpy.log(s) ** 2
                    k_particle = 1.2 + 0.05 * numpy.log(s)
                    fract_E = 0.40 + 0.15 * numpy.tanh(numpy.log10(energy_after_loss / 10))
                n_ch_avg_collision_atom = V * n_ch_avg
                k_particle_atom = V * k_particle
                p_neg_bin = k_particle_atom / (k_particle_atom + n_ch_avg_collision_atom) if (k_particle_atom + n_ch_avg_collision_atom) > 0 else 0
                charged_secondaries_formed = numpy.random.negative_binomial(k_particle_atom,p_neg_bin) if k_particle_atom > 0 and p_neg_bin > 0 else 0
                number_of_secondaries_formed = round(charged_secondaries_formed / 0.66) if charged_secondaries_formed > 0 else 0
                E_secondaries = fract_E * energy_after_loss
                max_rerolls = 2000
                reroll_count = 0
                success = False
                while reroll_count < max_rerolls:
                    reroll_count += 1
                    alpha_array_particle = frac_arr * alpha_o_for_fraction_particles_perturbations
                    fraction_array_secondary_particles = numpy.random.dirichlet(alpha_array_particle)
                    number_of_secondary_particles_attempt = numpy.round(fraction_array_secondary_particles * number_of_secondaries_formed).astype(int)
                    diff = number_of_secondaries_formed - numpy.sum(number_of_secondary_particles_attempt)
                    if diff != 0 and number_of_secondaries_formed > 0: max_idx = numpy.argmax(
                        fraction_array_secondary_particles); number_of_secondary_particles_attempt[max_idx] = max(0,number_of_secondary_particles_attempt[max_idx] + int(diff))
                    Q_init, B_init, S_init = 0, 0, 0
                    if ptype == "proton":
                        Q_init = 1 + int(V // 2);B_init = 1 + V;S_init = 0
                    elif ptype == "neutron":
                        Q_init = 0 + int(V // 2);B_init = 1 + V;S_init = 0
                    elif ptype == "kaon+":
                        Q_init = 1 + int(V // 2);B_init = 0 + V;S_init = 1
                    elif ptype == "kaon-":
                        Q_init = -1 + int(V // 2);B_init = 0 + V;S_init = -1
                    conserved_particles = enforce_conservation_laws(number_of_secondary_particles_attempt, Q_init,B_init, S_init)
                    if conserved_particles is not None: number_of_secondary_particles_integer = conserved_particles; success = True; break
                if not success:
                    print(f"  --> Re-roll limit ({max_rerolls}) reached for {ptype}. Skipping secondaries.");
                    continue
                n_total, total_mass, num_sec_int = total_number_of_particles_scaled_to_abide_energy_conservation(E_secondaries, rest_mass_array, number_of_secondary_particles_integer)
                KE_arrays = KE_distribution_among_n_total_particles(E_secondaries, total_mass, n_total, num_sec_int);
                E_total_arrays = E_total_of_secondary_particles(*KE_arrays)
                types = ["pion+", "pion-", "pion_o", "kaon+", "kaon-", "proton", "neutron"]
                for i, e_array in enumerate(E_total_arrays):
                    for energy_val in e_array: hydronic_particles.append({"type": types[i], "total_energy": numpy.float64(energy_val),"atmospheric_depth": numpy.float64(X)})

            elif ptype in ["pion-", "pion+"]:
                # This block handles the competition between pion decay and interaction.
                # (Comments are omitted for brevity as the logic is similar to hadrons, but with decay included)
                rest_mass = m_pi_rest
                is_plus = (ptype == "pion+")
                frac_arr = fract_particles_array_pion_plus_collision if is_plus else fract_particles_array_pion_minus_collision
                effective_sigma = (0.78 * sigma_146[0] + 0.21 * sigma_166[0]) * 1e-27
                lambda_int = M_avg / (effective_sigma * N_A) if effective_sigma > 0 else float('inf')
                X_interaction, _, delta_X_int = distance_from_top_of_atmosphere(current_depth, 1.0, lambda_int)
                Gamma = current_energy / rest_mass
                Beta = numpy.sqrt(1 - 1 / Gamma ** 2) if Gamma > 1 else 0;
                T_mean_lab = 2.603e-8 * Gamma
                L = Beta * 2.99792458e8 * T_mean_lab
                dist_decay_meters = -L * numpy.log(random.uniform(1e-6, 1 - 1e-6))
                parent_alt_cm = total_altitude * 1e5
                log_arg_parent = current_depth / (scale_height * density_at_surface) if (scale_height * density_at_surface) > 0 else 0
                if current_depth > 0 and log_arg_parent > 0:
                    try:
                        parent_alt_cm = -scale_height * numpy.log(log_arg_parent)
                    except FloatingPointError:
                        parent_alt_cm = 0
                decay_alt_cm = parent_alt_cm - (dist_decay_meters * 100);
                log_arg_decay = numpy.exp(-decay_alt_cm / scale_height) if scale_height > 0 else 0
                if decay_alt_cm < 0 or log_arg_decay <= 0:
                    X_decay = ground_depth
                else:
                    X_decay = density_at_surface * scale_height * log_arg_decay
                delta_X_decay = max(0, X_decay - current_depth)
                if min(X_decay, X_interaction) >= ground_depth:
                    delta_X_to_ground = max(0, ground_depth - current_depth)
                    energy_at_ground = current_energy - (dEdX_MIP * delta_X_to_ground)
                    particle_in_observation["total_energy"] = numpy.float64(max(energy_at_ground, rest_mass))
                    particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                    non_interacting_particles.append(particle_in_observation)
                    continue
                if X_decay < X_interaction:
                    delta_X = delta_X_decay
                    energy_loss = dEdX_MIP * delta_X
                    energy_at_event = current_energy - energy_loss
                    X = X_decay
                    if (energy_at_event - rest_mass) <= energy_threshold or energy_at_event < rest_mass:
                        particle_in_observation["total_energy"] = numpy.float64(max(energy_at_event, rest_mass))
                        particle_in_observation["atmospheric_depth"] = numpy.float64(X)
                        non_interacting_particles.append(particle_in_observation)
                        continue
                    KE_dist = energy_at_event - m_mu_rest
                    KE_mu = random.uniform(1e-6, 1 - 1e-6) * KE_dist if KE_dist > 0 else 0
                    E_mu = KE_mu + m_mu_rest
                    E_nu = KE_dist - KE_mu if KE_dist > 0 else 0
                    muon_type = "muon_plus" if is_plus else "muon_minus"
                    non_interacting_particles.append({"type": "muon_nutrino", "total_energy": numpy.float64(E_nu),"atmospheric_depth": numpy.float64(X)})
                    hydronic_particles.append({"type": muon_type, "total_energy": numpy.float64(E_mu), "atmospheric_depth": numpy.float64(X)})
                else:
                    delta_X = delta_X_int
                    energy_loss = dEdX_MIP * delta_X
                    energy_at_event = current_energy - energy_loss
                    X = X_interaction
                    if (energy_at_event - rest_mass) <= energy_threshold or energy_at_event < rest_mass:
                        particle_in_observation["total_energy"] = numpy.float64(max(energy_at_event, rest_mass))
                        particle_in_observation["atmospheric_depth"] = numpy.float64(X)
                        non_interacting_particles.append(particle_in_observation)
                        continue
                    if energy_at_event <= 0.2:
                        kinetic_energy_remaining = energy_at_event - rest_mass
                        if kinetic_energy_remaining > energy_threshold:
                            delta_X_to_stop = kinetic_energy_remaining / dEdX_MIP if dEdX_MIP > 0 else float('inf')
                            X_stop = X + delta_X_to_stop
                            if X_stop < ground_depth:
                                particle_in_observation["total_energy"] = numpy.float64(rest_mass)
                                particle_in_observation["atmospheric_depth"] = numpy.float64(X_stop)
                                non_interacting_particles.append(particle_in_observation)
                            else:
                                delta_X_to_ground = max(0, ground_depth - X)
                                energy_at_ground = energy_at_event - (dEdX_MIP * delta_X_to_ground)
                                particle_in_observation["total_energy"] = numpy.float64(max(energy_at_ground, rest_mass))
                                particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                                non_interacting_particles.append(particle_in_observation)
                        else:
                            particle_in_observation["total_energy"] = numpy.float64(rest_mass)
                            particle_in_observation["atmospheric_depth"] = numpy.float64(X)
                            non_interacting_particles.append(particle_in_observation)
                        continue
                    particle_in_observation["total_energy"] = energy_at_event
                    particle_in_observation["atmospheric_depth"] = X
                    nuclei_value = particle_interacts_with_nitrogen_or_oxygen()
                    A = 14 if nuclei_value == 1 else 16
                    sigma_effective = sigma_146[0] if nuclei_value == 1 else sigma_166[0]
                    R_A = r_o * A ** (1 / 3)
                    V = sigma_effective * R_A / (1000 * numpy.pi * r_o ** 3)
                    V = max(1, int(round(V)))
                    s = m_pi_rest ** 2 + m_p_rest ** 2 + 2 * m_p_rest * energy_at_event
                    n_ch_avg = 2.1 + 0.23 * numpy.log(s) + 0.11 * numpy.log(s) ** 2
                    k_particle_pi = 1.2 + 0.05 * numpy.log(s)
                    n_ch_avg_collision_atom = V * n_ch_avg
                    k_particle_atom = V * k_particle_pi
                    p_neg_bin = k_particle_atom / (k_particle_atom + n_ch_avg_collision_atom) if (k_particle_atom + n_ch_avg_collision_atom) > 0 else 0
                    charged_secondaries_formed = numpy.random.negative_binomial(k_particle_atom,p_neg_bin) if k_particle_atom > 0 and p_neg_bin > 0 else 0
                    number_of_secondaries_formed = round(charged_secondaries_formed / 0.66) if charged_secondaries_formed > 0 else 0
                    fract_E = 0.40 + 0.15 * numpy.tanh(numpy.log10(energy_at_event / 10))
                    E_secondaries = fract_E * energy_at_event
                    max_rerolls = 2000
                    reroll_count = 0
                    success = False
                    while reroll_count < max_rerolls:
                        reroll_count += 1
                        for i in range(len(frac_arr)): alpha_array_particle[i] = frac_arr[i] * alpha_o_for_fraction_particles_perturbations
                        fraction_array_secondary_particles = numpy.random.dirichlet(alpha_array_particle)
                        number_of_secondary_particles_attempt = numpy.round(fraction_array_secondary_particles * number_of_secondaries_formed).astype(int)
                        diff = number_of_secondaries_formed - numpy.sum(number_of_secondary_particles_attempt)
                        if diff != 0 and number_of_secondaries_formed > 0: max_idx = numpy.argmax(fraction_array_secondary_particles); number_of_secondary_particles_attempt[max_idx] = max(0,number_of_secondary_particles_attempt[max_idx] + int(diff))
                        Q_init = (1 if is_plus else -1) + int(V // 2)
                        B_init = 0 + V
                        S_init = 0
                        conserved_particles = enforce_conservation_laws(number_of_secondary_particles_attempt, Q_init, B_init, S_init)
                        if conserved_particles is not None: number_of_secondary_particles_integer = conserved_particles; success = True; break
                    if not success:
                        print(f"  --> Re-roll limit ({max_rerolls}) reached for {ptype}. Skipping secondaries.")
                        continue
                    n_total, total_mass, num_sec_int = total_number_of_particles_scaled_to_abide_energy_conservation(E_secondaries, rest_mass_array, number_of_secondary_particles_integer)
                    KE_arrays = KE_distribution_among_n_total_particles(E_secondaries, total_mass, n_total,num_sec_int)
                    E_total_arrays = E_total_of_secondary_particles(*KE_arrays)
                    types = ["pion+", "pion-", "pion_o", "kaon+", "kaon-", "proton", "neutron"]
                    for i, e_array in enumerate(E_total_arrays):
                        for energy_val in e_array: hydronic_particles.append({"type": types[i], "total_energy": numpy.float64(energy_val),"atmospheric_depth": numpy.float64(X)})

            # --- MUON PROPAGATION BLOCK ---
            elif ptype in ["muon_plus", "muon_minus"]:
                rest_mass = m_mu_rest
                kinetic_energy_initial = current_energy - rest_mass
                if kinetic_energy_initial > energy_threshold:
                    # Calculate depth needed to stop via ionization.
                    delta_X_to_stop = kinetic_energy_initial / dEdX_MIP if dEdX_MIP > 0 else float('inf')
                    X_stop = current_depth + delta_X_to_stop

                    if X_stop < ground_depth:  # Stops in the atmosphere.
                        particle_in_observation["total_energy"] = numpy.float64(rest_mass)
                        particle_in_observation["atmospheric_depth"] = numpy.float64(X_stop)
                        non_interacting_particles.append(particle_in_observation)
                    else:  # Hits the ground before stopping.
                        delta_X_to_ground = max(0, ground_depth - current_depth)
                        energy_at_ground = current_energy - (dEdX_MIP * delta_X_to_ground)
                        particle_in_observation["total_energy"] = numpy.float64(max(energy_at_ground, rest_mass))
                        particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                        non_interacting_particles.append(particle_in_observation)
                else:  # Already stopped.
                    non_interacting_particles.append(particle_in_observation)

            # --- NEUTRAL PION DECAY BLOCK ---
            elif ptype == "pion_o":
                # Decays instantly into two photons. No propagation needed.
                if current_energy > m_pi_o_rest:
                    forward_E, backward_E = photons_energy_split_when_pi_o_decays(current_energy, m_pi_o_rest)
                    hydronic_particles.append({"type": "photon", "total_energy": numpy.float64(forward_E),"atmospheric_depth": numpy.float64(current_depth)})
                    hydronic_particles.append({"type": "photon", "total_energy": numpy.float64(backward_E),"atmospheric_depth": numpy.float64(current_depth)})

            # --- PHOTON INTERACTION BLOCK ---
            elif ptype == "photon":
                # Photons are uncharged, so no ionization loss. They either pair-produce or stop.
                W_pair = check_pair_production_can_even_take_place(current_energy)
                if W_pair == 1:
                    X_event, alt_km, _ = distance_from_top_of_atmosphere(current_depth, 9.0 / 7.0, X_o)

                    if X_event >= ground_depth: continue  # Interaction would be underground, so particle effectively lost.

                    # If interaction is in atmosphere, create new particles.
                    E_e_plus, E_e_minus = Energy_distribution_in_e_plus_e_minus_pair_from_photon_it_decayed(
                        current_energy)
                    hydronic_particles.append({"type": "positron", "total_energy": numpy.float64(E_e_plus),"atmospheric_depth": numpy.float64(X_event)})
                    hydronic_particles.append({"type": "electron", "total_energy": numpy.float64(E_e_minus),"atmospheric_depth": numpy.float64(X_event)})
                else:  # Below pair production threshold.
                    non_interacting_particles.append(particle_in_observation)

            # --- ELECTRON/POSITRON INTERACTION BLOCK ---
            elif ptype in ["electron", "positron"]:
                rest_mass = m_e_rest
                W_brem = Will_bremsstrahlung_happen_or_not(current_energy)

                if W_brem == 1:  # High energy: competes between Brem and stopping.
                    X_event, alt_km, delta_X = distance_from_top_of_atmosphere(current_depth, 1.0, X_o)

                    if X_event >= ground_depth:  # Hits ground before Brem point.
                        delta_X_to_ground = max(0, ground_depth - current_depth)
                        energy_at_ground = current_energy - (dEdX_MIP * delta_X_to_ground)
                        particle_in_observation["total_energy"] = numpy.float64(max(energy_at_ground, rest_mass))
                        particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                        non_interacting_particles.append(particle_in_observation)
                        continue

                    # If Brem is in atmosphere, lose energy on the way.
                    energy_loss = dEdX_MIP * delta_X
                    energy_after_loss = current_energy - energy_loss
                    kinetic_energy_after_loss = energy_after_loss - rest_mass

                    if kinetic_energy_after_loss <= energy_threshold or energy_after_loss < rest_mass:  # Stops before Brem point.
                        particle_in_observation["total_energy"] = numpy.float64(max(energy_after_loss, rest_mass))
                        particle_in_observation["atmospheric_depth"] = numpy.float64(X_event)
                        non_interacting_particles.append(particle_in_observation)
                        continue

                    # If still energetic, create Brem products.
                    if kinetic_energy_after_loss > 0:
                        KE_final_e, E_gamma = Energy_distribution_electron_and_photon_or_positron_and_photon(
                            kinetic_energy_after_loss)
                        hydronic_particles.append({"type": ptype, "total_energy": numpy.float64(KE_final_e + rest_mass),"atmospheric_depth": numpy.float64(X_event)})
                        hydronic_particles.append({"type": "photon", "total_energy": numpy.float64(E_gamma),"atmospheric_depth": numpy.float64(X_event)})
                    else:  # If KE is zero, it stopped.
                        particle_in_observation["total_energy"] = numpy.float64(rest_mass)
                        particle_in_observation["atmospheric_depth"] = numpy.float64(X_event)
                        non_interacting_particles.append(particle_in_observation)

                else:  # Low energy: only ionization loss until it stops.
                    KE_initial = current_energy - rest_mass
                    if KE_initial > energy_threshold:
                        delta_X_to_stop = KE_initial / dEdX_MIP if dEdX_MIP > 0 else float('inf')
                        X_stop = current_depth + delta_X_to_stop

                        if X_stop < ground_depth:  # Stops in atmosphere.
                            particle_in_observation["total_energy"] = numpy.float64(rest_mass)
                            particle_in_observation["atmospheric_depth"] = numpy.float64(X_stop)
                            non_interacting_particles.append(particle_in_observation)
                        else:  # Hits ground before stopping.
                            delta_X_boundary = max(0, ground_depth - current_depth)
                            energy_at_boundary = current_energy - (dEdX_MIP * delta_X_boundary)
                            particle_in_observation["total_energy"] = numpy.float64(max(energy_at_boundary, rest_mass))
                            particle_in_observation["atmospheric_depth"] = numpy.float64(ground_depth)
                            non_interacting_particles.append(particle_in_observation)
                    else:  # Already below threshold.
                        non_interacting_particles.append(particle_in_observation)

        # After a single shower is finished, add its final particles to the master list.
        print(f"Shower {shower_id} finished, producing {len(non_interacting_particles)} final particles.")
        for p in non_interacting_particles:
            p['shower_id'] = shower_id  # Tag each particle with its shower ID.
        all_final_particles.extend(non_interacting_particles)

    total_time = time.time() - start_time
    print(f"\n--- All {num_showers} simulations finished in {total_time:.2f} seconds ---")

    # Save the master list of all particles from all showers to a single CSV file.
    if not all_final_particles:
        print("No non-interacting particles were produced in any shower.")
    else:
        print(f"Saving {len(all_final_particles)} total particles to {output_filename}...")
        fieldnames = ['shower_id', 'type', 'total_energy', 'atmospheric_depth']
        try:
            with open(output_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for p in all_final_particles:
                    writer.writerow({'shower_id': p['shower_id'],'type': str(p['type']),'total_energy': float(p['total_energy']),'atmospheric_depth': float(p['atmospheric_depth'])})
            print(f"Successfully saved data to {output_filename}")
        except IOError as e:
            print(f"Error writing to file: {e}")

