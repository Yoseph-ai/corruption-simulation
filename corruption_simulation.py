import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm, entropy
import random

class CorruptionSimulation:
    def __init__(self, N=5000, T=20, shielding=0.6, shock_time=5, shock_magnitude=1.2):
        # Core parameters
        self.N = N
        self.T = T
        self.shielding = shielding
        self.shock_time = shock_time
        self.shock_magnitude = shock_magnitude
        
        # Statistical parameters
        self.μ_G, self.σ_G = 55.04, 5.5
        self.μ_p, self.σ_p = 299.92, 30.0
        self.μ_r, self.σ_r = 0.175, 0.02
        self.β_G, self.β_p, self.β_r = 0.15, -0.15, -0.25  # Adjusted parameters
        
        # Initialize agents
        self.initialize_agents()
        
    def initialize_agents(self):
        # Initialize arrays
        self.G = np.abs(norm.rvs(self.μ_G, self.σ_G, self.N))
        self.p = np.abs(norm.rvs(self.μ_p, self.σ_p, self.N))
        self.r = np.clip(norm.rvs(self.μ_r, self.σ_r, self.N), 0.05, 0.95)
        
        # States and roles - use object dtype for full string storage
        self.m = np.zeros(self.N)  # Moral state [-1,1]
        self.s = np.zeros(self.N)  # System impact [-1,1]
        self.roles = np.empty(self.N, dtype=object)  # Changed to object dtype
        self.roles[:] = 'Bystander'  # Initialize all as Bystanders
        
        # Initialize specific roles
        self.initialize_roles()
    
    def initialize_roles(self):
        # Perpetrators (12%)
        idx_p = np.random.choice(self.N, int(0.12*self.N), replace=False)
        self.m[idx_p] = np.random.uniform(-0.8, -0.3, len(idx_p))
        self.s[idx_p] = np.random.uniform(-0.9, -0.4, len(idx_p))
        self.roles[idx_p] = 'Perpetrator'
        
        # Whistleblowers (8%)
        idx_w = np.random.choice(np.setdiff1d(range(self.N), idx_p), 
                               int(0.08*self.N), replace=False)
        self.m[idx_w] = np.random.uniform(0.6, 0.9, len(idx_w))
        self.s[idx_w] = np.random.uniform(0.3, 0.7, len(idx_w))
        self.roles[idx_w] = 'Whistleblower'
        
        # Bystanders (35% - circular around origin)
        idx_b = np.random.choice(np.setdiff1d(range(self.N), np.concatenate([idx_p, idx_w])), 
                               int(0.35*self.N), replace=False)
        angles = np.random.uniform(0, 2*np.pi, len(idx_b))
        radii = np.random.uniform(0, 0.2, len(idx_b))
        self.m[idx_b] = radii * np.cos(angles)
        self.s[idx_b] = radii * np.sin(angles)
        self.roles[idx_b] = 'Bystander'
        
        # Enforcers (6%)
        idx_e = np.random.choice(np.setdiff1d(range(self.N), np.concatenate([idx_p, idx_w, idx_b])), 
                               int(0.06*self.N), replace=False)
        self.m[idx_e] = np.random.uniform(0.2, 0.8, len(idx_e))
        self.s[idx_e] = np.random.uniform(0.4, 0.8, len(idx_e))
        self.roles[idx_e] = 'Enforcer'
        
        # Neutral for remainder
        remaining = np.setdiff1d(range(self.N), np.concatenate([idx_p, idx_w, idx_b, idx_e]))
        self.roles[remaining] = 'Neutral'
    
    def binary_decision(self, i):
        threshold = (self.p[i] * self.r[i]) / max(1 - self.r[i], 0.01)
        return 1 if self.G[i] > threshold else 0
    
    def get_probability(self, i):
        z_G = (self.G[i] - self.μ_G) / self.σ_G
        z_p = (self.p[i] - self.μ_p) / self.σ_p
        z_r = (self.r[i] - self.μ_r) / self.σ_r
        logit = self.β_G*z_G + self.β_p*z_p + self.β_r*z_r
        return 1 / (1 + np.exp(-logit))
    
    def probabilistic_decision(self, i):
        prob = self.get_probability(i)
        return 1 if np.random.random() < prob else 0
    
    def hybrid_decision(self, i):
        prob_val = self.get_probability(i)
        
        # Add noise to probability
        noisy_prob = prob_val + np.random.normal(0, 0.05)  # Reduced noise magnitude
        noisy_prob = np.clip(noisy_prob, 0.01, 0.99)  # Keep within valid range
        
        if noisy_prob < 0.2 or noisy_prob > 0.8:
            return self.binary_decision(i)
        else:
            return 1 if np.random.random() < noisy_prob else 0
    
    def update_role(self, i):
        m, s = self.m[i], self.s[i]
        if m < -0.3 and s < -0.3:
            return 'Perpetrator'
        elif m < 0 and 0 < s <= 0.3:
            return 'Colluder'
        elif m > 0 and s < -0.3:
            return 'Victim'
        elif m > 0.6 and s > 0.3:
            return 'Whistleblower'
        elif m > 0.2 and s > 0.4:
            return 'Enforcer'
        elif m > 0.7 and s > 0.7:
            return 'Clean'
        elif np.sqrt(m**2 + s**2) <= 0.2:
            return 'Bystander'
        else:
            return 'Neutral'
    
    def update_agent(self, i, t):
        # Make corruption decision
        corrupt = self.hybrid_decision(i)
        
        # Apply institutional shock
        if t == self.shock_time:
            self.r[i] = np.clip(self.r[i] * self.shock_magnitude, 0.05, 0.95)
            # Add state perturbation to ensure system reacts
            self.m[i] = np.clip(self.m[i] + np.random.uniform(-0.2, 0.2), -1, 1)
        
        # Update states based on role
        if self.roles[i] == 'Perpetrator':
            self.m[i] -= 0.15 * corrupt  # Moral decay when corrupt
            self.s[i] -= 0.12 * corrupt  # System impact decreases
            
            # Rehabilitation chance when not acting corruptly
            if not corrupt and np.random.random() < 0.05:  # 5% chance per step
                self.roles[i] = 'Bystander'
                
        elif self.roles[i] == 'Whistleblower':
            if corrupt:
                # Reduced penalties with shielding
                self.m[i] -= 0.1 * (1 - self.shielding)
                self.s[i] -= 0.15 * (1 - self.shielding)
            else:
                self.m[i] += 0.15  # Increased reward
                self.s[i] += 0.12 * self.shielding  # Shielding effect
                
        elif self.roles[i] == 'Bystander':
            # Entropic drift toward passivity
            self.m[i] -= 0.01 * (1 - abs(self.m[i]))
        elif self.roles[i] == 'Enforcer':
            # Enforcers improve system when ethical
            if not corrupt:
                self.s[i] += 0.05
        
        # Bound states
        self.m[i] = np.clip(self.m[i], -1, 1)
        self.s[i] = np.clip(self.s[i], -1, 1)
        
        # Update role
        self.roles[i] = self.update_role(i)
        
        return corrupt
    
    def run(self):
        history = []
        agent_history = []  # Store agent states for animation
        
        for t in range(self.T):
            corruption_count = 0
            for i in range(self.N):
                corrupt = self.update_agent(i, t)
                corruption_count += corrupt
            
            # Record state
            role_counts = pd.Series(self.roles).value_counts(normalize=True)
            history.append({
                't': t,
                'avg_m': np.mean(self.m),
                'avg_s': np.mean(self.s),
                'corruption_rate': corruption_count / self.N,
                **role_counts.to_dict()
            })
            
            # Store agent states for animation
            agent_history.append({
                'm': self.m.copy(),
                's': self.s.copy(),
                'roles': self.roles.copy()
            })
        
        return pd.DataFrame(history), agent_history
    
    def plot_results(self, results):
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Role distribution
        roles = ['Perpetrator', 'Whistleblower', 'Bystander', 'Enforcer']
        for role in roles:
            if role in results:
                axs[0].plot(results['t'], results[role], label=role, linewidth=2)
        axs[0].axvline(self.shock_time, color='gray', linestyle='--', alpha=0.7)
        axs[0].set_title('Role Distribution Over Time')
        axs[0].set_ylabel('Fraction of Agents')
        axs[0].legend()
        axs[0].grid(alpha=0.3)
        
        # System states
        axs[1].plot(results['t'], results['avg_m'], 'b-', label='Moral State', linewidth=2)
        axs[1].plot(results['t'], results['avg_s'], 'r-', label='System Impact', linewidth=2)
        axs[1].axvline(self.shock_time, color='gray', linestyle='--', alpha=0.7)
        axs[1].set_title('Average System States')
        axs[1].set_ylabel('State Value')
        axs[1].legend()
        axs[1].grid(alpha=0.3)
        
        # Corruption rate
        axs[2].plot(results['t'], results['corruption_rate'], 'g-', linewidth=3)
        axs[2].axvline(self.shock_time, color='gray', linestyle='--', alpha=0.7)
        axs[2].set_title('System Corruption Rate')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Corruption Rate')
        axs[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('corruption_simulation_results.png', dpi=300)
        plt.close()  # Close figure to prevent Tkinter conflicts
    
    def create_animation(self, results, agent_history):
        """Create animated visualization of role changes"""
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        
        # Custom color map for roles
        role_colors = {
            'Perpetrator': '#d62728',   # red
            'Colluder': '#9467bd',      # purple
            'Victim': '#1f77b4',        # blue
            'Whistleblower': '#2ca02c', # green
            'Enforcer': '#ff7f0e',      # orange
            'Clean': '#ffff00',         # yellow
            'Bystander': '#7f7f7f',     # gray
            'Neutral': '#e0e0e0'        # light gray
        }
        
        # Role distribution plot (left)
        ax[0].set_xlim(0, self.T)
        ax[0].set_ylim(0, 1)
        ax[0].set_title('Role Distribution Over Time')
        ax[0].set_xlabel('Time Step')
        ax[0].set_ylabel('Fraction of Agents')
        ax[0].grid(alpha=0.3)
        
        # 2D state space plot (right)
        ax[1].set_xlim(-1, 1)
        ax[1].set_ylim(-1, 1)
        ax[1].set_title('Agent Distribution in Moral-Impact Space')
        ax[1].set_xlabel('System Impact (s)')
        ax[1].set_ylabel('Moral State (m)')
        ax[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        ax[1].axvline(0, color='k', linestyle='--', alpha=0.3)
        ax[1].axvline(-0.3, color='r', linestyle=':', alpha=0.3)
        ax[1].axvline(0.3, color='r', linestyle=':', alpha=0.3)
        ax[1].axhline(-0.3, color='b', linestyle=':', alpha=0.3)
        ax[1].axhline(0.3, color='b', linestyle=':', alpha=0.3)
        
        # Initialize plots
        time_text = ax[0].text(0.02, 0.95, '', transform=ax[0].transAxes)
        role_lines = {}
        for role, color in role_colors.items():
            role_lines[role], = ax[0].plot([], [], label=role, color=color, lw=2)
        
        # Initialize scatter plot
        scatter = ax[1].scatter([], [], alpha=0.6, s=10)
        
        # Data storage
        x_data = {role: [] for role in role_colors}
        y_data = {role: [] for role in role_colors}
        
        # Animation update function
        def update(frame):
            # Update time text
            if frame < len(results):
                corruption = results["corruption_rate"].iloc[frame]
            else:
                corruption = 0
            time_text.set_text(f'Time: {frame}/{self.T}\nCorruption: {corruption:.1%}')
            
            # Update role distribution plot
            for role in role_colors:
                if role in results.columns and frame < len(results):
                    x_data[role].append(frame)
                    y_data[role].append(results[role].iloc[frame])
                    role_lines[role].set_data(x_data[role], y_data[role])
            
            # Update scatter plot
            current_data = agent_history[frame]
            scatter.set_offsets(np.column_stack((current_data['s'], current_data['m'])))
            
            # Use get() with default color to prevent KeyErrors
            colors = [role_colors.get(role, '#000000') for role in current_data['roles']]
            scatter.set_color(colors)
            
            return [time_text] + list(role_lines.values()) + [scatter]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=self.T, interval=500, blit=False  # Disable blitting to prevent Tkinter errors
        )
        
        # Add legend
        ax[0].legend(loc='upper right')
        
        # Save animation with robust error handling
        print("Saving animation... (this may take several minutes)")
        try:
            # First try MP4
            ani.save('role_evolution.mp4', writer='ffmpeg', fps=2, dpi=150)
            print("Animation saved as role_evolution.mp4")
        except Exception as e:
            print(f"MP4 save failed: {e}")
            try:
                # Then try GIF
                print("Saving as GIF instead of MP4")
                ani.save('role_evolution.gif', writer='pillow', fps=2)
                print("Animation saved as role_evolution.gif")
            except Exception as e:
                # Final fallback to static frames
                print(f"Animation failed: {e}. Saving static frames instead")
                
                # Create figure for static frames
                fig, ax = plt.subplots(1, 2, figsize=(16, 7))
                
                # Initialize empty plots
                role_lines = {}
                for role, color in role_colors.items():
                    role_lines[role], = ax[0].plot([], [], label=role, color=color, lw=2)
                
                # Setup axes
                ax[0].set_xlim(0, self.T)
                ax[0].set_ylim(0, 1)
                ax[0].set_title('Role Distribution Over Time')
                ax[0].set_xlabel('Time Step')
                ax[0].set_ylabel('Fraction of Agents')
                ax[0].grid(alpha=0.3)
                ax[0].legend(loc='upper right')
                
                ax[1].set_xlim(-1, 1)
                ax[1].set_ylim(-1, 1)
                ax[1].set_title('Agent Distribution in Moral-Impact Space')
                ax[1].set_xlabel('System Impact (s)')
                ax[1].set_ylabel('Moral State (m)')
                ax[1].axhline(0, color='k', linestyle='--', alpha=0.3)
                ax[1].axvline(0, color='k', linestyle='--', alpha=0.3)
                
                time_text = ax[0].text(0.02, 0.95, '', transform=ax[0].transAxes)
                
                # Store data for each role
                x_data = {role: [] for role in role_colors}
                y_data = {role: [] for role in role_colors}
                
                # Save each frame individually
                for t in range(self.T):
                    # Update time text
                    if t < len(results):
                        time_text.set_text(f'Time: {t}/{self.T}\nCorruption: {results["corruption_rate"].iloc[t]:.1%}')
                    else:
                        time_text.set_text(f'Time: {t}/{self.T}')
                    
                    # Update role distribution plot
                    for role in role_colors:
                        if role in results.columns and t < len(results):
                            x_data[role].append(t)
                            y_data[role].append(results[role].iloc[t])
                            role_lines[role].set_data(x_data[role], y_data[role])
                    
                    # Update scatter plot
                    current_data = agent_history[t]
                    ax[1].clear()
                    ax[1].set_xlim(-1, 1)
                    ax[1].set_ylim(-1, 1)
                    ax[1].set_title(f'Agent Distribution at t={t}')
                    colors = [role_colors.get(r, '#000000') for r in current_data['roles']]
                    ax[1].scatter(current_data['s'], current_data['m'], alpha=0.6, s=10, c=colors)
                    
                    # Save frame
                    plt.tight_layout()
                    plt.savefig(f'frame_{t:03d}.png', dpi=150)
                    print(f"Saved frame {t:03d}.png")
                
                print(f"Saved {self.T} frames as PNG files")
                plt.close(fig)
        
        plt.close()
        return ani

def plot_role_trajectory(role_history, role_labels, time_steps):
    """Plot evolution of role counts over time"""
    role_counts = {role: [] for role in role_labels}
    
    for t in range(time_steps):
        roles_at_t = role_history[:, t]
        for role in role_labels:
            role_counts[role].append(np.sum(roles_at_t == role))

    plt.figure(figsize=(12, 6))
    for role, counts in role_counts.items():
        plt.plot(range(time_steps), counts, label=role)
    plt.title("Role Evolution Trajectories Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Agent Count")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('role_trajectory.png', dpi=300)
    plt.close()

def plot_role_entropy(role_history, role_labels):
    """Calculate and plot entropy of role distribution"""
    entropies = []
    for t in range(role_history.shape[1]):
        roles_at_t = role_history[:, t]
        counts = np.array([np.sum(roles_at_t == role) for role in role_labels])
        probs = counts / np.sum(counts)
        entropies.append(entropy(probs, base=2))

    plt.figure(figsize=(10, 4))
    plt.plot(entropies, color='crimson')
    plt.title("Entropy of Role Distribution Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Entropy (bits)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('role_entropy.png', dpi=300)
    plt.close()

def compare_ms_clustering(m_matrix, s_matrix, shock_time):
    """Compare agent distribution before and after shock"""
    t_pre = max(0, shock_time - 1)
    t_post = min(shock_time + 5, m_matrix.shape[1]-1)

    plt.figure(figsize=(12, 5))

    # Before Shock
    plt.subplot(1, 2, 1)
    plt.scatter(m_matrix[:, t_pre], s_matrix[:, t_pre], alpha=0.6, color='steelblue')
    plt.title(f"Agent Distribution Before Shock (t={t_pre})")
    plt.xlabel("Moral State")
    plt.ylabel("System Impact")
    plt.grid(True)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])

    # After Shock
    plt.subplot(1, 2, 2)
    plt.scatter(m_matrix[:, t_post], s_matrix[:, t_post], alpha=0.6, color='darkorange')
    plt.title(f"Agent Distribution After Shock (t={t_post})")
    plt.xlabel("Moral State")
    plt.ylabel("System Impact")
    plt.grid(True)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])

    plt.tight_layout()
    plt.savefig('shock_comparison.png', dpi=300)
    plt.close()

def analyze_results(results, shock_time):
    print("\n=== SIMULATION SUMMARY ===")
    print(f"Initial Corruption: {results['corruption_rate'].iloc[0]:.2%}")
    print(f"Final Corruption: {results['corruption_rate'].iloc[-1]:.2%}")
    print(f"Change: {(results['corruption_rate'].iloc[-1]-results['corruption_rate'].iloc[0]):.2%}")
    
    # Role analysis
    print("\n=== ROLE DISTRIBUTION CHANGES ===")
    roles = ['Perpetrator', 'Whistleblower', 'Bystander']
    for role in roles:
        if role in results:
            init = results[role].iloc[0]
            final = results[role].iloc[-1]
            print(f"{role}: {init:.2%} → {final:.2%} (Δ: {final-init:+.2%})")
    
    # Shock response - fixed indexing
    results['t'] = results['t'].astype(int)
    shock_idx = results.index[results['t'] == shock_time].tolist()
    if shock_idx:
        shock_idx = shock_idx[0]
        if shock_idx > 0 and shock_idx < len(results):
            before = results['corruption_rate'].iloc[shock_idx-1]
            after = results['corruption_rate'].iloc[shock_idx]
            print(f"\nShock Response at t={shock_time}: Immediate drop of {before-after:.2%}")
        else:
            print("\nShock time not valid for response calculation")
    else:
        print("\nShock time not found in results")

# Main execution
if __name__ == "__main__":
    print("Starting Corruption System Simulation...")
    
    # Configuration
    config = {
        'N': 5000,           # Reduced for animation performance
        'T': 20,
        'shielding': 0.6,
        'shock_time': 5,
        'shock_magnitude': 1.2  # Reduced shock magnitude
    }
    
    # Create and run simulation
    sim = CorruptionSimulation(**config)
    results, agent_history = sim.run()
    
    # Plot standard results
    sim.plot_results(results)
    analyze_results(results, config['shock_time'])
    
    # Create animation
    sim.create_animation(results, agent_history)
    
    # Prepare data for additional plots
    role_labels = ["Perpetrator", "Colluder", "Victim", "Whistleblower", 
                  "Enforcer", "Clean", "Bystander", "Neutral"]
    
    # Convert agent history to matrices
    role_history = np.array([step['roles'] for step in agent_history]).T
    m_matrix = np.array([step['m'] for step in agent_history]).T
    s_matrix = np.array([step['s'] for step in agent_history]).T
    
    # Generate additional visualizations
    plot_role_trajectory(role_history, role_labels, config['T'])
    plot_role_entropy(role_history, role_labels)
    compare_ms_clustering(m_matrix, s_matrix, config['shock_time'])
    
    # Save results
    results.to_csv('simulation_results.csv', index=False)
    print("\n=== SIMULATION COMPLETE ===")
    print("Results saved to simulation_results.csv")
    print("Visualizations saved as PNG files")