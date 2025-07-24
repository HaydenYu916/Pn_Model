import numpy as np


def led_step(
    pwm_percent,
    ambient_temp,
    base_ambient_temp=25.0,
    dt=0.1,
    max_ppfd=500.0,
    max_power=100.0,
    thermal_resistance=2.5,
    thermal_mass=0.5,
):
    """
    Core LED simulation step function
    LED acts as both light source and heater affecting ambient temperature

    Args:
        pwm_percent: PWM duty cycle (0-100%)
        ambient_temp: Current ambient temperature (°C) - evolves due to LED heating
        base_ambient_temp: Base ambient temperature without LED heating (°C)
        dt: Time step (seconds)
        max_ppfd: Maximum PPFD at 100% PWM (μmol/m²/s)
        max_power: Maximum power at 100% PWM (W)
        thermal_resistance: How much LED heating affects ambient temp (K/W)
        thermal_mass: Thermal inertia of the environment

    Returns:
        tuple: (ppfd, new_ambient_temp, power_consumption, efficiency)
    """
    # Clamp PWM to valid range
    pwm_fraction = np.clip(pwm_percent / 100.0, 0.0, 1.0)

    # PPFD output (linear with PWM)
    ppfd = max_ppfd * pwm_fraction

    # LED efficiency (decreases at high power)
    efficiency = 0.8 + 0.2 * np.exp(-pwm_fraction * 2.0)

    # Power consumption
    power = (max_power * pwm_fraction) / efficiency

    # Heat generation (power not converted to light)
    heat_power = power * (1 - efficiency)

    # Target ambient temperature due to LED heating
    target_ambient_temp = base_ambient_temp + heat_power * thermal_resistance

    # Thermal dynamics of environment (first-order)
    thermal_time_constant = thermal_mass * 30.0  # seconds
    alpha = dt / thermal_time_constant
    new_ambient_temp = ambient_temp + alpha * (target_ambient_temp - ambient_temp)

    return ppfd, new_ambient_temp, power, efficiency


def led_steady_state(
    pwm_percent,
    base_ambient_temp=25.0,
    max_ppfd=500.0,
    max_power=100.0,
    thermal_resistance=2.5,
):
    """
    Calculate LED steady-state values including ambient temperature rise

    Args:
        pwm_percent: PWM duty cycle (0-100%)
        base_ambient_temp: Base ambient temperature without LED (°C)
        max_ppfd: Maximum PPFD at 100% PWM (μmol/m²/s)
        max_power: Maximum power at 100% PWM (W)
        thermal_resistance: How much LED heating affects ambient temp (K/W)

    Returns:
        tuple: (ppfd, final_ambient_temp, power_consumption, efficiency)
    """
    pwm_fraction = np.clip(pwm_percent / 100.0, 0.0, 1.0)

    # PPFD output
    ppfd = max_ppfd * pwm_fraction

    # Efficiency
    efficiency = 0.8 + 0.2 * np.exp(-pwm_fraction * 2.0)

    # Power consumption
    power = (max_power * pwm_fraction) / efficiency

    # Steady-state ambient temperature (base + LED heating effect)
    heat_power = power * (1 - efficiency)
    final_ambient_temp = base_ambient_temp + heat_power * thermal_resistance

    return ppfd, final_ambient_temp, power, efficiency


import numpy as np
import matplotlib.pyplot as plt


def led_step(
    pwm_percent,
    ambient_temp,
    base_ambient_temp=25.0,
    dt=0.1,
    max_ppfd=500.0,
    max_power=100.0,
    thermal_resistance=2.5,
    thermal_mass=0.5,
):
    """LED simulation with heating/cooling"""
    pwm_fraction = np.clip(pwm_percent / 100.0, 0.0, 1.0)

    # PPFD output
    ppfd = max_ppfd * pwm_fraction

    # LED efficiency
    efficiency = 0.8 + 0.2 * np.exp(-pwm_fraction * 2.0)

    # Power consumption
    power = (max_power * pwm_fraction) / efficiency

    # Heat generation (when PWM=0, heat_power=0)
    heat_power = power * (1 - efficiency)

    # Target temperature (when PWM=0, target = base_ambient_temp)
    target_ambient_temp = base_ambient_temp + heat_power * thermal_resistance

    # Thermal dynamics
    thermal_time_constant = thermal_mass * 30.0
    alpha = dt / thermal_time_constant
    new_ambient_temp = ambient_temp + alpha * (target_ambient_temp - ambient_temp)

    return ppfd, new_ambient_temp, power, efficiency


def run_led_on_off_example():
    """Example showing LED heating and cooling behavior"""

    # Simulation parameters
    base_ambient = 25.0  # Room temperature without LED
    ambient_temp = 25.0  # Starting temperature
    dt = 1.0  # 1 second time steps

    # Storage for plotting
    time_data = []
    temp_data = []
    ppfd_data = []
    pwm_data = []

    print("LED On/Off Temperature Example")
    print("=" * 50)
    print(f"Base ambient temperature: {base_ambient}°C")
    print()

    current_time = 0

    # Phase 1: LED ON for 60 seconds at 75% PWM
    print("Phase 1: LED ON (75% PWM) - Heating up")
    print("Time\tPWM\tPPFD\tTemp\tRise")
    print("-" * 40)

    for t in range(60):
        ppfd, ambient_temp, power, eff = led_step(75.0, ambient_temp, base_ambient, dt)

        # Store data
        time_data.append(current_time)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        pwm_data.append(75.0)
        current_time += dt

        # Print every 10 seconds
        if t % 10 == 0:
            temp_rise = ambient_temp - base_ambient
            print(f"{t}s\t75%\t{ppfd:.0f}\t{ambient_temp:.1f}°C\t+{temp_rise:.1f}°C")

    print()

    # Phase 2: LED OFF for 60 seconds (0% PWM)
    print("Phase 2: LED OFF (0% PWM) - Cooling down")
    print("Time\tPWM\tPPFD\tTemp\tDiff from base")
    print("-" * 45)

    for t in range(60):
        ppfd, ambient_temp, power, eff = led_step(0.0, ambient_temp, base_ambient, dt)

        # Store data
        time_data.append(current_time)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        pwm_data.append(0.0)
        current_time += dt

        # Print every 10 seconds
        if t % 10 == 0:
            temp_diff = ambient_temp - base_ambient
            print(f"{60+t}s\t0%\t{ppfd:.0f}\t{ambient_temp:.1f}°C\t+{temp_diff:.1f}°C")

    print()

    # Phase 3: LED ON again for 30 seconds at 50% PWM
    print("Phase 3: LED ON again (50% PWM) - Moderate heating")
    print("Time\tPWM\tPPFD\tTemp\tRise")
    print("-" * 40)

    for t in range(30):
        ppfd, ambient_temp, power, eff = led_step(50.0, ambient_temp, base_ambient, dt)

        # Store data
        time_data.append(current_time)
        temp_data.append(ambient_temp)
        ppfd_data.append(ppfd)
        pwm_data.append(50.0)
        current_time += dt

        # Print every 10 seconds
        if t % 10 == 0:
            temp_rise = ambient_temp - base_ambient
            print(
                f"{120+t}s\t50%\t{ppfd:.0f}\t{ambient_temp:.1f}°C\t+{temp_rise:.1f}°C"
            )

    # Final summary
    final_temp_rise = ambient_temp - base_ambient
    print()
    print("=" * 50)
    print("SUMMARY:")
    print(f"Final temperature: {ambient_temp:.1f}°C")
    print(f"Final temperature rise: +{final_temp_rise:.1f}°C above base")
    print(f"Max temperature reached: {max(temp_data):.1f}°C")
    print(f"Max temperature rise: +{max(temp_data) - base_ambient:.1f}°C")

    # Plot results
    plot_results(time_data, temp_data, ppfd_data, pwm_data, base_ambient)

    return time_data, temp_data, ppfd_data, pwm_data


def plot_results(time_data, temp_data, ppfd_data, pwm_data, base_ambient):
    """Plot the simulation results"""

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("LED On/Off Heating and Cooling Example", fontsize=14)

    # Temperature plot
    ax1.plot(time_data, temp_data, "r-", linewidth=2, label="Ambient Temperature")
    ax1.axhline(
        y=base_ambient, color="k", linestyle="--", alpha=0.5, label="Base Ambient"
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature Response")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add phase annotations
    ax1.axvspan(0, 60, alpha=0.2, color="red", label="LED ON (75%)")
    ax1.axvspan(60, 120, alpha=0.2, color="blue", label="LED OFF (0%)")
    ax1.axvspan(120, 150, alpha=0.2, color="orange", label="LED ON (50%)")

    # PPFD plot
    ax2.plot(time_data, ppfd_data, "g-", linewidth=2)
    ax2.set_ylabel("PPFD (μmol/m²/s)")
    ax2.set_title("Light Output (PPFD)")
    ax2.grid(True, alpha=0.3)

    # PWM plot
    ax3.plot(time_data, pwm_data, "b-", linewidth=2)
    ax3.set_ylabel("PWM (%)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("PWM Control Signal")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def simple_on_off_test():
    """Simple test showing basic on/off behavior"""

    print("\nSimple On/Off Test")
    print("=" * 30)

    base_ambient = 25.0
    ambient_temp = 25.0

    # Turn LED ON
    print("Turning LED ON (100% PWM)...")
    for i in range(5):
        ppfd, ambient_temp, power, eff = led_step(
            100.0, ambient_temp, base_ambient, dt=5.0
        )
        print(
            f"After {(i+1)*5}s: {ambient_temp:.1f}°C (+{ambient_temp-base_ambient:.1f}°C)"
        )

    print("\nTurning LED OFF (0% PWM)...")
    for i in range(5):
        ppfd, ambient_temp, power, eff = led_step(
            0.0, ambient_temp, base_ambient, dt=5.0
        )
        print(
            f"After {(i+1)*5}s: {ambient_temp:.1f}°C (+{ambient_temp-base_ambient:.1f}°C)"
        )

    print(f"\nNote: Temperature cooling toward base ambient ({base_ambient}°C)")


# Example usage
if __name__ == "__main__":
    # Initialize environment state
    base_ambient = 25.0  # Room temperature without LED
    ambient_temp = 25.0  # Starting ambient temperature
    dt = 0.1  # 100ms time step

    print("LED Environmental Heating Simulation")
    print("=" * 40)

    # Test step function
    pwm = 75.0
    ppfd, ambient_temp, power, eff = led_step(pwm, ambient_temp, base_ambient, dt=dt)

    print(f"PWM: {pwm}%")
    print(f"PPFD: {ppfd:.1f} μmol/m²/s")
    print(f"Ambient Temperature: {ambient_temp:.1f} °C")
    print(f"Power: {power:.1f} W")
    print(f"Efficiency: {eff*100:.1f}%")
    print(f"Temperature Rise: {ambient_temp - base_ambient:.1f} °C")

    # Test steady-state
    print("\nSteady-state values:")
    ppfd_ss, temp_ss, power_ss, eff_ss = led_steady_state(pwm, base_ambient)
    print(f"SS PPFD: {ppfd_ss:.1f} μmol/m²/s")
    print(f"SS Ambient Temperature: {temp_ss:.1f} °C")
    print(f"SS Temperature Rise: {temp_ss - base_ambient:.1f} °C")
    print(f"SS Power: {power_ss:.1f} W")
    print(f"SS Efficiency: {eff_ss*100:.1f}%")

    # Simulation example - LED heating up the environment
    print("\nEnvironment heating over time:")
    ambient_temp = base_ambient  # Start at base ambient
    for t in range(0, 100, 10):  # 10 seconds intervals
        ppfd, ambient_temp, power, eff = led_step(
            75.0, ambient_temp, base_ambient, dt=1.0
        )
        temp_rise = ambient_temp - base_ambient
        print(
            f"t={t}s: Ambient={ambient_temp:.1f}°C (rise: +{temp_rise:.1f}°C), PPFD={ppfd:.0f}"
        )
    # Initialize environment state
    base_ambient = 25.0  # Room temperature without LED
    ambient_temp = 25.0  # Starting ambient temperature
    dt = 0.1  # 100ms time step

    print("LED Environmental Heating Simulation")
    print("=" * 40)

    # Test step function
    pwm = 75.0
    ppfd, ambient_temp, power, eff = led_step(pwm, ambient_temp, base_ambient, dt=dt)

    print(f"PWM: {pwm}%")
    print(f"PPFD: {ppfd:.1f} μmol/m²/s")
    print(f"Ambient Temperature: {ambient_temp:.1f} °C")
    print(f"Power: {power:.1f} W")
    print(f"Efficiency: {eff*100:.1f}%")
    print(f"Temperature Rise: {ambient_temp - base_ambient:.1f} °C")

    # Test steady-state
    print("\nSteady-state values:")
    ppfd_ss, temp_ss, power_ss, eff_ss = led_steady_state(pwm, base_ambient)
    print(f"SS PPFD: {ppfd_ss:.1f} μmol/m²/s")
    print(f"SS Ambient Temperature: {temp_ss:.1f} °C")
    print(f"SS Temperature Rise: {temp_ss - base_ambient:.1f} °C")
    print(f"SS Power: {power_ss:.1f} W")
    print(f"SS Efficiency: {eff_ss*100:.1f}%")

    # Simulation example - LED heating up the environment
    print("\nEnvironment heating over time:")
    ambient_temp = base_ambient  # Start at base ambient
    for t in range(0, 100, 10):  # 10 seconds intervals
        ppfd, ambient_temp, power, eff = led_step(
            75.0, ambient_temp, base_ambient, dt=1.0
        )
        temp_rise = ambient_temp - base_ambient
        print(
            f"t={t}s: Ambient={ambient_temp:.1f}°C (rise: +{temp_rise:.1f}°C), PPFD={ppfd:.0f}"
        )
    # Run the detailed example
    run_led_on_off_example()

    # Run simple test
    simple_on_off_test()
