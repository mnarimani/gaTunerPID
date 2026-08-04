import numpy as np
from typing import Dict, Any, List


class SystemSimulator:
    """Simulates control system and evaluates performance"""

    def __init__(self, dynamics_func, system_config: Dict[str, Any]):
        """
        Args:
            dynamics_func: Function with signature (t, x, u) -> dx/dt
            system_config: Dict with dt, max_time, target, etc.
        """
        self.dynamics_func = dynamics_func
        self.dt = system_config['dt']
        self.max_time = system_config['max_time']
        self.target = system_config['target']
        self.num_inputs = system_config['num_inputs']
        self.input_channel = system_config['input_channel']
        self.output_channel = system_config['output_channel']
        self.min_ctrl = system_config['min_ctrl']
        self.max_ctrl = system_config['max_ctrl']
        self.trim_values = np.array(system_config.get('trim_values', [0.0] * self.num_inputs))
        self.trim_ics = np.array(system_config.get('trim_ics', [0.0] * system_config.get('num_states', 2)))
        self.num_states = system_config.get('num_states', 2)

    def evaluate_pid(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate PID controller with fixed initial conditions (no randomness)"""
        Kp = params['Kp']
        Ki = params['Ki']
        Kd = params['Kd']

        # Use fixed initial conditions from trim_ics
        result = self._simulate_pid(self.trim_ics, Kp, Ki, Kd)

        if not result['success']:
            return {
                'success': False,
                'metrics': None,
                'warnings': ['Simulation failed with fixed initial conditions']
            }

        return {
            'success': True,
            'metrics': result['metrics'],
            'controller_parameters': params,
            'warnings': []
        }

    def _simulate_pid(self, x0: np.ndarray, Kp: float, Ki: float, Kd: float) -> Dict[str, Any]:
        """Simulate single trajectory with PID controller"""
        time_steps = np.arange(0, self.max_time, self.dt)
        trajectory = []
        control_signals = []

        x = x0.copy()
        integral = 0.0
        prev_error = self.target - x[self.output_channel]

        try:
            for t in time_steps:
                # PID control
                error = self.target - x[self.output_channel]
                integral += error * self.dt
                derivative = (error - prev_error) / self.dt

                u = Kp * error + Ki * integral + Kd * derivative

                # Full control vector
                u_vec = self.trim_values.copy()
                u += u_vec[self.input_channel]
                u = np.clip(u, self.min_ctrl, self.max_ctrl)
                u_vec[self.input_channel] = u

                # Integrate dynamics
                dx = self.dynamics_func(t, x, u_vec)
                x = x + dx * self.dt

                # Check for instability
                if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
                    return {'success': False, 'metrics': None}

                trajectory.append(x[self.output_channel])
                control_signals.append(u)
                prev_error = error

            # Compute metrics
            trajectory = np.array(trajectory)
            control_signals = np.array(control_signals)

            metrics = self._compute_metrics(trajectory, control_signals, time_steps)

            return {'success': True, 'metrics': metrics}

        except Exception as e:
            print(f"Simulation error: {e}")
            return {'success': False, 'metrics': None}

    def _compute_metrics(self, trajectory: np.ndarray, control: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """Compute performance metrics from trajectory"""
        error = trajectory - self.target

        # MSE
        mse = np.mean(error ** 2)

        # Get initial error magnitude for overshoot calculation
        initial_error = np.abs(error[0])

        # Settling time (2% criterion)
        settling_band = 0.02 * abs(self.target) if self.target != 0 else 0.02
        settled_idx = np.where(np.abs(error) > settling_band)[0]
        settling_time = time[settled_idx[-1]] if len(settled_idx) > 0 else time[0]

        # Overshoot calculation (based on reference implementation)
        if initial_error > 1e-6:
            # Find when we first get close to target (within 10% of initial error)
            approach_threshold = 0.1 * initial_error
            approach_indices = np.where(np.abs(error) < approach_threshold)[0]

            if len(approach_indices) > 0:
                # Look for overshoot after first approach
                first_approach = approach_indices[0]

                if first_approach < len(error) - 1:
                    errors_after_approach = error[first_approach:]

                    # Check if error changes sign (crosses target)
                    initial_sign = np.sign(error[0])
                    crossed_indices = np.where(np.sign(errors_after_approach) == -initial_sign)[0]

                    if len(crossed_indices) > 0:
                        # Find maximum deviation beyond target
                        max_overshoot_error = np.max(np.abs(errors_after_approach[crossed_indices]))
                        # Express as percentage of initial error
                        overshoot = (max_overshoot_error / initial_error) * 100
                    else:
                        overshoot = 0.0
                else:
                    overshoot = 0.0
            else:
                overshoot = 0.0
        else:
            overshoot = 0.0

        # Normalized control effort
        max_abs_u = max(abs(self.min_ctrl), abs(self.max_ctrl))
        num_steps = len(control)
        max_possible_effort = max_abs_u * num_steps
        control_effort = np.sum(np.abs(control)) / max_possible_effort

        # Rise time (10%-90%)
        if self.target != 0:
            band_10 = self.target * 0.1
            band_90 = self.target * 0.9
            idx_10 = np.where(trajectory >= band_10)[0]
            idx_90 = np.where(trajectory >= band_90)[0]
            rise_time = (time[idx_90[0]] - time[idx_10[0]]) if len(idx_10) > 0 and len(idx_90) > 0 else self.max_time
        else:
            rise_time = 0.0

        # Steady-state error
        steady_state_error = abs(np.mean(trajectory[-int(0.2 / self.dt):]) - self.target)

        # Count sign changes in control signal
        control_signs = np.sign(control)
        # Remove zeros to avoid false crossings
        control_signs_nonzero = control_signs[control_signs != 0]
        # Count transitions
        sign_changes = np.diff(control_signs_nonzero)

        return {
            'mse': float(mse),
            'settling_time': float(settling_time),
            'overshoot': float(overshoot),
            'control_effort': float(control_effort),
            'rise_time': float(rise_time),
            'steady_state_error': float(steady_state_error),
            'control_zero_crossings': int(np.count_nonzero(sign_changes))
        }

    def simulate_and_return_trajectory(self, params: Dict[str, float], x0: np.ndarray = None) -> Dict[str, Any]:
        """Run one simulation and return full time series for plotting"""
        if x0 is None:
            x0 = np.random.uniform(-1.0, 1.0, self.num_states)

        Kp = params['Kp']
        Ki = params['Ki']
        Kd = params['Kd']

        time_steps = np.arange(0, self.max_time, self.dt)
        trajectory = []
        control_signals = []

        x = x0.copy()
        integral = 0.0
        prev_error = self.target - x[self.output_channel]

        try:
            for t in time_steps:
                error = self.target - x[self.output_channel]
                integral += error * self.dt
                derivative = (error - prev_error) / self.dt

                u = Kp * error + Ki * integral + Kd * derivative

                u_vec = self.trim_values.copy()
                u += u_vec[self.input_channel]
                u = np.clip(u, self.min_ctrl, self.max_ctrl)
                u_vec[self.input_channel] = u

                dx = self.dynamics_func(t, x, u_vec)
                x = x + dx * self.dt

                if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e6):
                    return {'success': False}

                trajectory.append(x[self.output_channel])
                control_signals.append(u)
                prev_error = error

            return {
                'success': True,
                'time': time_steps,
                'trajectory': np.array(trajectory),
                'control_signals': np.array(control_signals)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}