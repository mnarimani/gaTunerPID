import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
import json

from src.ga_optimizer import GAOptimizer
from src.simulator import SystemSimulator


def load_case_study(json_filename: str) -> Dict[str, Any]:
    """Load a case study JSON file from the case_studies directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "case_studies", "json", json_filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Case study file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Post-process: trim_values can be float or list; ensure it's a list
    if isinstance(data.get("trim_values"), (int, float)):
        data["trim_values"] = [float(data["trim_values"])]
    elif isinstance(data.get("trim_values"), list):
        data["trim_values"] = [float(v) for v in data["trim_values"]]
    else:
        data["trim_values"] = [0.0]

    return data


def get_available_case_studies() -> list:
    """Get list of available JSON case studies."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(base_dir, "case_studies", "json")

    if not os.path.exists(json_dir):
        return []

    return [f for f in os.listdir(json_dir) if f.endswith('.json')]


st.set_page_config(page_title="GA-Based PID Controller Tuning", layout="wide")

st.title("🎛️ GA-Based PID Controller Tuning")
st.markdown("Upload dynamics code and tune PID controllers using Genetic Algorithms")

# Sidebar for system configuration
with st.sidebar:
    st.header("⚙️ System Configuration")

    # Option to choose between JSON case study or file upload
    input_mode = st.radio(
        "Configuration Mode:",
        ["Load JSON Case Study", "Upload Dynamics File"],
        help="Choose to load a predefined case study or upload your own dynamics file"
    )

    dynamics_path = None
    case_study_data = None

    if input_mode == "Load JSON Case Study":
        # Get available case studies
        available_studies = get_available_case_studies()

        if not available_studies:
            st.error("❌ No case studies found in case_studies/json/")
        else:
            selected_study = st.selectbox(
                "Select Case Study:",
                available_studies,
                help="Choose from available predefined case studies"
            )

            if st.button("📂 Load Case Study", use_container_width=True):
                try:
                    case_study_data = load_case_study(selected_study)
                    st.session_state['case_study_data'] = case_study_data
                    st.success(f"✅ Loaded: {case_study_data['system_name']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading case study: {str(e)}")

            # Load from session state if available
            if 'case_study_data' in st.session_state:
                case_study_data = st.session_state['case_study_data']

                # Display case study info
                st.info(f"**System:** {case_study_data['system_name']}\n\n"
                        f"**Description:** {case_study_data['system_description']}\n\n"
                        f"**Objective:** {case_study_data['control_objective']}")

    else:  # Upload Dynamics File
        uploaded_file = st.file_uploader("Upload Python dynamics file (.py)", type=['py'])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w') as tmp:
                tmp.write(uploaded_file.read().decode('utf-8'))
                dynamics_path = tmp.name
            st.success(f"✅ Uploaded: {uploaded_file.name}")

    st.divider()

    # Simulation parameters - use case study values if available
    st.subheader("Simulation Parameters")

    if case_study_data:
        # Use case study defaults but allow override
        dt = st.number_input("Time step (dt)", value=0.01, min_value=0.001, max_value=0.1, format="%.4f", disabled=True)
        max_time = st.number_input("Simulation time (s)", value=10.0, min_value=0.1, max_value=100.0, disabled=True)
        target = st.number_input("Target setpoint", value=float(case_study_data['target']), format="%.4f",
                                 disabled=True)
    else:
        dt = st.number_input("Time step (dt)", value=0.01, min_value=0.001, max_value=0.1, format="%.4f")
        max_time = st.number_input("Simulation time (s)", value=10.0, min_value=0.1, max_value=100.0)
        target = st.number_input("Target setpoint", value=0.1, format="%.4f")

    st.divider()

    # System dimensions - use case study values if available
    st.subheader("System Dimensions")

    if case_study_data:
        num_states = st.number_input("Number of states", value=int(case_study_data['num_states']), min_value=1,
                                     max_value=20, disabled=True)
        num_inputs = st.number_input("Number of inputs", value=int(case_study_data['num_inputs']), min_value=1,
                                     max_value=10, disabled=True)
        input_channel = st.number_input("Input channel (0-indexed)", value=int(case_study_data['input_channel']),
                                        min_value=0, max_value=num_inputs - 1, disabled=True)
        output_channel = st.number_input("Output channel (0-indexed)", value=int(case_study_data['output_channel']),
                                         min_value=0, max_value=num_states - 1, disabled=True)
    else:
        num_states = st.number_input("Number of states", value=3, min_value=1, max_value=20)
        num_inputs = st.number_input("Number of inputs", value=1, min_value=1, max_value=10)
        input_channel = st.number_input("Input channel (0-indexed)", value=0, min_value=0, max_value=num_inputs - 1)
        output_channel = st.number_input("Output channel (0-indexed)", value=0, min_value=0, max_value=num_states - 1)

    st.divider()

    # Control limits - use case study values if available
    st.subheader("Control Limits")

    if case_study_data:
        min_ctrl = st.number_input("Min control", value=float(case_study_data['min_ctrl']), format="%.4f",
                                   disabled=True)
        max_ctrl = st.number_input("Max control", value=float(case_study_data['max_ctrl']), format="%.4f",
                                   disabled=True)
        trim_values = case_study_data['trim_values']
        st.text_input("Trim values (comma-separated)", value=",".join([str(v) for v in trim_values]), disabled=True)
    else:
        min_ctrl = st.number_input("Min control", value=-0.5, format="%.4f")
        max_ctrl = st.number_input("Max control", value=0.5, format="%.4f")

        # Trim values
        trim_values_str = st.text_input("Trim values (comma-separated)", value=",".join(["0.0"] * num_inputs))
        try:
            trim_values = [float(v.strip()) for v in trim_values_str.split(",") if v.strip()]
            if len(trim_values) != num_inputs:
                trim_values = [0.0] * num_inputs
                st.warning(f"⚠️ Using zeros for {num_inputs} inputs")
        except ValueError:
            trim_values = [0.0] * num_inputs
            st.warning("⚠️ Invalid trim values, using zeros")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 PID Parameter Ranges")

    Kp_range = st.slider("Kp search range", 0.0, 100.0, (1.0, 50.0), step=0.1)
    Ki_range = st.slider("Ki search range", 0.0, 20.0, (0.01, 5.0), step=0.01)
    Kd_range = st.slider("Kd search range", 0.0, 20.0, (0.01, 10.0), step=0.01)

with col2:
    st.header("🎯 Target Metrics")

    target_mse = st.number_input("Target MSE", value=0.1, min_value=0.001, format="%.4f")
    target_settling = st.number_input("Target settling time (s)", value=3.0, min_value=0.1, format="%.2f")
    target_overshoot = st.number_input("Target overshoot (%)", value=10.0, min_value=0.0, format="%.2f")
    target_control_effort = st.number_input("Target control effort", value=50.0, min_value=0.1, format="%.2f")

st.divider()

# Weights
st.header("⚖️ Optimization Weights")
col_w1, col_w2, col_w3, col_w4 = st.columns(4)

with col_w1:
    w_mse = st.number_input("MSE weight", value=1.0, min_value=0.0, max_value=10.0, format="%.2f")
with col_w2:
    w_settling = st.number_input("Settling time weight", value=0.5, min_value=0.0, max_value=10.0, format="%.2f")
with col_w3:
    w_overshoot = st.number_input("Overshoot weight", value=0.5, min_value=0.0, max_value=10.0, format="%.2f")
with col_w4:
    w_control = st.number_input("Control effort weight", value=0.1, min_value=0.0, max_value=10.0, format="%.2f")

st.divider()

# GA Configuration
st.header("🧬 Genetic Algorithm Settings")
col_ga1, col_ga2, col_ga3 = st.columns(3)

with col_ga1:
    ga_population = st.number_input("Population size", value=20, min_value=5, max_value=100)
with col_ga2:
    ga_generations = st.number_input("Generations", value=30, min_value=5, max_value=200)
with col_ga3:
    ga_samples = st.number_input("Evaluation samples per individual", value=10, min_value=1, max_value=50)

seed = st.number_input("Random seed", value=42, min_value=0)

st.divider()

# Run GA button
if st.button("🚀 Run GA Optimization", type="primary", use_container_width=True):
    # Check if we have either a case study or uploaded file
    if case_study_data is None and dynamics_path is None:
        st.error("❌ Please either load a case study or upload a dynamics file first!")
    else:
        try:
            # Load dynamics function
            exec_globals = {}

            if case_study_data:
                # Load from case study JSON
                code = case_study_data['python_code']
                exec(code, exec_globals)
            else:
                # Load from uploaded file
                with open(dynamics_path, 'r') as f:
                    code = f.read()
                exec(code, exec_globals)

            if 'dynamics' not in exec_globals:
                st.error("❌ No 'dynamics' function found!")
            else:
                dynamics = exec_globals['dynamics']

                # Create system configuration
                system_config = {
                    'dt': dt,
                    'max_time': max_time,
                    'target': target,
                    'num_inputs': num_inputs,
                    'input_channel': input_channel,
                    'output_channel': output_channel,
                    'min_ctrl': min_ctrl,
                    'max_ctrl': max_ctrl,
                    'trim_values': trim_values,
                    'num_states': num_states
                }

                # Create simulator
                simulator = SystemSimulator(dynamics, system_config)

                # Create GA config
                ga_config = {
                    'population_size': ga_population,
                    'generations': ga_generations,
                    'num_samples': ga_samples,
                    'seed': seed
                }

                # Create optimizer
                optimizer = GAOptimizer(simulator, ga_config)

                # Define targets and weights
                targets = {
                    'mse': target_mse,
                    'settling_time': target_settling,
                    'overshoot': target_overshoot,
                    'control_effort': target_control_effort
                }

                weights = {
                    'mse': w_mse,
                    'settling_time': w_settling,
                    'overshoot': w_overshoot,
                    'control_effort': w_control
                }

                param_ranges = {
                    'Kp': list(Kp_range),
                    'Ki': list(Ki_range),
                    'Kd': list(Kd_range)
                }

                # Run optimization with progress
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)

                with st.spinner("🧬 Running GA optimization..."):
                    results = optimizer.optimize_pid(targets, weights, param_ranges)
                    progress_bar.progress(1.0)

                # Store results in session state
                st.session_state['results'] = results
                st.session_state['simulator'] = simulator
                st.session_state['targets'] = targets
                st.session_state['weights'] = weights

                st.success("✅ Optimization completed!")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback

            st.code(traceback.format_exc())

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    simulator = st.session_state['simulator']
    targets = st.session_state['targets']
    weights = st.session_state['weights']

    st.divider()
    st.header("📈 Optimization Results")

    if results['success']:
        # Display best parameters
        st.subheader("🎯 Best PID Controller")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

        with col_r1:
            st.metric("Kp", f"{results['controller_parameters']['Kp']:.4f}")
        with col_r2:
            st.metric("Ki", f"{results['controller_parameters']['Ki']:.4f}")
        with col_r3:
            st.metric("Kd", f"{results['controller_parameters']['Kd']:.4f}")
        with col_r4:
            st.metric("Final Cost", f"{results['cost']:.4f}")

        # Display achieved metrics
        st.subheader("📊 Achieved Metrics vs Targets")

        metrics_data = []
        for metric_name in ['mse', 'settling_time', 'overshoot', 'control_effort']:
            achieved = results['achieved_metrics'][metric_name]
            target_val = targets[metric_name]
            weight = weights[metric_name]
            deviation = abs((achieved - target_val) / max(target_val, 1e-6)) * 100

            metrics_data.append({
                'Metric': metric_name,
                'Target': f"{target_val:.4f}",
                'Achieved': f"{achieved:.4f}",
                'Weight': f"{weight:.2f}",
                'Deviation (%)': f"{deviation:.2f}"
            })

        st.table(metrics_data)

        # Plot GA progress
        if 'progress' in results and results['progress']['iteration']:
            st.subheader("🧬 GA Optimization Progress")

            progress = results['progress']

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Cost Evolution', 'Kp Evolution', 'Ki Evolution', 'Kd Evolution'],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )

            # Cost
            fig.add_trace(
                go.Scatter(x=progress['iteration'], y=progress['best_cost'],
                           mode='lines+markers', name='Cost',
                           line=dict(color='red', width=2),
                           marker=dict(size=4)),
                row=1, col=1
            )

            # Kp
            fig.add_trace(
                go.Scatter(x=progress['iteration'], y=progress['Kp'],
                           mode='lines+markers', name='Kp',
                           line=dict(color='blue', width=2),
                           marker=dict(size=4)),
                row=1, col=2
            )

            # Ki
            fig.add_trace(
                go.Scatter(x=progress['iteration'], y=progress['Ki'],
                           mode='lines+markers', name='Ki',
                           line=dict(color='green', width=2),
                           marker=dict(size=4)),
                row=2, col=1
            )

            # Kd
            fig.add_trace(
                go.Scatter(x=progress['iteration'], y=progress['Kd'],
                           mode='lines+markers', name='Kd',
                           line=dict(color='orange', width=2),
                           marker=dict(size=4)),
                row=2, col=2
            )

            fig.update_xaxes(title_text="Generation", row=1, col=1)
            fig.update_xaxes(title_text="Generation", row=1, col=2)
            fig.update_xaxes(title_text="Generation", row=2, col=1)
            fig.update_xaxes(title_text="Generation", row=2, col=2)

            fig.update_yaxes(title_text="Cost", row=1, col=1)
            fig.update_yaxes(title_text="Kp", row=1, col=2)
            fig.update_yaxes(title_text="Ki", row=2, col=1)
            fig.update_yaxes(title_text="Kd", row=2, col=2)

            fig.update_layout(height=600, showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

        # Simulate and plot trajectory
        st.subheader("📉 Closed-Loop Simulation")

        num_test_runs = st.slider("Number of test simulations", 1, 50, 10)

        if st.button("▶️ Run Test Simulations", use_container_width=True):
            with st.spinner(f"Running {num_test_runs} simulations..."):
                all_times = []
                all_trajectories = []
                all_controls = []

                for _ in range(num_test_runs):
                    traj_result = simulator.simulate_and_return_trajectory(
                        results['controller_parameters']
                    )
                    if traj_result['success']:
                        all_times.append(traj_result['time'])
                        all_trajectories.append(traj_result['trajectory'])
                        all_controls.append(traj_result['control_signals'])

                if all_trajectories:
                    # Find max length
                    max_len = max(len(t) for t in all_trajectories)
                    time_vec = all_times[0][:max_len]

                    # Create matrices for statistics
                    traj_matrix = np.full((len(all_trajectories), max_len), np.nan)
                    ctrl_matrix = np.full((len(all_controls), max_len), np.nan)

                    for i, (traj, ctrl) in enumerate(zip(all_trajectories, all_controls)):
                        traj_matrix[i, :len(traj)] = traj
                        ctrl_matrix[i, :len(ctrl)] = ctrl

                    # Statistics
                    traj_mean = np.nanmean(traj_matrix, axis=0)
                    traj_std = np.nanstd(traj_matrix, axis=0)
                    ctrl_mean = np.nanmean(ctrl_matrix, axis=0)
                    ctrl_std = np.nanstd(ctrl_matrix, axis=0)

                    # Plot
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=['System Output (mean ± 1σ)', 'Control Input (mean ± 1σ)'],
                        vertical_spacing=0.12
                    )

                    # Output
                    fig.add_trace(
                        go.Scatter(x=time_vec, y=traj_mean, mode='lines',
                                   name='Mean Output',
                                   line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_vec, y=traj_mean + traj_std,
                                   mode='lines', line=dict(width=0),
                                   showlegend=False),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_vec, y=traj_mean - traj_std,
                                   mode='lines', fill='tonexty',
                                   name='±1σ', line=dict(width=0),
                                   fillcolor='rgba(0,100,200,0.2)'),
                        row=1, col=1
                    )
                    fig.add_hline(y=simulator.target, line_dash="dash",
                                  line_color="green", row=1, col=1,
                                  annotation_text="Target")

                    # Control
                    fig.add_trace(
                        go.Scatter(x=time_vec, y=ctrl_mean, mode='lines',
                                   name='Mean Control',
                                   line=dict(color='orange', width=2)),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_vec, y=ctrl_mean + ctrl_std,
                                   mode='lines', line=dict(width=0),
                                   showlegend=False),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_vec, y=ctrl_mean - ctrl_std,
                                   mode='lines', fill='tonexty',
                                   name='±1σ', line=dict(width=0),
                                   fillcolor='rgba(255,140,0,0.2)'),
                        row=2, col=1
                    )

                    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                    fig.update_yaxes(title_text="Output", row=1, col=1)
                    fig.update_yaxes(title_text="Control", row=2, col=1)

                    fig.update_layout(height=700, showlegend=True)

                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"✅ Successfully simulated {len(all_trajectories)}/{num_test_runs} runs")
                else:
                    st.error("❌ All test simulations failed!")

        # Export controller
        st.divider()
        st.subheader("💾 Export Controller")

        export_format = st.radio("Export format:", ["Python dict", "JSON", "CSV"], horizontal=True)

        controller_params = results['controller_parameters']

        if export_format == "Python dict":
            export_str = str(controller_params)
        elif export_format == "JSON":
            import json

            export_str = json.dumps(controller_params, indent=2)
        else:  # CSV
            export_str = "Parameter,Value\n" + "\n".join(
                [f"{k},{v}" for k, v in controller_params.items()])

        st.code(export_str,
                language="python" if export_format == "Python dict" else "json" if export_format == "JSON" else "csv")

        st.download_button(
            "📥 Download Controller",
            export_str,
            file_name=f"pid_controller.{'py' if export_format == 'Python dict' else 'json' if export_format == 'JSON' else 'csv'}",
            use_container_width=True
        )
    else:
        st.error("❌ Optimization failed!")
        if 'warnings' in results:
            for warning in results['warnings']:
                st.warning(f"⚠️ {warning}")

# Cleanup temp file
if dynamics_path and os.path.exists(dynamics_path):
    try:
        os.unlink(dynamics_path)
    except:
        pass