import logging
import time
import pygad
import numpy as np
from typing import Dict, Any, List, Optional

from src.simulator import SystemSimulator
from src.logger import get_logger
from src.callbacks import get_callback

logger = get_logger(__name__)


class GAOptimizer:
    """Genetic Algorithm optimizer for controller tuning."""

    def __init__(self, simulator: SystemSimulator, config: Dict[str, Any]):
        self.simulator       = simulator
        self.population_size = config['population_size']
        self.generations     = config['generations']
        self.seed            = config.get('seed', None)

        # -- Shared-experiment timing
        # experiment_start_time: time.time() stamp from the very beginning of
        # the whole comparison / LLM-GA workflow run (set by the caller).
        # Using a shared timestamp lets cumulative_wall_time be consistent
        # across multiple GA attempts within the same experiment.
        self.experiment_start_time: Optional[float] = config.get('experiment_start_time', None)

        # Wall-clock budget for the ENTIRE experiment (seconds).
        # GA stops if time.time() - experiment_start_time >= max_wall_clock.
        self.max_wall_clock: float = config.get('max_wall_clock', float('inf'))

        # NFE already accumulated by previous GA runs in this experiment
        # (LLM-GA multi-attempt case).
        # cumulative_nfe = nfe_offset + evaluations_in_this_run.
        self.nfe_offset: int = config.get('nfe_offset', 0)
        self.num_evaluations: int = 0
        self.current_attempt: int = config.get('current_attempt', 1)  # ← ADD

        # Counter reset at the start of each optimize_pid() call.
        self.num_evaluations: int = 0

    # =========================================================================

    def compute_baseline_cost(
            self,
            metrics: Dict[str, float],
            fixed_targets: Dict[str, float],
    ) -> float:
        """
        Baseline cost: sum of relative deviations from fixed targets.
        Formula: sum_i (achieved_i - target_i) / target_i
        A value <= 0 means all targets met or bettered.
        """
        cost = 0.0
        for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
            if metric in metrics and metric in fixed_targets:
                achieved = metrics[metric]
                target   = fixed_targets[metric]
                cost += (achieved - target) / max(target, 1e-6)
        return cost

    def _compute_success_score(
            self,
            metrics: Dict[str, float],
            fixed_targets: Dict[str, float],
    ) -> int:
        """25 points per metric that meets or beats its fixed target (max 100)."""
        score = 0
        for metric in ['mse', 'settling_time', 'overshoot', 'control_effort']:
            if metric in metrics and metric in fixed_targets:
                if metrics[metric] <= fixed_targets[metric]:
                    score += 25
        return score

    # =========================================================================

    def optimize_pid(
            self,
            weights: Dict[str, float],
            param_ranges: Dict[str, List[float]],
            fixed_targets: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Optimize PID gains using GA.

        Termination (whichever fires first)
        ------------------------------------
        1. All ``generations`` completed  -- normal PyGAD finish.
        2. Wall-clock budget exceeded     -- checked at end of every generation:
               time.time() - experiment_start_time >= max_wall_clock
        3. Score = 100/100               -- all four fixed targets met.

        Per-generation records in ``progress``
        ---------------------------------------
        cumulative_nfe        : total NFE from experiment start (nfe_offset + this run)
        cumulative_wall_time  : wall-clock seconds from experiment start
        best_baseline_so_far  : running minimum of best_baseline_cost
        best_baseline_cost    : best solution's baseline cost this generation
        mean_baseline_cost    : population mean baseline cost
        (all legacy fields retained for backward compatibility)

        Returned extras
        ---------------
        stop_reason           : 'normal' | 'wall_clock' | 'score_100'
        target_hit_nfe        : cumulative NFE when score first = 100 (or None)
        target_hit_time       : wall-clock seconds when score first = 100 (or None)
        target_hit_generation : generation index when score first = 100 (or None)
        """
        self.weights       = weights
        self.fixed_targets = fixed_targets
        self.num_evaluations = 0

        # Fall back: if the caller never set experiment_start_time, start now.
        if self.experiment_start_time is None:
            self.experiment_start_time = time.time()

        # -- Fitness function--------
        def fitness_func(ga_instance, solution, solution_idx):
            self.num_evaluations += 1

            Kp, Ki, Kd = solution
            result = self.simulator.evaluate_pid({'Kp': Kp, 'Ki': Ki, 'Kd': Kd})
            if not result['success']:
                return -10000
            m = result['metrics']
            cost = sum(
                self.weights.get(met, 1.0) * (m[met] ** 2)
                for met in ['mse', 'settling_time', 'overshoot', 'control_effort']
                if met in m
            )
            return -cost

        # -- Progress dict-----------
        progress = {
            # Legacy fields
            'iteration': [],
            'best_cost': [],    # GA fitness (best solution)
            'mean_cost': [],    # GA fitness (population mean)
            'best_params': [],
            'Kp': [], 'Ki': [], 'Kd': [],
            'mse': [], 'settling_time': [], 'overshoot': [], 'control_effort': [],
            'metrics_success':    [],
            'warnings':           [],
            'best_baseline_cost': [],    # best solution, this generation
            'mean_baseline_cost': [],    # population mean, this generation
            'cumulative_nfe':       [],  # nfe_offset + num_evaluations
            'cumulative_wall_time': [],  # seconds from experiment_start_time
            'best_baseline_so_far': [],  # running min of best_baseline_cost
            # Add GA-named aliases expected by MATLAB
            'best_ga_cost': [],
            'mean_ga_cost': [],
            'success_score': [],  # 0/25/50/75/100 per generation
            'best_score_so_far': [],  # running max of success_score
        }

        _running_best_score = [0]  # companion to _running_best

        _running_best = [float('inf')]

        _target_hit = {'nfe': None, 'time': None, 'generation': None}

        _stop_reason = ['normal']

        # -- on_generation callback--
        def on_generation(ga_instance):
            now = time.time()
            elapsed = now - self.experiment_start_time
            cum_nfe = self.nfe_offset + self.num_evaluations
            gen_num = ga_instance.generations_completed
            # logger.info(f"  [DEBUG] on_generation START | gen={gen_num}")

            # Best solution this generation
            # solution, fitness, _ = ga_instance.best_solution()
            solution, fitness, _ = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness
            )
            # logger.info(f"  [DEBUG] on_generation END | gen={gen_num}")

            # Evaluate the best solution
            best_params = {'Kp': solution[0], 'Ki': solution[1], 'Kd': solution[2]}
            best_eval = self.simulator.evaluate_pid(best_params)

            # Calculate baseline costs for entire population
            pop = ga_instance.population
            gen_baseline_costs = []
            for sol in pop:
                params = {'Kp': sol[0], 'Ki': sol[1], 'Kd': sol[2]}
                result = self.simulator.evaluate_pid(params)
                if result['success']:
                    bc = self.compute_baseline_cost(result['metrics'], self.fixed_targets)
                    gen_baseline_costs.append(bc)
                else:
                    gen_baseline_costs.append(np.nan)

            if best_eval['success']:
                m = best_eval['metrics']
                bc = self.compute_baseline_cost(m, self.fixed_targets)
                scr = self._compute_success_score(m, self.fixed_targets)
                progress['mse'].append(m.get('mse'))
                progress['settling_time'].append(m.get('settling_time'))
                progress['overshoot'].append(m.get('overshoot'))
                progress['control_effort'].append(m.get('control_effort'))
                progress['best_baseline_cost'].append(bc)
                progress['metrics_success'].append(True)
                progress['warnings'].append(best_eval.get('warnings', []))
                progress['success_score'].append(scr)
                if scr > _running_best_score[0]:
                    _running_best_score[0] = scr
                progress['best_score_so_far'].append(_running_best_score[0])
            else:
                bc = float('inf')
                scr = 0
                progress['mse'].append(np.nan)
                progress['settling_time'].append(np.nan)
                progress['overshoot'].append(np.nan)
                progress['control_effort'].append(np.nan)
                progress['best_baseline_cost'].append(np.nan)
                progress['metrics_success'].append(False)
                progress['warnings'].append(best_eval.get('warnings', []))

            # Population mean baseline cost
            pop = ga_instance.population
            gen_bl = []
            for sol in pop:
                r = self.simulator.evaluate_pid(
                    {'Kp': sol[0], 'Ki': sol[1], 'Kd': sol[2]}
                )
                if r['success']:
                    gen_bl.append(
                        self.compute_baseline_cost(r['metrics'], self.fixed_targets)
                    )
                else:
                    gen_bl.append(np.nan)
            progress['mean_baseline_cost'].append(
                np.nanmean(gen_bl) if gen_bl else np.nan
            )

            # GA fitness cost (population mean)
            pop_fitness = ga_instance.last_generation_fitness
            progress['mean_cost'].append(-np.mean(pop_fitness))

            # Legacy
            progress['iteration'].append(gen_num)
            progress['best_cost'].append(-fitness)
            progress['best_ga_cost'].append(-fitness)
            progress['mean_ga_cost'].append(-np.mean(pop_fitness))
            progress['best_params'].append(solution.tolist())
            progress['Kp'].append(solution[0])
            progress['Ki'].append(solution[1])
            progress['Kd'].append(solution[2])

            # New rich tracking
            progress['cumulative_nfe'].append(cum_nfe)
            progress['cumulative_wall_time'].append(elapsed)

            if bc < _running_best[0]:
                _running_best[0] = bc
            progress['best_baseline_so_far'].append(_running_best[0])

            # Target-hit detection
            if scr == 100 and _target_hit['nfe'] is None:
                _target_hit['nfe'] = cum_nfe
                _target_hit['time'] = elapsed
                _target_hit['generation'] = gen_num
                logger.info(
                    f"  TARGET HIT at gen {gen_num} | "
                    f"NFE={cum_nfe} | wall={elapsed:.2f}s"
                )

            logger.debug(
                f"Gen {gen_num}: best_bl={bc:.4f} best_so_far={_running_best[0]:.4f} "
                f"score={scr}/100 NFE={cum_nfe} wall={elapsed:.2f}s"
            )

            # -- Stream progress to Streamlit (if registered) --------------
            try:
                # from src.callbacks import get_callback
                _cb = get_callback()
                if _cb is not None:
                    _cb({
                        'event_type': 'generation',
                        'attempt': self.current_attempt,
                        'generation': gen_num,
                        'cumulative_nfe': cum_nfe,
                        'cumulative_wall_time': elapsed,
                        'best_baseline_so_far': _running_best[0],
                        'best_baseline_cost': bc if bc != float('inf') else None,
                        'mse': progress['mse'][-1] if progress['mse'] else None,
                        'settling_time': progress['settling_time'][-1] if progress['settling_time'] else None,
                        'overshoot': progress['overshoot'][-1] if progress['overshoot'] else None,
                        'control_effort': progress['control_effort'][-1] if progress['control_effort'] else None,
                        'Kp': float(solution[0]),
                        'Ki': float(solution[1]),
                        'Kd': float(solution[2]),
                        'success_score': scr,
                        'best_score_so_far': _running_best_score[0],
                        'param_ranges': {
                            'Kp': [param_ranges['Kp'][0], param_ranges['Kp'][1]],
                            'Ki': [param_ranges['Ki'][0], param_ranges['Ki'][1]],
                            'Kd': [param_ranges['Kd'][0], param_ranges['Kd'][1]],
                        },
                        'weights': dict(self.weights),
                        'pop_size': self.population_size,
                        'num_gen': self.generations,
                    })
            except Exception:
                pass

            # -- Early termination---
            if scr == 100:
                _stop_reason[0] = 'score_100'
                logger.info(
                    f"  STOP (score=100) gen={gen_num} NFE={cum_nfe} wall={elapsed:.2f}s"
                )
                raise StopIteration

            if elapsed >= self.max_wall_clock:
                _stop_reason[0] = 'wall_clock'
                logger.info(
                    f"  STOP (wall_clock {elapsed:.1f}s >= {self.max_wall_clock}s) "
                    f"gen={gen_num} NFE={cum_nfe}"
                )
                raise StopIteration

        # -- Build and run PyGAD-----
        ga_kwargs = dict(
            num_generations=self.generations,
            num_parents_mating=max(2, self.population_size // 5),
            fitness_func=fitness_func,
            sol_per_pop=self.population_size,
            num_genes=3,
            gene_space=[
                {'low': param_ranges['Kp'][0], 'high': param_ranges['Kp'][1]},
                {'low': param_ranges['Ki'][0], 'high': param_ranges['Ki'][1]},
                {'low': param_ranges['Kd'][0], 'high': param_ranges['Kd'][1]},
            ],
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_num_genes=1,
            on_generation=on_generation,
            suppress_warnings=True,
        )

        # Only pass random_seed if it's not None (allow stochastic runs)
        if self.seed is not None:
            ga_kwargs['random_seed'] = self.seed

        # Initialize PyGAD
        ga_instance = pygad.GA(**ga_kwargs)

        _pygad_logger = logging.getLogger('pygad.pygad')
        _prev_level = _pygad_logger.level
        _pygad_logger.setLevel(logging.CRITICAL)
        try:
            ga_instance.run()
        except StopIteration:
            logger.info(
                f"  GA halted early: {_stop_reason[0]} "
                f"({self.num_evaluations} evals in this run)"
            )
        finally:
            _pygad_logger.setLevel(_prev_level)  # always restore

        # Get best solution
        solution, fitness, _ = ga_instance.best_solution()
        Kp_opt, Ki_opt, Kd_opt = solution

        # Evaluate final controller
        params = {'Kp': Kp_opt, 'Ki': Ki_opt, 'Kd': Kd_opt}
        final_result = self.simulator.evaluate_pid(params)

        return {
            'success': final_result['success'],
            'controller_parameters': params,
            'achieved_metrics': final_result.get('metrics', {}),
            'ga_cost': -fitness,
            'baseline_cost': (
                self.compute_baseline_cost(final_result['metrics'], self.fixed_targets)
                if final_result['success'] else float('inf')
            ),
            'warnings': final_result.get('warnings', []),
            'progress': progress,
            'num_evaluations': self.num_evaluations,
            # Termination metadata
            'stop_reason': _stop_reason[0],
            'target_hit_nfe': _target_hit['nfe'],
            'target_hit_time': _target_hit['time'],
            'target_hit_generation': _target_hit['generation'],
        }