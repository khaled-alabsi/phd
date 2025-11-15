import numpy as np
import pandas as pd
from typing import Tuple

# Fault injection configuration (multi-variable, type-controlled)
FAULT_CONFIG: dict[int, list[dict[str, object]]] = {
    1: [  # One fault only
        {'var': 'welding_temp', 'type': 'trend', 'slope': 0.08}
    ],
    2: [  # 2 faults
        {'var': 'clamp_pressure', 'type': 'drift', 'end': 3},
        {'var': 'cycle_time', 'type': 'spike', 'prob': 0.02, 'scale': 3}
    ],
    3: [  # 3 faults
        {'var': 'robot_torque', 'type': 'shift', 'value': 20},
        {'var': 'weld_time', 'type': 'flat', 'start': 100, 'length': 50},
        {'var': 'robot_current', 'type': 'cos'}
    ],
    4: [  # 4 faults
        {'var': 'paint_viscosity', 'type': 'drop', 'magnitude': -0.4, 'at': 100},
        {'var': 'conveyor_speed', 'type': 'invert'},
        {'var': 'robot_current', 'type': 'noise', 'scale': 4},
        {'var': 'welding_temp', 'type': 'trend', 'slope': 0.05}
    ],
    5: [  # 5 faults
        {'var': 'ambient_temp', 'type': 'sin', 'amp': 2},
        {'var': 'humidity', 'type': 'trend', 'slope': -0.1},
        {'var': 'sensor_drift', 'type': 'drift', 'end': 0.4},
        {'var': 'paint_viscosity', 'type': 'shift', 'value': 0.3},
        {'var': 'cycle_time', 'type': 'spike', 'prob': 0.04, 'scale': 2}
    ],
    6: [  # 6 faults
        {'type': 'correlated_shift', 'vars': ['robot_torque', 'robot_current'], 'value': 10},
        {'var': 'robot_torque', 'type': 'trend', 'slope': 0.3},
        {'var': 'weld_time', 'type': 'flat', 'start': 50, 'length': 60},
        {'var': 'conveyor_speed', 'type': 'drift', 'end': 0.2},
        {'var': 'ambient_temp', 'type': 'shift', 'value': 5},
        {'var': 'clamp_pressure', 'type': 'noise', 'scale': 1}
    ],
    7: [{'var': 'quality_score', 'type': 'cos'},
        {'var': 'welding_temp', 'type': 'trend', 'slope': -0.1}],
    8: [{'var': 'cycle_time', 'type': 'invert'},
        {'var': 'sensor_drift', 'type': 'drift', 'end': -0.5},
        {'var': 'robot_torque', 'type': 'spike', 'prob': 0.03, 'scale': 15}],
    9: [{'var': 'paint_viscosity', 'type': 'shift', 'value': 0.5},
        {'var': 'robot_current', 'type': 'trend', 'slope': 0.1},
        {'var': 'welding_temp', 'type': 'drift', 'end': 10},
        {'var': 'ambient_temp', 'type': 'flat', 'start': 70, 'length': 30}],
    10: [{'type': 'correlated_shift', 'vars': ['robot_torque', 'weld_time'], 'value': 12}],
    11: [{'var': 'sensor_drift', 'type': 'trend', 'slope': 0.02},
         {'var': 'cycle_time', 'type': 'spike', 'prob': 0.05, 'scale': 5}],
    12: [{'var': 'ambient_temp', 'type': 'sin', 'amp': 1.5},
         {'var': 'clamp_pressure', 'type': 'drop', 'magnitude': -1.5, 'at': 120}],
    13: [{'var': 'humidity', 'type': 'drift', 'end': 8},
         {'var': 'quality_score', 'type': 'trend', 'slope': -0.02},
         {'var': 'robot_current', 'type': 'noise', 'scale': 2}],
    14: [{'var': 'robot_torque', 'type': 'shift', 'value': 25},
         {'var': 'welding_temp', 'type': 'trend', 'slope': 0.1},
         {'var': 'robot_current', 'type': 'stuck'}],
    15: [{'var': 'cycle_time', 'type': 'invert'},
         {'var': 'humidity', 'type': 'flat', 'start': 90, 'length': 50},
         {'var': 'spray_pressure', 'type': 'noise', 'scale': 3}],
    16: [{'type': 'correlated_shift', 'vars': ['ambient_temp', 'humidity'], 'value': 5}],
    17: [{'var': 'robot_current', 'type': 'drift', 'end': -2},
         {'var': 'robot_torque', 'type': 'trend', 'slope': -0.2},
         {'var': 'clamp_pressure', 'type': 'shift', 'value': 1}],
    18: [{'var': 'conveyor_speed', 'type': 'drop', 'magnitude': -0.5, 'at': 150}],
    19: [{'var': 'quality_score', 'type': 'cos'},
         {'var': 'robot_torque', 'type': 'shift', 'value': -10}],
    20: [{'type': 'correlated_shift', 'vars': ['welding_temp', 'spray_pressure'], 'value': 8},
         {'var': 'paint_viscosity', 'type': 'sin', 'amp': 2},
         {'var': 'robot_current', 'type': 'noise', 'scale': 2}],
}


PROCESS_CONFIG: dict[str, dict[str, object]] = {
    'welding_temp':        {'mean': 350, 'std': 10, 'dist': 'normal'},
    'clamp_pressure':      {'mean': 5.0, 'std': 0.5, 'dist': 'normal'},
    'robot_torque':        {'mean': 120, 'std': 15, 'dist': 'normal'},
    'paint_viscosity':     {'mean': 0.85, 'std': 0.1, 'dist': 'lognormal'},
    'conveyor_speed':      {'mean': 1.5, 'std': 0.2, 'dist': 'normal'},
    'ambient_temp':        {'mean': 25, 'std': 2, 'dist': 'normal'},
    'humidity':            {'dist': 'uniform', 'low': 30, 'high': 60},
    'sensor_drift':        {'dist': 'exponential', 'scale': 0.1},
    'cycle_time':          {'dist': 'chi2', 'df': 4},
    'quality_score':       {'dist': 'beta', 'a': 2, 'b': 5},
    'robot_current':       {'formula': 'robot_torque * 0.08 + noise(0.5)'},
    'weld_time':           {'formula': '350 / welding_temp + noise(0.01)'},
    'spray_pressure':      {'formula': 'np.sin(paint_viscosity * 3) * 5 + 10 + noise(0.5)'},
    'dynamic_A':           {'mean': 50, 'std': 5, 'dist': 'normal'},
    'dynamic_B':           {'mean': 100, 'std': 10, 'dist': 'normal'},
    'formula_X':           {'formula': 'np.log1p(cycle_time) * 3 + noise(0.2)'},
    'formula_Y':           {'formula': 'np.sqrt(ambient_temp) * 2 + noise(0.2)'},
}

def noise(std: float, size: int) -> np.ndarray:
    return np.random.normal(0, std, size=size)

def simulate_normal_data(n_samples: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = pd.DataFrame(index=range(n_samples))

    for var, cfg in PROCESS_CONFIG.items():
        if 'formula' in cfg:
            continue
        dist = cfg['dist']
        if dist == 'normal':
            df[var] = np.random.normal(cfg['mean'], cfg['std'], n_samples)
        elif dist == 'lognormal':
            mu = np.log(cfg['mean']**2 / np.sqrt(cfg['std']**2 + cfg['mean']**2))
            sigma = np.sqrt(np.log(1 + (cfg['std']**2 / cfg['mean']**2)))
            df[var] = np.random.lognormal(mu, sigma, n_samples)
        elif dist == 'uniform':
            df[var] = np.random.uniform(cfg['low'], cfg['high'], n_samples)
        elif dist == 'exponential':
            df[var] = np.random.exponential(cfg['scale'], n_samples)
        elif dist == 'chi2':
            df[var] = np.random.chisquare(cfg['df'], n_samples)
        elif dist == 'beta':
            df[var] = np.random.beta(cfg['a'], cfg['b'], n_samples)

    for var, cfg in PROCESS_CONFIG.items():
        if 'formula' not in cfg:
            continue
        context = df.to_dict(orient='series')
        df[var] = eval(cfg['formula'], {'np': np, 'noise': lambda std: noise(std, n_samples)}, context)

    dynamic_start = n_samples // 3
    dynamic_length = n_samples // 3
    drift_A = np.linspace(0, 10, dynamic_length)
    drift_B = np.linspace(0, 20, dynamic_length)
    df.loc[dynamic_start:dynamic_start + dynamic_length - 1, 'dynamic_A'] += drift_A
    df.loc[dynamic_start:dynamic_start + dynamic_length - 1, 'dynamic_B'] += drift_B

    random_start = n_samples // 2
    random_length = n_samples // 4
    mask = np.zeros(n_samples, dtype=bool)
    mask[random_start:random_start + random_length] = True
    np.random.shuffle(mask)
    df.loc[mask, 'formula_X'] += np.random.normal(3, 1, size=mask.sum())
    df.loc[mask, 'formula_Y'] += np.random.normal(-2, 1, size=mask.sum())

    df['faultNumber'] = 0
    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    for col in ['faultNumber', 'simulationRun', 'sample']:
        if col in cols:
            cols.remove(col)
    return df[['faultNumber', 'simulationRun', 'sample'] + cols]

def generate_dataset(
    n_normal_simulations: int = 20,
    n_samples_per_simulation: int = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normal_runs = []
    faulty_runs = []

    for run_id in range(1, n_normal_simulations + 1):
        df = simulate_normal_data(n_samples_per_simulation, seed=run_id)
        df['simulationRun'] = run_id
        df['sample'] = np.arange(1, n_samples_per_simulation + 1)
        df = reorder_columns(df)
        normal_runs.append(df)

    # For now, simulate faulty with same process (placeholder)
    for run_id in range(1, n_normal_simulations + 1):
        df = simulate_normal_data(n_samples_per_simulation, seed=run_id + 100)
        df['simulationRun'] = run_id
        df['sample'] = np.arange(1, n_samples_per_simulation + 1)
        df = reorder_columns(df)
        faulty_runs.append(df)

    return (
        pd.concat(normal_runs, ignore_index=True),
        pd.concat(faulty_runs, ignore_index=True)
    )

