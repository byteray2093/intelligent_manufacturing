"""Trajectory validation, persistence and E→F phase adapter (no second planner)."""
from pathlib import Path
import numpy as np
from src.contracts import JointTrajectory
from src.planning.trajectory import QuinticJointSegment, CartesianLineSegment
from src.kinematics.ik import get_fk

PHASE_NAMES = ['approach', 'descend', 'lift', 'transfer', 'lower', 'retreat', 'return_home']


def _pack(t, q, dq=None, ddq=None):
    tr = JointTrajectory(t, q)
    t, q = tr.time_s, tr.q_rad
    if dq is None:
        dq = np.gradient(q, t, axis=0)
    if ddq is None:
        ddq = np.gradient(dq, t, axis=0)
    for name, arr in [('dq', dq), ('ddq', ddq)]:
        if np.shape(arr) != q.shape or not np.all(np.isfinite(arr)):
            raise ValueError(f'{name} must be finite Nx6')
    return dict(time=t, q_traj=q, dq_traj=np.asarray(dq), ddq_traj=np.asarray(ddq),
                dt=float(np.median(np.diff(t))), duration=float(t[-1]), kind='e_module')


def load_e_trajectory(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            tr = _pack(z['time'], z['q_traj'], z.get('dq_traj'), z.get('ddq_traj'))
            for key in ['segment_starts', 'segment_durations', 'segment_names']:
                if key in z:
                    tr[key] = z[key].copy()
            tr['scope'] = str(z.get('trajectory_scope', 'unspecified'))
    elif path.suffix == '.csv':
        rows = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8-sig')
        names = rows.dtype.names
        cols = ([f'q{i}' for i in range(1, 7)] if 'q6' in names
                else [f'q{i}' for i in range(6)])
        tcol = 't' if 't' in names else 'time'
        tr = _pack(rows[tcol], np.column_stack([rows[c] for c in cols]))
        tr['scope'] = 'unspecified'
    else:
        raise ValueError('Use .npz with phase metadata for full execution, or .csv for analysis')
    tr['source'] = str(path)
    return tr


def split_phases(tr):
    starts, durations = tr['segment_starts'], tr['segment_durations']
    if len(starts) != 7 or len(durations) != 7:
        raise ValueError('Full pick/place requires seven phases')
    if not np.allclose(starts, np.r_[0, np.cumsum(durations)[:-1]], atol=1e-8):
        raise ValueError('Non-contiguous phase times')
    if not np.isclose(sum(durations), tr['time'][-1], atol=1e-8):
        raise ValueError('Phase times do not cover the trajectory')
    phases = []
    for i, (start, duration) in enumerate(zip(starts, durations)):
        lo = int(np.argmin(abs(tr['time'] - start)))
        hi = int(np.argmin(abs(tr['time'] - start - duration)))
        phase = _pack(tr['time'][lo:hi+1] - start, tr['q_traj'][lo:hi+1],
                      tr['dq_traj'][lo:hi+1], tr['ddq_traj'][lo:hi+1])
        phase['name'] = PHASE_NAMES[i]
        phases.append(phase)
    return phases


def quintic_joint_traj(q0, q1, duration, dt):
    seg = QuinticJointSegment(q0, q1, duration, dt)
    return _pack(seg.t, seg.q, seg.dq, seg.ddq)


def cartesian_line_traj(T_start, T_goal, duration, dt, ik_fn, q_seed,
                        pos_tol=1e-3, fail_fast=True):
    seg = CartesianLineSegment(get_fk, ik_fn, T_start, T_goal, duration, dt,
                               q_start=q_seed)
    return dict(_pack(seg.t, seg.q, seg.dq, seg.ddq), success=seg.success,
                ik_failures=[], tcp_traj=seg.pos)
