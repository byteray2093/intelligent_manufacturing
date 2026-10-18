"""Unified project CLI. Run from the repository root: python -m src.main."""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ = ROOT / 'data/pick_place_traj.npz'


def plan():
    from src.planning.trajectory import plan_pick_and_place
    from src.kinematics.ik import model
    home = model.key_qpos[0, :6].copy()
    limits = model.jnt_range[:6].copy()
    tr = plan_pick_and_place(home, dict(up_pos=(-0.35, 0.4), down_pos=(0.35, 0.4)),
                             dt=0.002, joint_limits=limits)
    if not tr.within_limits:
        raise RuntimeError('Planned trajectory violates joint limits')
    if np.max(tr.peak_velocity) > 1.001 or np.max(tr.peak_acceleration) > 3.001:
        raise RuntimeError('Planned velocity/acceleration exceeds declared limits')
    DEFAULT_TRAJ.parent.mkdir(exist_ok=True)
    tr.to_npz(DEFAULT_TRAJ)
    tr.to_csv(DEFAULT_TRAJ.with_suffix('.csv'))
    info = dict(backend=tr.backend, samples=len(tr.t), duration_s=tr.T_total,
                peak_velocity_rad_s=tr.peak_velocity.tolist(),
                peak_acceleration_rad_s2=tr.peak_acceleration.tolist(),
                segment_names=tr.segment_names,
                segment_durations_s=[float(s.t[-1]) for s in tr.segments],
                within_joint_limits=tr.within_limits)
    DEFAULT_TRAJ.with_suffix('.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description='UR5e kinematics, planning and MuJoCo pick/place')
    ap.add_argument('command', choices=['plan', 'demo', 'analyze'], nargs='?', default='demo')
    args, rest = ap.parse_known_args(argv)
    if args.command == 'plan':
        return plan()
    if args.command == 'analyze':
        from src.analysis.integrated import main as analyze
        return analyze()
    from src.control.controller import main as demo
    if '--traj-file' not in rest:
        if not DEFAULT_TRAJ.exists():
            plan()
        rest = ['--traj-file', str(DEFAULT_TRAJ)] + rest
    return demo(rest)


if __name__ == '__main__':
    raise SystemExit(main())
