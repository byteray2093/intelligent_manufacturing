"""Real B–G integration regressions; no pre-recorded success metrics."""
from pathlib import Path
import numpy as np
import pytest
from src.kinematics.fk import fk_flange, fk_tcp
from src.kinematics.ik import get_fk, ik, ik_dls_limited, model
from src.analysis.mujoco_backend import UR5eModel
from src.planning.io import load_e_trajectory, split_phases
from src.control.controller import PickPlaceController

ROOT = Path(__file__).resolve().parents[1]


def test_independent_fk_uses_explicit_tool_frame():
    backend = UR5eModel()
    for q in np.random.default_rng(42).uniform(-1.5,1.5,(20,6)):
        assert np.linalg.norm(fk_flange(q)[:3,3]-backend.fk_flange(q)[:3,3]) < .002
        assert np.linalg.norm(fk_tcp(q)[:3,3]-backend.fk_tcp(q)[:3,3]) < .002
        assert np.allclose(fk_tcp(q)[:3,:3],backend.fk_tcp(q)[:3,:3],atol=1e-12)


def test_dls_roundtrip_and_unreachable_residual():
    home=model.key_qpos[0,:6]
    target=get_fk(home+np.array([.1,.1,-.1,.05,0,0]))
    result=ik(target,home)
    assert result.success and result.position_error_m < 1e-4
    target[:3,3]=[3,3,3]
    q,ok,info=ik_dls_limited(target,home,max_iter=3)
    assert not ok
    assert np.isclose(info['pos_err'],np.linalg.norm(get_fk(q)[:3,3]-target[:3,3]))
    assert np.all(q>=model.jnt_range[:6,0]) and np.all(q<=model.jnt_range[:6,1])


def test_mujoco_jacobian_against_independent_finite_difference():
    backend=UR5eModel();q=backend.home_q();eps=1e-5
    J=backend.jacobian_tcp(q)
    numeric=np.column_stack([(get_fk(q+np.eye(6)[i]*eps)[:3,3]-get_fk(q-np.eye(6)[i]*eps)[:3,3])/(2*eps) for i in range(6)])
    assert np.max(abs(J[:3]-numeric))<1e-8
    assert np.linalg.svd(backend.jacobian_tcp(np.zeros(6)),compute_uv=False)[-1]<1e-10


def test_phase_loader_preserves_derivatives_and_all_seven_phases():
    path=ROOT/'data/pick_place_traj.npz'
    tr=load_e_trajectory(path);phases=split_phases(tr)
    with np.load(path) as z:assert np.array_equal(tr['ddq_traj'],z['ddq_traj'])
    assert len(phases)==7
    assert sum(p['duration'] for p in phases)==pytest.approx(tr['duration'])
    for a,b in zip(phases,phases[1:]):assert np.array_equal(a['q_traj'][-1],b['q_traj'][0])
    csv=load_e_trajectory(path.with_suffix('.csv'))
    assert np.allclose(csv['q_traj'],tr['q_traj'],atol=1e-6)


def test_invalid_time_is_rejected(tmp_path):
    path=tmp_path/'bad.npz'
    np.savez(path,time=[0,.01,.01],q_traj=np.zeros((3,6)))
    with pytest.raises(ValueError):load_e_trajectory(path)


def test_three_real_pick_place_cycles_and_synchronized_logger():
    ctrl=PickPlaceController(verbose=False)
    out=ctrl.run(cycles=3,record_video=False,traj_file=str(ROOT/'data/pick_place_traj.npz'))
    assert out['summary']['all_consecutive_success']
    assert all(c['used_e_traj'] for c in out['cycles'])
    assert all(c['place_dxy_mm']<2 for c in out['cycles'])
    times=np.array([r['t'] for r in out['log_rows']]);assert np.allclose(np.diff(times),ctrl.dt)
    for row in out['log_rows'][::200]:
        assert np.linalg.norm(get_fk(row['q_actual'])[:3,3]-row['ee_pos'])<2e-6
    assert 'transfer_e' not in {r['segment'] for r in out['log_rows']}
    # No 0.4 rad takeover jump: every commanded transition stays smooth.
    cmd=np.array([r['q_cmd'] for r in out['log_rows']])
    assert np.max(abs(np.diff(cmd,axis=0)))<.01


def test_execution_obeys_nonuniform_time_and_n_minus_one_intervals():
    ctrl=PickPlaceController(verbose=False);ctrl.reset_to_home()
    t=np.array([0,.001,.004,.01])
    q=np.tile(ctrl.home_q,(len(t),1));q[:,0]+=.1*t
    ctrl._run_traj(dict(time=t,q_traj=q),'nonuniform')
    assert ctrl.data.time==pytest.approx(.01)
    assert len(ctrl.log_rows)==5


def test_actual_command_derivatives_respect_bounds():
    tr=load_e_trajectory(ROOT/'data/pick_place_traj.npz')
    v=np.gradient(tr['q_traj'],tr['time'],axis=0)
    a=np.gradient(v,tr['time'],axis=0)
    assert np.max(abs(v))<=1.001
    assert np.max(abs(a))<=3.001
