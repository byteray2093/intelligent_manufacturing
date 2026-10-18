"""Current integrated evidence using G's metrics and model backend."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from src.analysis.mujoco_backend import UR5eModel
from src.analysis import metrics, validation, tracking
from src.analysis.exec_log import load_exec_log
from src.planning.io import load_e_trajectory, split_phases
from src.planning.trajectory import QuinticJointSegment
from src.kinematics.ik import ik_dls_limited, get_fk

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/integrated'


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    model = UR5eModel()
    tr = load_e_trajectory(ROOT / 'data/pick_place_traj.npz')
    phases = split_phases(tr)
    log = load_exec_log(ROOT / 'results/logs/exec_log_latest.csv')
    commanded = model.fk_batch(log.q_cmd)
    actual = model.fk_batch(log.q_actual)
    pe, re = tracking.batch_pose_errors(commanded, actual)
    # SI normalization: L=1 metre; J_norm=[J_position/L; J_rotation].
    L = 1.0
    sigma = np.array([np.linalg.svd(np.diag([1/L]*3+[1]*3) @ model.jacobian_tcp(q),
                                    compute_uv=False)[-1] for q in log.q_actual])
    joint_err = log.q_actual - log.q_cmd
    checks = validation.check_c_fk(model) + validation.check_numeric_jacobian(model) + validation.check_flange_tcp(model)
    # c_fk_tcp deliberately demonstrates the incorrect flange-as-TCP comparison.
    write_csv('kinematics_validation.csv', checks)
    required = [row for row in checks if row['check'] != 'c_fk_tcp']
    if not all(row['passed'] for row in required):
        raise RuntimeError('Kinematics cross validation failed')
    segment_rows = []
    for run in log.segments():
        sl = slice(run.i0,run.i1)
        segment_rows.append(dict(cycle=run.cycle, segment=run.segment, samples=run.n,
            joint_error_max_deg=float(np.degrees(abs(joint_err[sl]).max())),
            tcp_error_max_mm=float(pe[sl].max()*1000),
            sigma_min=float(sigma[sl].min())))
    write_csv('execution_segments.csv', segment_rows)
    # Independent D solver check, including a geometric outside target.
    targets = [get_fk(model.home_q()), get_fk(phases[3]['q_traj'][-1])]
    far = np.eye(4); far[:3,3] = [3, 3, 3]; targets.append(far)
    ik_rows=[]
    for i, target in enumerate(targets):
        q, ok, info = ik_dls_limited(target, model.home_q())
        ik_rows.append(dict(case=['home','transfer_end','outside'][i],success=bool(ok),
            pos_error_mm=float(info['pos_err']*1000), rot_error_rad=float(info['rot_err']),
            iterations=int(info['iterations'])))
    write_csv('ik_cases.csv',ik_rows)
    if not all(row['success'] for row in ik_rows[:2]) or ik_rows[2]['success']:
        raise RuntimeError('IK validation failed')
    # Same endpoints, duration and 1 rad/s / 3 rad/s² bounds for both methods.
    transfer=phases[3]
    q0,q1=transfer['q_traj'][0],transfer['q_traj'][-1]
    joint=QuinticJointSegment(q0,q1,transfer['duration'],tr['dt'])
    common=max(joint.T,transfer['duration'])
    if not np.isclose(common,transfer['duration']):
        raise RuntimeError('Comparison needs resampling to common duration')
    comparisons=[]; poses=[]
    for name,q,dq,ddq in [('joint_quintic',joint.q,joint.dq,joint.ddq),
                          ('cartesian_line',transfer['q_traj'],transfer['dq_traj'],transfer['ddq_traj'])]:
        T=model.fk_batch(q);poses.append(T)
        comparisons.append(dict(method=name,duration_s=float(common),
            tcp_path_m=metrics.cartesian_path_length_m(T),straightness=metrics.straightness_ratio(T),
            max_deviation_mm=metrics.max_endpoint_deviation_m(T)*1000,
            peak_velocity_rad_s=float(abs(dq).max()),peak_acceleration_rad_s2=float(abs(ddq).max())))
    write_csv('planning_comparison.csv',comparisons)
    # Workspace is FK sampling, not proof that every enclosed pose is IK reachable or collision-free.
    lo,hi=model.joint_limits(); rng=np.random.default_rng(20261018)
    sample_q=rng.uniform(lo,hi,size=(20000,6));points=model.positions_batch(sample_q)
    workspace=dict(samples=len(points),seed=20261018,
                   min_xyz_m=points.min(axis=0).tolist(),max_xyz_m=points.max(axis=0).tolist(),
                   interpretation='FK samples only; excludes collision-free reachability claims')
    write_csv('workspace_samples.csv',[dict(x=p[0],y=p[1],z=p[2]) for p in points])
    # Verify exact singular q=0 and a well-conditioned home example.
    singular_rows=[]
    for name,q in [('zero',np.zeros(6)),('home',model.home_q())]:
        sv=np.linalg.svd(model.jacobian_tcp(q),compute_uv=False)
        singular_rows.append(dict(configuration=name,sigma_min=float(sv[-1]),
                                  characteristic_length_m=L,near_singular=bool(sv[-1]<0.02)))
    write_csv('singularity_cases.csv',singular_rows)
    lag=tracking.detect_pose_lag(model,log,max_lag=2)
    summary=dict(cycles=log.cycles, samples=log.n,
        joint_error_max_deg=float(np.degrees(abs(joint_err).max())),
        joint_rmse_rad=float(np.sqrt(np.mean(joint_err**2))),
        tcp_error_max_mm=float(pe.max()*1000),tcp_error_rms_mm=float(np.sqrt(np.mean(pe**2))*1000),
        orientation_error_max_deg=float(np.degrees(re.max())),
        logger_pose_max_error_mm=float(np.linalg.norm(actual[:,:3,3]-log.ee_pos,axis=1).max()*1000),
        logger_lag_samples=int(lag['lag_samples']),sigma_min=float(sigma.min()),
        jacobian_characteristic_length_m=L,near_singularity_threshold=0.02,
        near_singular_samples=int(np.sum(sigma<0.02)),workspace=workspace,
        log_sha256=hashlib.sha256(Path(log.path).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(Path(model.path).read_bytes()).hexdigest() if hasattr(model,'path') else model.describe()['model_sha256_head16'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
    ax[0].plot(log.t,np.degrees(abs(joint_err).max(axis=1)));ax[0].set_ylabel('Max joint error (deg)')
    ax[1].plot(log.t,pe*1000);ax[1].set_ylabel('TCP error (mm)')
    ax[2].plot(log.t,sigma);ax[2].axhline(0.02,color='r',ls='--');ax[2].set_ylabel('Normalized sigma min');ax[2].set_xlabel('Simulation time (s)')
    for a in ax:a.grid(alpha=.3)
    fig.tight_layout();fig.savefig(OUT/'tracking.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(11,5));a=fig.add_subplot(121,projection='3d')
    for name,T in zip(['Joint quintic','Cartesian line'],poses):a.plot(*T[:,:3,3].T,label=name)
    a.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Same transfer endpoints / duration');a.set_zlim(0.30,0.46);a.legend()
    b=fig.add_subplot(122,projection='3d');b.scatter(*points[::5].T,s=1,alpha=.15)
    b.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='FK workspace samples (not collision checked)')
    fig.tight_layout();fig.savefig(OUT/'planning_workspace.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2))
    return 0
