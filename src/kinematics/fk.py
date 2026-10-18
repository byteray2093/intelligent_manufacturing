import numpy as np

# UR5e 标准 DH 参数
A     = [0,      -0.425,  -0.3922, 0,      0,      0]
D     = [0.1625,  0,       0,      0.1333, 0.0997, 0.0996]
ALPHA = [np.pi/2, 0,       0,      np.pi/2, -np.pi/2, 0]

def dh_transform(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1],
    ])

def fk(q):
    q = np.asarray(q, dtype=float).reshape(6)
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(A[i], ALPHA[i], D[i], q[i])

    # 补偿矩阵：彻底解决 90度旋转 和 9.89cm 位置差异
    T_compensate = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, -0.0996],
        [0, 0, 0, 1]
    ])
    return T @ T_compensate

fk_flange = fk


def fk_tcp(q):
    """C's independent DH flange FK followed by B's fixed tool transform.

    DH geometry differs slightly from Menagerie (up to ~1.3 mm); keep that
    residual visible in validation rather than fitting away model differences.
    """
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    T[:3, 3] = [0, 0.1, 0]
    return fk_flange(q) @ T
