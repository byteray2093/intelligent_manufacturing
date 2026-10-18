"""Current integrated acceptance; fail immediately if any stage fails."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',action='store_true',help='Also render the 3-cycle mp4')
    args=ap.parse_args()
    os.environ.setdefault('MPLBACKEND','Agg')
    commands=[['-m','src.main','plan'],['-m','pytest','-q'],
              ['-m','src.main','demo','--cycles','3']+([] if args.video else ['--no-video']),
              ['-m','src.main','analyze']]
    for cmd in commands:
        print('Running:',sys.executable,*cmd,flush=True)
        subprocess.run([sys.executable,*cmd],cwd=ROOT,check=True)
    print('Integrated checks passed. Inspect results/integrated and the demo video.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
