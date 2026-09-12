"""Bounded frozen-policy diagnostics on the GPUs allocated to a scheduled job."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading

from oh_my_duck.core.paths import project_root
from oh_my_duck.rl.training.registry import default_registry
from .campaign import worker_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = project_root()
    plan = json.loads(args.config.read_text())
    if plan.get('schema_version') != 1 or not plan.get('cases'):
        parser.error('Expected nonempty schema-1 cases')
    seen = set()
    for case in plan['cases']:
        name = case['id']
        if name in seen or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
            parser.error('Cases require unique filesystem-safe IDs')
        seen.add(name)
        policy = root / case['policy']
        if hashlib.sha256(policy.read_bytes()).hexdigest() != case['policy_sha256']:
            raise ValueError(f'Policy hash changed: {name}')
    if args.dry_run:
        print(json.dumps({'cases': len(seen), 'policy_hashes': 'verified', 'learners': 0}))
        return 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        count = subprocess.check_output([str(root/'.envs/mujoco/bin/python'), '-c', 'import torch; print(torch.cuda.device_count())'], text=True).strip()
        visible = ','.join(str(i) for i in range(int(count)))
    devices = [x.strip() for x in visible.split(',') if x.strip()]
    if not devices:
        raise RuntimeError('No GPUs allocated; refusing implicit local fallback')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'plan':plan,'devices':devices,'started_at':datetime.now(timezone.utc).isoformat(),'cases':{}}
    slots = queue.Queue()
    for device in devices: slots.put(device)
    lock = threading.Lock()
    children = []
    stopped = threading.Event()
    def save():
        temporary = output/'result.json.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(output/'result.json')
    def stop(signum, frame):
        stopped.set()
        for child in list(children):
            if child.poll() is None:
                try: os.killpg(child.pid, signum)
                except ProcessLookupError: pass
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop)
    save()
    def run(case):
        device = slots.get()
        try:
            if stopped.is_set(): return case['id'], {'status':'interrupted'}
            target = output/case['id']
            arguments = ['--task',case['task'],'--policy',str(root/case['policy']),'--output',str(target),'--seed',str(case['seed'])]
            if case['backend'] == 'cpu-bam':
                command = [str(root/'.envs/mujoco/bin/python'),str(root/'omd.py'),'rehearsal','--',*arguments]
                backend_env = {}
            else:
                arguments += ['--record-rewards',*case.get('arguments',[])]
                spec = default_registry().get(case['backend'],root).command('eval',arguments)
                command,backend_env = spec.argv,spec.environment
            env = {**worker_environment(device,output.name),**backend_env,'CUDA_VISIBLE_DEVICES':device,'WANDB_MODE':'offline',
                   'OMD_PROJECT_ROOT':str(root),'PYTHONPATH':str(root/'src'),'MUJOCO_GL':'egl'}
            with (output/(case['id']+'.log')).open('x') as stream:
                child = subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                with lock:
                    children.append(child)
                    report['cases'][case['id']]={'status':'running','pid':child.pid,'gpu':device,'command':list(command)}
                    save()
                code = child.wait()
            result_path = target/'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if result.get('policy_sha256') not in (None,case['policy_sha256']):
                raise ValueError('Evaluated policy hash mismatch')
            return case['id'],{'status':'execution_completed' if code in (0,2) and result else 'execution_failed',
                               'exit_code':code,'gpu':device,'result':result}
        finally:
            slots.put(device)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {pool.submit(run,case):case['id'] for case in plan['cases']}
        for future in as_completed(futures):
            try: key,value = future.result()
            except Exception as error: key,value = futures[future],{'status':'execution_failed','error':repr(error)}
            with lock: report['cases'][key]=value;save()
    report['status']='interrupted' if stopped.is_set() else ('completed' if all(v['status']=='execution_completed' for v in report['cases'].values()) else 'execution_failed')
    report['finished_at']=datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
