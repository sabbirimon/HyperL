#!/usr/bin/env python3
"""Run generated local HyperL cases on the explicitly selected Radeon, with CPU checks.
No downloads, driver changes or CPU/other-device fallback. Writes a new report only.
"""
import argparse,json,os,platform,subprocess,tempfile,time
from datetime import datetime,timezone
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli',required=True,type=Path)
    parser.add_argument('--bridge',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device-name',help='Exact Metal name; otherwise select the sole Radeon Pro 560X')
    args=parser.parse_args()
    for executable in (args.cli,args.bridge):
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable,os.X_OK):parser.error('Explicit absolute reviewed executable paths required')
    if args.output.exists() or args.output.is_symlink():parser.error('Choose a new report path; never overwrites')
    report={'format':'hyperl-metal-qualification/1','timeUtc':datetime.now(timezone.utc).isoformat(),'os':platform.mac_ver()[0],'status':'unrun','requestedDevice':args.device_name or 'Radeon Pro 560X','scope':'Generated bounded elementwise cases only; no model, speedup, capacity or general backend qualification','cases':[]}
    def command(arguments):
        start=time.perf_counter();p=subprocess.run([str(args.cli),*map(str,arguments)],capture_output=True,text=True,timeout=23)
        if len(p.stdout)>16*1024*1024 or len(p.stderr)>65536:raise ValueError('CLI output limit')
        return p,(time.perf_counter()-start)*1000
    exit_code=3
    try:
        p,_=command(['metal-probe',args.bridge])
        if p.returncode:raise ValueError(p.stderr[:2048])
        probe=json.loads(p.stdout);report['probe']=probe
        if probe.get('format')!='hyperl-metal-probe/1' or probe.get('inferenceQualified') is not False:raise ValueError('Unexpected discovery contract')
        devices=probe.get('devices',[])
        if len(devices)>128:raise ValueError('Device count limit')
        matches=[d['name'] for d in devices if d['name']==args.device_name] if args.device_name else [d['name'] for d in devices if 'Radeon Pro 560X' in d['name']]
        if len(matches)!=1:
            report['reason']='Exact requested Radeon unavailable or ambiguous in this process; no fallback'
        else:
            name=matches[0];report['device']=name
            program={'format':'hyperl/1','inputs':['x','w'],'instructions':[{'output':'value','operation':'multiply','inputs':['x','w']},{'output':'positive','operation':'relu','inputs':['value']}],'output':'positive'}
            with tempfile.TemporaryDirectory(prefix='hyperl-metal-qualify-') as temporary:
                folder=Path(temporary);program_file=folder/'program.json';inputs_file=folder/'inputs.json'
                for n in (3,257,65536):
                    program_file.write_text(json.dumps(program))
                    values={'x':[-1,2,3],'w':[2,3,4]} if n==3 else {'x':[(i%17)-8 for i in range(n)],'w':[(i%7)-3 for i in range(n)]}
                    inputs_file.write_text(json.dumps(values))
                    cpu,cpu_ms=command(['run',program_file,inputs_file]);gpu,wall_ms=command(['metal-run',args.bridge,name,program_file,inputs_file])
                    if cpu.returncode or gpu.returncode:raise ValueError((cpu.stderr+gpu.stderr)[:2048])
                    reference=json.loads(cpu.stdout);result=json.loads(gpu.stdout)
                    if result.get('device')!=name or result.get('cpuVerified') is not True or result.get('metrics',{}).get('gpuCompletionConfirmed') is not True:raise ValueError('Device or completion mismatch')
                    actual=result['result']
                    if len(actual)!=n or len(reference)!=n or any(abs(a-b)>1e-6*max(1,abs(b)) for a,b in zip(actual,reference)):raise ValueError('CPU-reference mismatch')
                    report['cases'].append({'elements':n,'status':'pass','cpuCliWallMs':cpu_ms,'metalCliWallMsIncludingCpuVerification':wall_ms,'nativeMetrics':result['metrics'],'sample':actual[:3]})
                # Exercise add in a dependency chain and a final input/output branch.
                program['instructions'].insert(1,{'output':'shifted','operation':'add','inputs':['value','x']})
                program['instructions'][-1]['inputs']=['shifted'];program_file.write_text(json.dumps(program));inputs_file.write_text(json.dumps({'x':[-2,-1,0,1,2],'w':[2,3,4,-5,6]}))
                p,wall_ms=command(['metal-run',args.bridge,name,program_file,inputs_file])
                if p.returncode:raise ValueError(p.stderr[:2048])
                result=json.loads(p.stdout)
                if result['device']!=name or result['result']!=[0,0,0,0,14] or not result['cpuVerified']:raise ValueError('Add-chain mismatch')
                report['cases'].append({'elements':5,'status':'pass','operation':'multiply/add/relu','metalCliWallMsIncludingCpuVerification':wall_ms,'nativeMetrics':result['metrics']})
                # Rejected by the CPU preflight before any overflowing GPU request.
                program['instructions'].pop(1);program['instructions'][-1]['inputs']=['value'];program_file.write_text(json.dumps(program))
                inputs_file.write_text(json.dumps({'x':[3.4028234663852886e38],'w':[2]}))
                p,_=command(['metal-run',args.bridge,name,program_file,inputs_file])
                if p.returncode!=2 or 'nonfinite' not in p.stderr:raise ValueError('Overflow-before-ReLU was not rejected')
                report['cases'].append({'status':'expected-rejection','operation':'overflow before relu','gpuDispatched':False})
            report['status']='passed-listed-cases';exit_code=0
    except (ValueError,KeyError,TypeError,OSError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
        report['status']='failed';report['reason']=str(error)[:2048];exit_code=2
    with args.output.open('x') as output:json.dump(report,output,indent=2);output.write('\n')
    print(json.dumps({'status':report['status'],'report':str(args.output),'reason':report.get('reason')}))
    return exit_code

if __name__=='__main__':raise SystemExit(main())
