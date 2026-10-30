"""Evaluate all twenty demo tasks with hosted models and in-process workers."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from gyms.decomposer_demo.tasks import task_ids
from gyms.decomposer_demo.run import episode


def summarize(results):
    return {'benchmark':'decomposer-demo','harness':results[0]['harness'] if results else None,
            'attempts':len(results), 'mean_score':sum(r['score'] for r in results)/len(results) if results else None,
            'passed':sum(r['passed'] for r in results),'errors':sum(r['status']!='finished' for r in results)}


async def main(args):
    from gyms.decomposer_demo.models import create_hosted_model, WORKER_MODELS
    if min(args.repetitions,args.concurrency)<1:
        raise ValueError('Repetitions and concurrency must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    policy=create_hosted_model(args.model);worker=create_hosted_model(args.worker_model) if args.worker_model else None
    results=[]
    manifest={'benchmark':'decomposer-demo','status':'running','model':args.model,'worker_model':args.worker_model,'worker_models':{name:args.worker_model or model for name,model in WORKER_MODELS.items()},
              'language':args.language,'harness':args.harness,'tasks':20,'repetitions':args.repetitions,
              'concurrency':args.concurrency,'started_at':time.time(),'episodes':results}
    def save():
        temp=args.output/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2));temp.replace(args.output/'manifest.json')
    save();semaphore=asyncio.Semaphore(args.concurrency)
    async def run(id,attempt):
        async with semaphore:
            result=await episode(id,policy,worker_policy=worker,language=args.language,harness=args.harness)
            (args.output/f'{id}-{attempt}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
            results.append({k:result[k] for k in ('task_id','environment','language','harness','status','score','passed','error_type')})
            save();print(f'{len(results)}/{20*args.repetitions}: {id} score={result["score"]} status={result["status"]}',flush=True)
    try:
        async with asyncio.TaskGroup() as group:
            for id in task_ids():
                for attempt in range(1,args.repetitions+1):
                    group.create_task(run(id,attempt))
        manifest['status']='completed'
        summary=summarize(results)
        summary['by_environment']={env:summarize([r for r in results if r['environment']==env]) for env in ('calendar','estate','frontend')}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))
    finally:
        if manifest['status']=='running':manifest['status']='interrupted'
        manifest['finished_at']=time.time();save()
        for model in (m for m in (policy,worker) if m is not None):
            model.http_client.close();await model.http_async_client.aclose()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='lmrouter/qwen_3_5_4b_unlooped_non_thinking')
    parser.add_argument('--worker-model', help='Override all demo worker assignments with this model')
    parser.add_argument('--language',choices=['en','ru'],default='ru')
    parser.add_argument('--harness',choices=['decomposer','react'],default='decomposer')
    parser.add_argument('-n','--repetitions',type=int,default=1)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
