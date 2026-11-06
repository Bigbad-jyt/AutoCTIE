"""Joint training with validation-based checkpoint selection and resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from dataclasses import asdict,dataclass,field
from datetime import datetime,timezone
from pathlib import Path
import gc
import hashlib
import json
import math
import random
import sys
import transformers
import time
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from acti.batching import collate_records,to_device
from acti.evaluate import evaluate_rows,aggregate_saved_samples
from acti.io_utils import digest,read_jsonl,write_json,write_jsonl
from acti.models.joint_model import JointModel,JointConfig
from acti.registry import register
from acti.relation_samples import training_pairs
from acti.split import normalized_text


@dataclass
class TrainConfig:
    run_id:str
    configuration:str
    encoder_name:str
    encoder_path:str
    train_path:str
    validation_path:str
    test_path:str
    augmentation_path:str|None=None
    augmentation_source_sha256:str|None=None
    fold:int=1
    seed:int=43
    run_kind:str='smoke'
    original_equivalent_epochs:int=1
    effective_batch:int=16
    micro_batch:int=2
    evaluation_batch:int=2
    encoder_lr:float=2e-5
    head_lr:float=1e-3
    weight_decay:float=0.01
    gradient_clip:float=1.0
    warmup_ratio:float=0.1
    patience:int=5
    device:str='cpu'
    cpu_threads:int=2
    deterministic:bool=True
    joint:dict=field(default_factory=lambda:asdict(JointConfig()))

    def validate(self):
        expected={'C1':('BERT',False),'C2':('BERT',True),'C3':('CySecBERT',False),'C4':('CySecBERT',True)}
        if self.configuration not in expected or expected[self.configuration]!=(self.encoder_name,bool(self.augmentation_path)):
            raise ValueError('Configuration/encoder/augmentation mismatch')
        if self.run_kind not in ('smoke','unit_fixture','formal'):raise ValueError('Unknown run kind')
        if self.original_equivalent_epochs<1 or self.effective_batch<1 or not 1<=self.micro_batch<=self.effective_batch:
            raise ValueError('Invalid training budget')
        if self.evaluation_batch<1 or self.patience<1 or not 0<=self.warmup_ratio<1:raise ValueError('Invalid evaluation/scheduler config')


def selection_key(metrics):return (metrics['relation']['micro']['f1'],metrics['entity']['micro']['f1'])


def seed_everything(seed,deterministic):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark=False
    # Use deterministic recurrent kernels.
    torch.backends.cudnn.enabled=not deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False


def rng_state():
    name,keys,pos,gaussian,cached=np.random.get_state()
    return {'python':random.getstate(),'numpy':(name,keys.tolist(),pos,gaussian,cached),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);name,keys,pos,gaussian,cached=state['numpy']
    np.random.set_state((name,np.array(keys,dtype=np.uint32),pos,gaussian,cached))
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def make_optimizer(model,config):
    buckets={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=config.encoder_lr if name.startswith('shared.transformer.') else config.head_lr
        decay=0. if name.endswith('bias') or 'LayerNorm.weight' in name else config.weight_decay
        buckets.setdefault((lr,decay),[]).append(p)
    return torch.optim.AdamW([{'params':ps,'lr':lr,'weight_decay':wd} for (lr,wd),ps in buckets.items()])


def shuffled_group(records,seed,step,effective_batch):
    groups=math.ceil(len(records)/effective_batch);cycle,position=divmod(step,groups)
    indices=list(range(len(records)));random.Random(seed+cycle).shuffle(indices)
    chosen=indices[position*effective_batch:(position+1)*effective_batch]
    return [records[i] for i in chosen],cycle


def train_update(model,optimizer,group,config,data_cycle):
    """Accumulate CRF by sentence and BCE by class and pair."""
    model.train();optimizer.zero_grad(set_to_none=True)
    total_pairs=sum(len(training_pairs(r,config.seed,data_cycle)) for r in group)
    losses={'entity':0.,'relation':0.};pair_count=0
    for start in range(0,len(group),config.micro_batch):
        rows=group[start:start+config.micro_batch]
        batch=to_device(collate_records(rows,training=True,seed=config.seed,epoch=data_cycle),config.device)
        pair_n=sum(len(x) for x in batch['training_pairs'])
        result=model(batch,compute_loss=True)
        entity=result['entity_loss']*(len(rows)/len(group))
        relation=result['relation_loss']*(pair_n/total_pairs if total_pairs else 0.)
        loss=entity+model.config.lambda_relation*relation
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();losses['entity']+=float(entity.detach());losses['relation']+=float(relation.detach());pair_count+=pair_n
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
    optimizer.step()
    return dict(losses,gradient_norm_before_clip=float(norm),pairs=pair_count,
                loss=losses['entity']+model.config.lambda_relation*losses['relation'])


def predict_records(model,records,config):
    model.eval();predictions=[]
    for start in range(0,len(records),config.evaluation_batch):
        # Remove gold annotations before inference.
        rows=[{k:v for k,v in r.items() if k not in ('entities','relations','labels','label_ids','training_pairs')}
              for r in records[start:start+config.evaluation_batch]]
        batch=to_device(collate_records(rows),config.device)
        predictions.extend(model.predict(batch))
    return predictions


def save_evaluation(folder,gold,predictions):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    metrics,samples,errors=evaluate_rows(gold,predictions)
    write_jsonl(folder/'predictions.jsonl',predictions);write_json(folder/'metrics.json',metrics)
    write_jsonl(folder/'per_sample.jsonl',samples);write_jsonl(folder/'errors.jsonl',errors)
    recounted=aggregate_saved_samples(read_jsonl(folder/'per_sample.jsonl'))
    if recounted!={k:metrics[k] for k in ('entity','relation')}:raise AssertionError('Evaluation recount mismatch')
    return metrics


def _event(output,event,**fields):
    with (output/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'event':event,'at_utc':datetime.now(timezone.utc).isoformat(),**fields},sort_keys=True)+'\n')


def _checkpoint(model,optimizer,scheduler,output,state):
    folder=output/'checkpoints'/f"step_{state['step']:06d}"
    model.save(folder)
    torch.save({'state':state,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state()},folder/'training_state.pt')
    write_json(folder/'manifest.json',{'files_sha256':{p.name:digest(p) for p in folder.iterdir() if p.is_file()}})
    return folder


def load_training_inputs(config):
    """Load training data and verify the frozen augmentation pool."""
    config.validate()
    original=read_jsonl(config.train_path);validation=read_jsonl(config.validation_path)
    if not original or not validation:raise ValueError('Training/validation cannot be empty')
    records=list(original)
    if config.augmentation_path:
        augmentation=read_jsonl(config.augmentation_path)
        if config.run_kind=='formal':
            if any(r.get('software_test_only') for r in augmentation):
                raise ValueError('Synthetic software fixture forbidden in formal experiment')
            folder=Path(config.augmentation_path).parent
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if manifest['fold']!=config.fold:raise ValueError('Wrong frozen augmentation fold')
            if any(digest(folder/p)!=h for p,h in manifest['files_sha256'].items()):
                raise ValueError('Frozen augmentation hash mismatch')
            if digest(folder/'raw_pool.jsonl')!=config.augmentation_source_sha256:
                raise ValueError('Shared raw pool hash mismatch')
            raw=read_jsonl(folder/'raw_pool.jsonl')
            if len(augmentation)!=manifest['selected_samples'] or [r['sample_id'] for r in raw]!=[r['sample_id'] for r in augmentation]:
                raise ValueError('Frozen encoded/raw pool coverage mismatch')
            if any(r.get('software_test_only') is not False or r.get('source')!='LLM_and_first_author_accepted' for r in augmentation):
                raise ValueError('Only frozen real author-accepted samples are formal inputs')
        if len({r['sample_id'] for r in original+augmentation})!=len(original)+len(augmentation):
            raise ValueError('Duplicate augmentation sample ID')
        forbidden={normalized_text(r['text']) for r in original+validation}
        if any(normalized_text(r['text']) in forbidden for r in augmentation):raise ValueError('Augmentation exact duplicate')
        records+=augmentation
    return original,validation,records


def run_training(config,output,registry,resume_checkpoint=None,pause_after_update=None):
    config.validate();output=Path(output);registry=Path(registry)
    torch.set_num_threads(config.cpu_threads)
    if config.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA environment required')
    paths=[config.train_path,config.validation_path,config.test_path]
    if config.augmentation_path:paths.append(config.augmentation_path)
    inputs={str(Path(p).resolve()):digest(p) for p in paths}
    for name in ('config.json','model.safetensors'):
        p=Path(config.encoder_path)/name;inputs[str(p.resolve())]=digest(p)
    code_root=Path(__file__).resolve().parent
    code_hashes={str(p.relative_to(code_root)):digest(p) for p in [Path(__file__),code_root/'batching.py',code_root/'evaluate.py',code_root/'relation_samples.py',*sorted((code_root/'models').glob('*.py'))]}
    runtime={'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'torch_cuda':torch.version.cuda}
    signature=hashlib.sha256(json.dumps({'config':asdict(config),'inputs':inputs,'code':code_hashes,'runtime':runtime},sort_keys=True).encode()).hexdigest()
    if resume_checkpoint is None:
        if output.exists():raise FileExistsError(output)
        output.mkdir(parents=True)
        write_json(output/'config.json',asdict(config));write_json(output/'input_manifest.json',{'sha256':inputs,'signature':signature,'code_sha256':code_hashes,'runtime':runtime})
    else:
        if not Path(resume_checkpoint).resolve().is_relative_to(output.resolve()):raise ValueError('Resume checkpoint outside run directory')
        if (output/'summary.json').exists() and json.loads((output/'summary.json').read_text()).get('status')=='PASSED':raise ValueError('Completed run cannot be resumed')
        if not output.exists() or json.loads((output/'input_manifest.json').read_text())['signature']!=signature:
            raise ValueError('Resume config or inputs changed')
    register(registry,config.run_id,'RUNNING',run_kind=config.run_kind,formal_result=config.run_kind=='formal',output=str(output),resume=bool(resume_checkpoint))
    try:
        _event(output,'training_inputs_loaded',test_content_loaded=False)
        original,validation,records=load_training_inputs(config)
        if config.run_kind!='unit_fixture':
            if {r['report_id'] for r in original}&{r['report_id'] for r in validation}:raise ValueError('Train/validation report overlap')
            if {normalized_text(r['text']) for r in original}&{normalized_text(r['text']) for r in validation}:raise ValueError('Train/validation text overlap')
        original_steps=math.ceil(len(original)/config.effective_batch)
        max_steps=original_steps*config.original_equivalent_epochs
        warmup=math.ceil(max_steps*config.warmup_ratio)
        seed_everything(config.seed,config.deterministic)
        if resume_checkpoint is None:
            model,loading=JointModel.from_pretrained_encoder(config.encoder_path,JointConfig(**config.joint),head_seed=config.seed)
            state={'step':0,'signature':signature,'best_key':None,'best_checkpoint':None,'bad_checks':0,
                'samples_seen':0,'augmentation_samples_seen':0,'cuda_max_allocated':0,'cuda_max_reserved':0}
            model=model.to(config.device);optimizer=make_optimizer(model,config)
            scheduler=get_linear_schedule_with_warmup(optimizer,warmup,max_steps)
            parameters=model.parameter_report();head_hash=hashlib.sha256()
            for name,value in model.state_dict().items():
                if not name.startswith('shared.transformer.'):
                    head_hash.update(name.encode());head_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
            parameters['head_initial_state_sha256']=head_hash.hexdigest()
            write_json(output/'model_parameters.json',parameters)
            write_json(output/'loading_info.json',loading)
        else:
            checkpoint=Path(resume_checkpoint)
            manifest=json.loads((checkpoint/'manifest.json').read_text())
            if any(digest(checkpoint/p)!=h for p,h in manifest['files_sha256'].items()):raise ValueError('Checkpoint integrity mismatch')
            model=JointModel.load(checkpoint,config.device)
            payload=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
            state=payload['state']
            if state['signature']!=signature:raise ValueError('Checkpoint belongs to different run')
            optimizer=make_optimizer(model,config);scheduler=get_linear_schedule_with_warmup(optimizer,warmup,max_steps)
            optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);restore_rng(payload['rng'])
            del payload
            _event(output,'checkpoint_restored',step=state['step'],optimizer_and_scheduler_and_rng_restored=True)
        if config.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();original_ids={r['sample_id'] for r in original}
        stopped_early=False
        while state['step']<max_steps:
            group,cycle=shuffled_group(records,config.seed,state['step'],config.effective_batch)
            lr_used=[x['lr'] for x in optimizer.param_groups]
            losses=train_update(model,optimizer,group,config,cycle);scheduler.step();state['step']+=1
            state['samples_seen']+=len(group)
            state['augmentation_samples_seen']+=sum(r['sample_id'] not in original_ids for r in group)
            _event(output,'optimizer_update',step=state['step'],data_cycle=cycle,lr_used=lr_used,**losses)
            if state['step']==1 or state['step']%25==0:
                print(config.run_id,'update',state['step'],'/',max_steps,'loss',round(losses['loss'],4),flush=True)
            should_validate=state['step']%original_steps==0 or state['step']==max_steps
            if should_validate:
                predictions=predict_records(model,validation,config)
                folder=output/'validation'/f"step_{state['step']:06d}"
                metrics=save_evaluation(folder,validation,predictions);key=selection_key(metrics)
                improved=state['best_key'] is None or key>tuple(state['best_key'])
                if improved:
                    state['best_key']=key;state['best_checkpoint']=str(output/'checkpoints'/f"step_{state['step']:06d}");state['bad_checks']=0
                else:state['bad_checks']+=1
                _event(output,'validation_completed',step=state['step'],selection_key=key,improved=improved,test_content_loaded=False)
            if config.device.startswith('cuda'):
                state['cuda_max_allocated']=max(state['cuda_max_allocated'],torch.cuda.max_memory_allocated())
                state['cuda_max_reserved']=max(state['cuda_max_reserved'],torch.cuda.max_memory_reserved())
            should_pause=pause_after_update is not None and state['step']>=pause_after_update and state['step']<max_steps
            if should_validate or should_pause:
                checkpoint=_checkpoint(model,optimizer,scheduler,output,state)
                _event(output,'checkpoint_saved',step=state['step'],path=str(checkpoint))
            if should_pause:
                summary={'status':'PAUSED','step':state['step'],'resume_checkpoint':str(checkpoint),'max_steps':max_steps}
                write_json(output/'paused.json',summary);register(registry,config.run_id,'PAUSED',summary=summary)
                return summary
            if should_validate and state['bad_checks']>=config.patience:
                stopped_early=True;break
        _event(output,'training_finished',step=state['step'],stopped_early=stopped_early)
        selected=state['best_checkpoint']
        if selected is None:raise RuntimeError('No validation-selected checkpoint')
        _event(output,'checkpoint_selected',path=selected,selection_source='validation_only')
        training_seconds=time.perf_counter()-start
        del model,optimizer,scheduler;gc.collect()
        if config.device.startswith('cuda'):torch.cuda.empty_cache()
        selected_model=JointModel.load(selected,config.device);selected_model.eval()
        repeat_validation=predict_records(selected_model,validation,config)
        saved=read_jsonl(output/'validation'/Path(selected).name/'predictions.jsonl')
        if repeat_validation!=saved:raise AssertionError('Selected checkpoint validation predictions changed after reload')
        _event(output,'selected_checkpoint_validation_reload_verified')
        # Load the test set after checkpoint selection.
        test=read_jsonl(config.test_path);_event(output,'test_loaded_after_selection',samples=len(test))
        if config.run_kind!='unit_fixture':
            if {r['report_id'] for r in test}&{r['report_id'] for r in records+validation}:raise ValueError('Test report overlap')
            if {normalized_text(r['text']) for r in test}&{normalized_text(r['text']) for r in records+validation}:raise ValueError('Test text overlap')
        predictions=predict_records(selected_model,test,config)
        metrics=save_evaluation(output/'test',test,predictions)
        repeated=predict_records(selected_model,test,config)
        write_jsonl(output/'test'/'predictions_repeat.jsonl',repeated)
        if digest(output/'test'/'predictions.jsonl')!=digest(output/'test'/'predictions_repeat.jsonl'):
            raise AssertionError('Repeated checkpoint test predictions differ')
        second=evaluate_rows(test,repeated)[0]
        if second!=metrics:raise AssertionError('Repeated checkpoint test metrics differ')
        _event(output,'test_evaluated',repeat_predictions_and_metrics_identical=True)
        if any(digest(p)!=h for p,h in inputs.items()):raise ValueError('Run inputs changed')
        summary={'status':'PASSED','run_kind':config.run_kind,'formal_result':config.run_kind=='formal',
            'fold':config.fold,'configuration':config.configuration,'encoder':config.encoder_name,
            'original_train_samples':len(original),'training_samples':len(records),'validation_samples':len(validation),'test_samples':len(test),
            'augmentation_pool_size':len(records)-len(original),'augmentation_source_sha256':config.augmentation_source_sha256,
            'max_updates':max_steps,'completed_updates':state['step'],'original_equivalent_epoch_updates':original_steps,
            'warmup_updates':warmup,'samples_seen':state['samples_seen'],'augmentation_samples_seen':state['augmentation_samples_seen'],
            'best_checkpoint':selected,'best_validation_key':state['best_key'],'stopped_early':stopped_early,
            'repeat_checkpoint_evaluation_identical':True,'training_segment_seconds':training_seconds,
            'cuda_max_allocated_bytes':state['cuda_max_allocated'],'cuda_max_reserved_bytes':state['cuda_max_reserved']}
        write_json(output/'summary.json',summary)
        write_json(output/'output_manifest.json',{'files_sha256':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}})
        register(registry,config.run_id,'PASSED',summary=summary)
        return summary
    except Exception as exc:
        _event(output,'failed',error_type=type(exc).__name__,message=str(exc))
        register(registry,config.run_id,'FAILED',error_type=type(exc).__name__,message=str(exc),output=str(output),run_kind=config.run_kind)
        raise
