"""Bounded GPU verification; does not launch a production training run."""
import argparse, datetime, json, os, tempfile, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['kernels','deepspeed','model'],required=True); args=p.parse_args()
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT/'source')
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
start=time.monotonic(); torch.manual_seed(42); torch.set_num_threads(4)
report={'phase':args.phase,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))}
assert torch.cuda.is_available()
if args.phase=='kernels':
    import torchvision, flash_attn, deepspeed, triton
    from flash_attn import flash_attn_func,flash_attn_varlen_func
    import torch.nn.functional as F
    report.update(torchvision=torchvision.__version__,flash_attn=flash_attn.__version__,deepspeed=deepspeed.__version__,triton=triton.__version__)
    x=torch.randn(256,256,device='cuda',requires_grad=True)
    (x@x).square().mean().backward(); assert torch.isfinite(x.grad).all()
    for dtype in (torch.float16,torch.bfloat16):
        q,k,v=[torch.randn(2,64,4,128,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
        y=flash_attn_func(q,k,v,causal=True)
        ref=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True).transpose(1,2)
        torch.testing.assert_close(y,ref,atol=0.02,rtol=0.02)
        y.float().square().mean().backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,v))
        q,k,v=[t.detach().flatten(0,1).requires_grad_() for t in (q,k,v)]
        cu=torch.tensor([0,64,128],device='cuda',dtype=torch.int32)
        z=flash_attn_varlen_func(q,k,v,cu,cu,64,64,causal=True)
        z.float().square().mean().backward()
        assert all(torch.isfinite(t.grad).all() for t in (q,k,v))
        report[str(dtype)]='fixed/variable length attention forward/backward passed'
    keep=torchvision.ops.nms(torch.tensor([[0,0,10,10],[1,1,9,9]],device='cuda',dtype=torch.float32),torch.tensor([0.9,0.8],device='cuda'),0.5)
    assert keep.tolist()==[0]
    compiled=torch.compile(lambda a:(a.sin()+a.cos()).square(),fullgraph=True)
    a=torch.randn(128,device='cuda'); torch.testing.assert_close(compiled(a),(a.sin()+a.cos()).square())
    report['cuda_matrix_backward']='pass'; report['torchvision_cuda_nms']='pass'; report['triton_compile']='pass'
elif args.phase=='deepspeed':
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    import deepspeed, torch.distributed as dist
    os.environ.update(RANK='0',WORLD_SIZE='1',LOCAL_RANK='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29643')
    torch.cuda.set_device(0)
    dist.init_process_group('nccl',init_method='file://'+tempfile.mktemp(prefix='starvla-nccl-',dir='/tmp'),rank=0,world_size=1)
    try:
        m=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.GELU(),torch.nn.Linear(64,8)).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,fused=True)
        cfg={'train_micro_batch_size_per_gpu':2,'gradient_accumulation_steps':1,'bf16':{'enabled':True},'zero_optimization':{'stage':2},'zero_allow_untested_optimizer':True,'steps_per_print':100}
        engine,_,_,_=deepspeed.initialize(model=m,optimizer=opt,config=cfg,dist_init_required=False)
        losses=[]
        for _ in range(2):
            loss=engine(torch.randn(2,32,device='cuda',dtype=torch.bfloat16)).float().square().mean()
            assert torch.isfinite(loss); engine.backward(loss); engine.step(); losses.append(loss.item())
        with tempfile.TemporaryDirectory(prefix='starvla-ds-',dir='/tmp') as td:
            engine.save_checkpoint(td,tag='smoke')
            load_path,_=engine.load_checkpoint(td,tag='smoke'); assert load_path
        report.update(zero_stage=2,optimizer_steps=2,losses=losses,checkpoint_roundtrip='pass',nccl='pass')
    finally:
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()
elif args.phase=='model':
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat
    cfg=OmegaConf.load(ROOT/'configs/QwenOFT_vla.yaml'); apply_config_compat(cfg)
    DATASET_NAMED_MIXTURES['gpu_upgrade_smoke']=[DATASET_NAMED_MIXTURES['libero_all'][0]]
    cfg.datasets.vla_data.data_mix='gpu_upgrade_smoke'
    print('Loading real LIBERO sample',flush=True)
    ds=get_vla_dataset(cfg.datasets.vla_data); batch=[ds.datasets[0][0]]
    print('Loading QwenOFT 4B',flush=True)
    model=build_framework(cfg).cuda().train()
    model.qwen_vl_interface.model.gradient_checkpointing_enable()
    model.qwen_vl_interface.model.config.use_cache=False
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6,fused=True)
    losses=[]; norms=[]
    tracked=next(model.action_model.parameters()); before=tracked.detach().float().clone()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss=model(examples=batch)['action_loss']; assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize()
        losses.append(loss.item()); norms.append(norm.item())
        print('MODEL_STEP',step+1,'loss',loss.item(),'grad_norm',norm.item(),flush=True)
    assert not torch.equal(before,tracked.detach().float())
    model.eval(); pred=model.predict_action(examples=batch)['normalized_actions']
    import numpy as np
    assert pred.shape==(1,8,7) and np.isfinite(pred).all()
    report.update(framework='QwenOFT',batch_size=1,real_libero_sample=True,optimizer_steps=2,losses=losses,grad_norms=norms,action_head_updated=True,prediction_shape=list(pred.shape),actual_attention=model.qwen_vl_interface.model.config._attn_implementation)
torch.cuda.synchronize()
report.update(status='pass',peak_gpu_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3,3),elapsed_seconds=round(time.monotonic()-start,2),observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
output=ROOT/'logs'/('gpu_upgrade_'+args.phase+'.json'); output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
