"""CPU preparation checks. Does not claim GPU execution or training success."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT/'source')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
parser=argparse.ArgumentParser()
parser.add_argument('--phase',default='all')
args=parser.parse_args()
report={}
if args.phase=='imports':
    import torch
    torch.set_num_threads(1)
    for name in ('torchvision','transformers','flash_attn','deepspeed','pyarrow','av','diffusers'):
        print('IMPORTING',name,flush=True)
        mod=importlib.import_module(name)
        print('IMPORT_OK',name,flush=True)
        report[name]=getattr(mod,'__version__','imported')
    from starVLA.model.framework.base_framework import _auto_import_framework_modules
    print('IMPORTING framework registry',flush=True)
    _auto_import_framework_modules()
    print('IMPORT_OK framework registry',flush=True)
    report.update(torch=torch.__version__,cuda_build=torch.version.cuda,gpu_count=torch.cuda.device_count(),framework_imports='pass')
elif args.phase.startswith('trainer:'):
    # Each entry point creates a process-global Accelerator during import.
    # They must be verified in separate processes, just as they are launched.
    mode=args.phase.split(':',1)[1]
    assert mode in ('vla','cotrain')
    module='starVLA.training.train_starvla'+('_cotrain' if mode=='cotrain' else '')
    importlib.import_module(module)
    report=dict(module=module,import_status='pass',training_started=False)
elif args.phase.startswith('robot:'):
    import numpy as np
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    cfg=OmegaConf.load(ROOT/'configs/QwenOFT_vla.yaml')
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    suite=args.phase.split(':',1)[1]
    matches=[item for item in DATASET_NAMED_MIXTURES['libero_all'] if item[0]==suite]
    assert len(matches)==1
    DATASET_NAMED_MIXTURES['preparation_check']=matches
    cfg.datasets.vla_data.data_mix='preparation_check'
    ds=get_vla_dataset(cfg.datasets.vla_data)
    single=ds.datasets[0]
    indices=sorted(set([0,len(single)//2,len(single)-1]))
    for index in indices:
        sample=single[index]
        assert sample['action'].shape==(8,7),sample['action'].shape
        assert np.isfinite(sample['action']).all()
        assert len(sample['image'])==2
        assert all(im.size==(224,224) for im in sample['image'])
        assert isinstance(sample['lang'],str) and sample['lang']
    # Exercise mixture sampling and collation too.
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import collate_fn
    batch=next(iter(DataLoader(ds,batch_size=2,num_workers=0,collate_fn=collate_fn)))
    assert len(batch)==2
    report=dict(dataset=suite,steps=len(single),sample_indices=indices,action_shape=[8,7],cameras=2,batch_size=2)
elif args.phase=='models':
    from transformers import AutoConfig,AutoProcessor
    import numpy as np
    import struct
    for name in ('Qwen3-VL-4B-Instruct','Qwen3-VL-4B-Instruct-Action'):
        path=ROOT/'source/playground/Pretrained_models'/name
        cfg=AutoConfig.from_pretrained(path,local_files_only=True)
        processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        index=json.loads((path/'model.safetensors.index.json').read_text())
        shard_names=set(index['weight_map'].values())
        tensors={}
        for shard in shard_names:
            with (path/shard).open('rb') as f:
                header_len=struct.unpack('<Q',f.read(8))[0]
                assert header_len<100*1024**2
                header=json.loads(f.read(header_len))
            data_size=(path/shard).stat().st_size-8-header_len
            for key,value in header.items():
                if key=='__metadata__':continue
                assert 0<=value['data_offsets'][0]<=value['data_offsets'][1]<=data_size
                tensors[key]=shard
        assert tensors==index['weight_map']
        report[name]=dict(model_type=cfg.model_type,tensors=len(tensors),shards=len(shard_names),processor=type(processor).__name__)
    processor=AutoProcessor.from_pretrained(ROOT/'source/playground/Pretrained_models/fast',trust_remote_code=True,local_files_only=True)
    tokens=processor(np.zeros((1,8,7),dtype=np.float32))
    decoded=processor.decode(tokens,time_horizon=8,action_dim=7)
    assert decoded.shape==(1,8,7)
    report['fast_tokenizer_roundtrip']='pass'
elif args.phase=='vlm':
    # Audit the complete annotation set before loading the tensor libraries.
    # json.loads temporarily keeps the large input string and parsed objects.
    print('VLM auditing annotation image paths',flush=True)
    path=ROOT/'data/LLaVA-OneVision-COCO'
    annotation_count=0
    images=set()
    with (path/'llava_jsons/sharegpt4v_coco.jsonl').open() as stream:
        for line in stream:
            item=json.loads(line)
            values=item.get('image',[])
            if isinstance(values,str):values=[values]
            assert values, item.get('id')
            images.update(values)
            annotation_count+=1
    assert images
    for name in images:
        assert (path/'images'/name).is_file(),name
    # Release the full image-path set before constructing the real loader.
    image_count=len(images)
    del images
    import gc
    gc.collect()
    import ctypes
    ctypes.CDLL(None).malloc_trim(0)
    print('VLM annotation audit passed',annotation_count,image_count,flush=True)
    from omegaconf import OmegaConf
    from starVLA.dataloader.vlm_datasets import make_vlm_dataloader
    from torch.utils.data import DataLoader
    cfg=OmegaConf.load(ROOT/'configs/QwenOFT_cotrain.yaml')
    import random
    random.seed(42)
    print('VLM constructing real loader',flush=True)
    original=make_vlm_dataloader(cfg)['train_dataloader']
    print('VLM fetching real batch',flush=True)
    loader=DataLoader(original.dataset,batch_size=1,num_workers=0,collate_fn=original.collate_fn)
    batch=next(iter(loader))
    assert batch['input_ids'].numel()>0 and batch['pixel_values'].numel()>0
    import torch
    assert torch.isfinite(batch['pixel_values']).all()
    assert batch['labels'].shape==batch['input_ids'].shape
    supervised_tokens=int((batch['labels']!=-100).sum())
    assert supervised_tokens>0
    assert batch['position_ids'].shape[-1]==batch['input_ids'].shape[-1]
    assert len(original.dataset)==annotation_count
    report=dict(annotations=annotation_count,referenced_images=image_count,supervised_tokens=supervised_tokens,batch_shapes={k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')})
elif args.phase=='all':
    subprocess.run(['bash',str(ROOT/'run_cpu_preflight.sh')],check=True)
    report=json.loads((ROOT/'preparation_verification.json').read_text())
else:raise ValueError(args.phase)
print(json.dumps(report,indent=2),flush=True)
(ROOT/'logs'/('preflight_'+args.phase.replace(':','_')+'.json')).write_text(json.dumps(report,indent=2)+'\n')
