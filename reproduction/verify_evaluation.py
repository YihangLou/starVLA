"""Check simulator imports and all task assets without creating a GPU renderer."""
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','glx')
os.environ.setdefault('PYOPENGL_PLATFORM','glx')
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT/'source')
import ctypes
ctypes.CDLL('libEGL.so.1')  # Check the GPU renderer's loader without initializing a device.
import mujoco
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from examples.LIBERO.eval_files import eval_libero
report={'mujoco':mujoco.__version__,'numpy':np.__version__,'renderer':'not exercised without GPU','egl_loader':'available','cpu_import_backend':os.environ['MUJOCO_GL'],'suites':{}}
for name in ('libero_spatial','libero_object','libero_goal','libero_10'):
    suite=benchmark.get_benchmark_dict()[name]()
    assert suite.n_tasks==10
    for i in range(suite.n_tasks):
        task=suite.get_task(i)
        assert (Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file).is_file()
        states=suite.get_task_init_states(i)
        assert len(states)>0
    report['suites'][name]={'tasks':suite.n_tasks,'all_initial_states_loaded':True}
(ROOT/'evaluation_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
