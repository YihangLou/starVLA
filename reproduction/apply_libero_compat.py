"""Allow only NumPy array globals used by pinned LIBERO initial states on torch 2.6."""
from pathlib import Path
import difflib
ROOT=Path(__file__).resolve().parent
p=ROOT/'LIBERO/libero/libero/benchmark/__init__.py'
old='        init_states = torch.load(init_states_path)'
new='''        # LIBERO initial states contain NumPy arrays, not only torch tensors.
        import numpy as np
        with torch.serialization.safe_globals([
            np.core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.float64)), type(np.dtype(np.float32)),
        ]):
            init_states = torch.load(init_states_path, weights_only=True)'''
s=p.read_text()
if new not in s:
    assert s.count(old)==1, 'LIBERO loader changed: inspect before patching'
    updated=s.replace(old,new)
    patch=''.join(difflib.unified_diff(s.splitlines(True),updated.splitlines(True),
        fromfile='a/libero/libero/benchmark/__init__.py',tofile='b/libero/libero/benchmark/__init__.py'))
    (ROOT/'libero_torch26.patch').write_text(patch)
    p.write_text(updated)
print('LIBERO torch 2.6 restricted NumPy loading compatibility applied')
