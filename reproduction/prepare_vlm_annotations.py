"""Normalize the pinned COCO annotations without modifying the downloaded source.

The published archive uses images/videos; this StarVLA snapshot consumes
image/video. JSONL also avoids decoding the entire JSON text beside Torch.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/LLaVA-OneVision-COCO'
SOURCE = DATA / 'llava_jsons/sharegpt4v_coco.json'
OUTPUT = SOURCE.with_suffix('.jsonl')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    annotations = json.loads(SOURCE.read_text())
    images = set()
    temporary = OUTPUT.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for item in annotations:
            assert isinstance(item, dict)
            for singular, plural in (('image', 'images'), ('video', 'videos')):
                if not item.get(singular) and item.get(plural):
                    item[singular] = item[plural]
            names = item.get('image', [])
            if isinstance(names, str):
                names = [names]
            assert names, item.get('id')
            for name in names:
                assert (DATA / 'images' / name).is_file(), name
                images.add(name)
            assert item.get('conversations'), item.get('id')
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(temporary, OUTPUT)
    report = dict(
        source=str(SOURCE.resolve()), source_sha256=sha256(SOURCE),
        output=str(OUTPUT.resolve()), output_sha256=sha256(OUTPUT),
        annotations=len(annotations), referenced_images=len(images),
        conversion='Copy nonempty images/videos to image/video when absent; retain all original fields and records; serialize as JSONL.',
        original_preserved=True,
    )
    (ROOT / 'vlm_annotation_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
