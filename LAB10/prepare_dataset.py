from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'dataset'
TEST_RATIO = 0.5
SEED = 42
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def main():
    val_images = DATASET / 'images' / 'val'
    val_labels = DATASET / 'labels' / 'val'
    test_images = DATASET / 'images' / 'test'
    test_labels = DATASET / 'labels' / 'test'

    if not val_images.exists() or not val_labels.exists():
        raise FileNotFoundError(f'Validation folders not found in: {DATASET}')

    if test_images.exists() and any(test_images.iterdir()):
        print('Test split already exists, skipping split')
    else:
        images = sorted(
            p for p in val_images.iterdir()
            if p.suffix.lower() in EXTENSIONS
        )

        if not images:
            raise RuntimeError('No validation images found')

        random.Random(SEED).shuffle(images)
        count = max(1, int(len(images) * TEST_RATIO))

        test_images.mkdir(parents=True, exist_ok=True)
        test_labels.mkdir(parents=True, exist_ok=True)

        for image in images[:count]:
            label = val_labels / f'{image.stem}.txt'
            shutil.move(str(image), str(test_images / image.name))

            if label.exists():
                shutil.move(str(label), str(test_labels / label.name))
            else:
                (test_labels / f'{image.stem}.txt').touch()

        print(f'Moved {count} images from val to test')

    yaml_text = (
        f'path: {DATASET.as_posix()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        '\n'
        'names:\n'
        '  0: person\n'
        '  1: vehicle\n'
    )
    (DATASET / 'data.yaml').write_text(yaml_text, encoding='utf-8')

    for split in ('train', 'val', 'test'):
        folder = DATASET / 'images' / split
        total = sum(1 for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        print(f'{split}: {total} images')


if __name__ == '__main__':
    main()
