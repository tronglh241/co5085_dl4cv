'''Compare two face images with a trained A2 model.

Prints the cosine similarity of the two embeddings and whether the model says they show the same person,
using the threshold chosen on the validation set (the same one used for the test accuracy), and saves a
figure with both images.

The model was trained on faces aligned to 112 x 112 with five landmarks (data/celeba_112). This script has no
face detector: an image that is not 112 x 112 is only centre-cropped and resized, so it works best with images
that are already aligned. Cross-check it on CelebA test images before trusting it on other photos.

Examples, from the repo root:
    python scripts/a2/demo.py a.jpg b.jpg
    python scripts/a2/demo.py a.jpg b.jpg --run resnet50_arcface --truth same --out demo_pair.png
'''

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from models.a2.embedding import FaceEmbeddingModel  # noqa: E402
from scripts.a2.result_io import read_result  # noqa: E402

SIZE = 112
RUNS = ('resnet50_softmax', 'resnet50_arcface', 'mobilenetv3_softmax', 'mobilenetv3_arcface')
# The same transform as the validation and test loaders of the configs.
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def newest_best(run):
    found = sorted((_repo_root / 'checkpoints' / 'a2' / f'train_{run}').glob('*/best_*.pt'),
                   key=lambda path: path.parent.name)
    return found[-1] if found else None


def validation_threshold(run, results_dir):
    path = Path(results_dir) / f'{run}.txt'
    if not path.is_file():
        return None
    return read_result(path).get('val', {}).get('threshold')


def load_model(run, checkpoint):
    backbone, head = run.rsplit('_', 1)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    model = FaceEmbeddingModel(
        backbone=backbone,
        head=head,
        num_classes=int(state['classifier.weight'].shape[0]),
        pretrained=False,
    )
    model.load_state_dict(state)
    return model.eval()


def load_face(path):
    '''The image as RGB, at 112 x 112. Returns the image and whether it had to be resized.'''
    with Image.open(path) as image:
        image = image.convert('RGB')

    if image.size == (SIZE, SIZE):
        return image, False

    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.BICUBIC)
    return image, True


@torch.no_grad()
def embed(model, image):
    batch = TRANSFORM(image).unsqueeze(0)
    return F.normalize(model(batch, return_embedding=True).float(), dim=1)[0]


def save_figure(path, images, similarity, threshold, same, truth, run):
    width, height = 5.6, 3.2
    figure = plt.figure(figsize=(width, height))
    # Two square panels at fixed positions, so the layout does not depend on the title length.
    side = 0.40 * width / height
    for left, image in zip((0.06, 0.54), images):
        axis = figure.add_axes((left, 0.07, 0.40, side))
        axis.imshow(image)
        axis.set_axis_off()

    verdict = 'SAME person' if same else 'DIFFERENT people'
    title = f'{verdict}\ncosine similarity {similarity:.3f} (threshold {threshold:.3f})'
    if truth is not None:
        correct = same == (truth == 'same')
        title += f'\nground truth: {truth}, model is {"correct" if correct else "WRONG"}'
    figure.suptitle(title, fontsize=9, color='tab:green' if same else 'tab:red', y=0.985)
    figure.text(0.5, 0.01, run, ha='center', fontsize=7, color='gray')
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Compare two face images with an A2 model.')
    parser.add_argument('image_a')
    parser.add_argument('image_b')
    parser.add_argument('--run', default='mobilenetv3_arcface', choices=RUNS)
    parser.add_argument('--checkpoint', default=None, help='default: the newest best_*.pt of the run')
    parser.add_argument('--threshold', type=float, default=None,
                        help='default: the validation threshold stored in the results of the run')
    parser.add_argument('--results-dir', default='reports/results/a2')
    parser.add_argument('--truth', choices=('same', 'different'), default=None,
                        help='the known answer, only used to annotate the figure')
    parser.add_argument('--out', default='demo_pair.png')
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else newest_best(args.run)
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit(f'no checkpoint for {args.run}: pass --checkpoint, or download one with '
                         f'`bash modal_run/a2.sh pull-checkpoints {args.run}`')

    threshold = args.threshold if args.threshold is not None else validation_threshold(args.run, args.results_dir)
    if threshold is None:
        raise SystemExit(f'no validation threshold for {args.run} in {args.results_dir}: pass --threshold')

    model = load_model(args.run, checkpoint)
    faces = []
    for path in (args.image_a, args.image_b):
        image, resized = load_face(path)
        if resized:
            print(f'warning: {path} is not {SIZE}x{SIZE}; it was centre-cropped and resized without face alignment')
        faces.append(image)

    similarity = float(embed(model, faces[0]) @ embed(model, faces[1]))
    same = similarity >= threshold

    print(f'model: {args.run} ({checkpoint})')
    print(f'cosine similarity: {similarity:.4f}')
    print(f'threshold (chosen on the validation set): {threshold:.4f}')
    print(f'decision: {"SAME person" if same else "DIFFERENT people"}')
    if args.truth is not None:
        print(f'ground truth: {args.truth} -> model is {"correct" if same == (args.truth == "same") else "WRONG"}')

    save_figure(args.out, faces, similarity, threshold, same, args.truth, args.run)
    print(f'figure: {args.out}')


if __name__ == '__main__':
    main()
