from pathlib import Path
from PIL import Image
from torchvision import transforms

oldroot = Path('datasets_raw/Cityscapes')
newroot_svg = Path('structure_generator/datasets/Cityscapes_256x512')
newroot_v2v = Path('image_generator/datasets/Cityscapes_256x512')

oldpaths = sorted(oldroot.glob('**/*.png'))
image_preprocess = transforms.Compose([
    transforms.Resize(256, Image.BICUBIC),
    ])
label_preprocess = transforms.Compose([
    transforms.Resize(256, Image.NEAREST),
    ])


def main(oldpath):
    data_type = oldpath.parts[2]
    if data_type == 'images':
        return
        newpath_svg = newroot_svg.joinpath(oldpath.parts[3] + '_B' + '/' + '_'.join(oldpath.stem.split('_')[:2]) + '/' + oldpath.name)
        newpath_v2v = newroot_v2v.joinpath(oldpath.parts[3] + '_B' + '/' + '_'.join(oldpath.stem.split('_')[:2]) + '/' + oldpath.name)
        resize = image_preprocess
    elif data_type == 'semantic_labels':
        newpath_svg = newroot_svg.joinpath(oldpath.parts[3] + '_A' + '/' + '_'.join(oldpath.stem.replace('color_mask_', '').split('_')[:2]) + '/' + oldpath.name)
        newpath_v2v = newroot_v2v.joinpath(oldpath.parts[3] + '_A' + '/' + '_'.join(oldpath.stem.replace('color_mask_', '').split('_')[:2]) + '/' + oldpath.name)
        resize = label_preprocess
    else:
        raise

    newpath_svg.parent.mkdir(parents=True, exist_ok=True)
    newpath_v2v.parent.mkdir(parents=True, exist_ok=True)

    img = resize(Image.open(oldpath))
    img.save(newpath_svg)
    img.save(newpath_v2v)

if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(32) as pool:
        pool.map(main, oldpaths)
