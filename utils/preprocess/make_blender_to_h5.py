import os
import glob
import h5py
import numpy as np
from PIL import Image

def blender_to_h5(data_dir, resize=None):
    image_path = glob.glob(os.path.join(data_dir, "*.png"))
    image_path.sort()

    # read frames
    images = []
    for path in image_path:
        image = Image.open(path)

        if resize is not None:
            image = image.resize(resize)

        image = np.array(image)
        images.append(image)
    
    images = np.stack(images, axis=0)   # (N, H, W, C)
    images = images.transpose(0, 3, 1, 2)   # (N, C, H, W)

    print(f'Saving {len(image)} imagea to h5...')

    # save to h5
    with h5py.File(os.path.join(data_dir, "images.h5"), "w") as f:
        f.create_dataset("images", data=images)
        
    

if __name__ == '__main__':
    base_dir = "/home/titan/projects_backup/intrieve_data/render_data/event_data_ldr"
    resize = (512, 512)
    scenes = ('hotdog', 'lego', 'mic', 'ficus', 'drums', 'chair', 'materials')

    for scene in scenes:
        data_dir = os.path.join(base_dir, scene)

        blender_to_h5(
            data_dir=data_dir,
            resize=resize,
        )