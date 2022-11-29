import os
import glob
try:
    from PIL import Image
except ImportError:
    import Image
import torchvision.transforms as transforms
import numpy as np

def data2npz(root, mode, index, input_shape=(3,128,128)):

    transform = transforms.Compose(
            [   transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
    )

    files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    for index in range(10000-200):
        img = Image.open(files[index % len(files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = transform(img_A).numpy().reshape((1,3,128,128))
        img_B = transform(img_B).numpy().reshape((1,3,128,128))

        if index == 0:
            img_A_all = img_A
            img_B_all = img_B
            continue

        img_A_all = np.concatenate((img_A_all,img_A), axis = 0)
        img_B_all = np.concatenate((img_B_all,img_B), axis = 0)

        if index % 100 == 0:
            print(index)

    return {"A": img_A_all, "B": img_B_all}

if __name__ == '__main__':
    img_dir ='/home/lys/680/dataset/archive/'
    dataset = data2npz(img_dir, 'train',0)
    print(dataset['A'].shape)
    print(dataset['B'].shape)
    np.savez("train.npz", A = dataset['A'], B = dataset['B'])

