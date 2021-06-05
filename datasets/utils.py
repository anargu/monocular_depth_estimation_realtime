from datasets.nyuv2 import NYUv2
from torchvision import transforms

def create_transforms():
    t = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomCrop(400)
    ])
    return t


if __name__ == '__main__':
    t = create_transforms()
    nyu_dataset = NYUv2(
        root='',
        train=True,
        download=False,
        depth_transform=t
    )