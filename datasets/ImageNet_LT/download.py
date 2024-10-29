
import os.path as osp
from tqdm import tqdm
import requests
import argparse
import os


def download_imagenet_whole(root):
    save_dir = f'./{root}/images'
    url_head = 'https://image-net.org/data/winter21_whole'

    os.makedirs(save_dir, exist_ok=True)
    
    def download_a_single_archive(url, desc=None):
        save_path = f'{save_dir}/{osp.basename(url)}'
        if osp.exists(save_path):
            return
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, \
            tqdm(desc=desc, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)


    with open('ImageNet21K_label.txt', 'r') as f:
        data = f.read().splitlines()
    n_cls = len(data)
    for i, wnid in enumerate(data):
        url = f'{url_head}/{wnid}.tar'
        desc = f'Downloading: {i+1}/{n_cls} {wnid}'
        download_a_single_archive(url, desc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--data_root_path', '--drp', default='./data/imagenet21k', help='data root path')
    args = parser.parse_args()

    download_imagenet_whole(args.data_root_path)
