from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, _utils
from torchvision import transforms


class VOCdataloader(DataLoader):
    def __init__(self, vocdevkitpath, year, image_set, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        self.dataset = VOCDetection(
            vocdevkitpath, year, image_set, False, transform=transforms.ToTensor())
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                         pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)


if __name__ == "__main__":
    vocdata = VOCdataloader('./', '2007', 'train')
    for i, (image, xml) in enumerate(vocdata):
        print(i)
        break
