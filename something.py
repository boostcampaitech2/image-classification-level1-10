from torch.utils.data import Dataset
from torchvision import transforms

class Subset_check(Dataset) :
    def __init__(self, dataset, indices, transform = None) :
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx) :
        image_tensor, labels = self.dataset[self.indices[idx]]
        image = transforms.ToPILImage()(image_tensor)
        image = self.transform(image)
        return image, labels
        # self.dataset[self.indices[idx]][0] = self.transform(self.dataset[self.indices[idx]][0])
        # return self.dataset[self.indices[idx]] 

    def set_transform(self, transform) :
        self.transform = transform

    def __len__(self) :
        return len(self.indices)

        