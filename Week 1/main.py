from dataset import MITSplitDataset

if __name__ == '__main__':
    data_dir = '../MIT_small_train_1_augmented/train/'
    dataset = MITSplitDataset(data_dir)

    print(f"Number of samples in the dataset: {len(dataset)}")

    img, label = dataset[0]
    print(f"Image shape: {img.size}, Label: {label}")
    img.show()
