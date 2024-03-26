from itertools import combinations

import numpy as np
import torch


def pairwise_distances(vectors1, vectors2):
    """
    Compute the pairwise Euclidean distances between two sets of vectors
    """
    # Compute the dot product between the two sets of vectors
    dot_product = torch.mm(vectors1, vectors2.t())
    # Compute the squared norm of the vectors
    sum_of_squares1 = vectors1.pow(2).sum(dim=1).view(-1, 1)
    sum_of_squares2 = vectors2.pow(2).sum(dim=1).view(1, -1)
    # Compute the pairwise Euclidean distances
    distances = sum_of_squares1 + sum_of_squares2 - 2 * dot_product
    return distances


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, img_embeddings, text_embeddings):
        raise NotImplementedError


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    hard_negative = np.random.choice(hard_negatives) if len(hard_negatives) > 0 else np.argmax(loss_values)
    return hard_negative


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    semihard_negative = np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else np.argmax(loss_values)
    return semihard_negative


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    Negative function should be one of the following: hardest, random_hard, semihard. It should take array of loss values
    for a given anchor-positive pair and all negative samples and return a negative index for that pair.
    """

    def __init__(self, margin, negative_selection_fn, cpu=True, img_to_text=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.img_to_text = img_to_text

    def get_triplets(self, img_embeddings, text_embeddings):
        if self.cpu:
            img_embeddings = img_embeddings.cpu()
            text_embeddings = text_embeddings.cpu()
        # Compute the pairwise Euclidean distances between the image and text embeddings
        pairwise_dist = pairwise_distances(img_embeddings, text_embeddings)
        pairwise_dist = pairwise_dist.cpu()

        triplets = []

        if self.img_to_text:
            # We use the image embeddings as the anchor and the text embeddings as positive and negative
            for i in range(img_embeddings.size(0)):
                # Compute the triplet loss values
                ap_distance = pairwise_dist[i][i]
                loss_values = ap_distance - pairwise_dist[i] + self.margin
                loss_values[i] = 0
                loss_values = loss_values.cpu().data.numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = int(hard_negative)
                    triplets.append([i, i, hard_negative])
        else:
            # We use the text embeddings as the anchor and the image embeddings as positive and negative
            for i in range(text_embeddings.size(0)):
                # Compute the triplet loss values
                ap_distance = pairwise_dist[i][i]
                loss_values = ap_distance - pairwise_dist[:, i] + self.margin
                loss_values[i] = 0
                loss_values = loss_values.cpu().data.numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = int(hard_negative)
                    triplets.append([i, i, hard_negative])

        triplets = np.array(triplets)
        return torch.LongTensor(triplets)


