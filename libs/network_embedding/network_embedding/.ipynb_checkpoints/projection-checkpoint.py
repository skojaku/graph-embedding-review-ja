import numpy as np
from scipy import sparse
from scipy import optimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


def pca(vec, dim=1, **pca_params):
    return PCA(n_components=dim, **pca_params).fit_transform(vec)


def get_semi_space(
    vec,
    anchor_vec,
    labels,
    label_order=None,
    dim=1,
    mode="lda",
    centering=True,
    **params
):

    if label_order is None:
        class_labels, class_ids = np.unique(labels, return_inverse=True)
        n_class = len(class_labels)
    else:
        label2cids = {l: i for i, l in enumerate(label_order)}
        class_ids = np.array([label2cids[l] for l in labels])
        n_class = len(label2cids)

    if mode == "simple":
        left_center = np.mean(anchor_vec[class_ids == 0, :], axis=0)
        right_center = np.mean(anchor_vec[class_ids == 1, :], axis=0)
        vr = right_center - left_center
        denom = np.linalg.norm(vec, axis=1) * np.linalg.norm(vr)
        denom = 1 / np.maximum(denom, 1e-20)
        ret_vec = sparse.diags(denom) @ (vec @ vr.T)

        denom = np.linalg.norm(anchor_vec, axis=1) * np.linalg.norm(vr)
        denom = 1 / np.maximum(denom, 1e-20)
        anch_vec = sparse.diags(denom) @ (anchor_vec @ vr.T)
    elif mode == "lda":
        lda = LinearDiscriminantAnalysis(n_components=dim, **params)
        lda.fit(anchor_vec, class_ids)
        ret_vec = lda.transform(vec)
        anch_vec = lda.transform(anchor_vec)

    if centering:
        class_centers = np.zeros((n_class, dim))
        for cid in range(n_class):
            class_centers[cid, :] = (
                np.mean(anch_vec[class_ids == cid, :], axis=0)
                if dim > 1
                else np.mean(anch_vec[class_ids == cid, :])
            )
        ret_vec -= np.mean(class_centers, axis=0) if dim > 1 else np.mean(class_centers)
    return ret_vec


def save_semi_axis(filename, vec_all, anchor_points, labels):
    anchor_vec = vec_all[anchor_points, :]
    np.savez(filename, anchor_vec=anchor_vec, labels=labels)


def calc_semi_axis_from_file(
    filename, vec, label_order=None, dim=1, mode="simple", **params
):
    data = np.load(filename)
    anchor_vec = data["anchor_vec"]
    labels = data["labels"]
    return calc_semi_axis(
        vec, anchor_vec, labels, label_order=label_order, dim=dim, mode=mode, **params
    )
