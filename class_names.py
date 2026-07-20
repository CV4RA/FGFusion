"""
Class name lists for CLIP prototype generation.
Prompt template: "a photo of {name}"
"""

PASCAL_20 = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv monitor',
]

WIKI_10 = [
    'art and architecture', 'biology', 'geography and places', 'history',
    'literature and theatre', 'media', 'music', 'royalty and nobility',
    'sport and recreation', 'warfare',
]

# NUS-WIDE-10K: 10 largest concept classes (standard in cross-modal literature)
NUS_DEEP_10 = [
    'animal', 'buildings', 'clouds', 'flowers', 'grass',
    'lake', 'person', 'airplane', 'water', 'window',
]

DATASET_CLASS_NAMES = {
    'pascal': PASCAL_20,
    'wiki': WIKI_10,
    'nus_deep': NUS_DEEP_10,
}

DATASET_NUM_CLASSES = {
    'pascal': 20,
    'wiki': 10,
    'nus_deep': 10,
    'INRIA': 100,
    'xmedianet_deep': 200,
}


def get_class_names(dataset_name):
    if dataset_name in DATASET_CLASS_NAMES:
        return list(DATASET_CLASS_NAMES[dataset_name])

    n = DATASET_NUM_CLASSES.get(dataset_name)
    if n is None:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

    # Try loading from text file: datasets/{name}_class_names.txt (one name per line)
    import os
    txt_path = os.path.join('datasets', '{}_class_names.txt'.format(dataset_name))
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) >= n:
            return names[:n]

    # Try loading from .mat metadata
    names = _try_load_from_mat(dataset_name, n)
    if names is not None:
        return names

    # Fallback: generic prompts (weaker but enables CLIP distillation)
    return ['object category {}'.format(i) for i in range(n)]


def _try_load_from_mat(dataset_name, n):
    import os
    import scipy.io as sio

    mat_paths = {
        'INRIA': 'datasets/INRIA-Websearch.mat',
        'xmedianet_deep': 'datasets/XMediaNet5View_Doc2Vec.mat',
    }
    path = mat_paths.get(dataset_name)
    if path is None or not os.path.exists(path):
        return None

    try:
        data = sio.loadmat(path)
        for key in ('class_names', 'classes', 'categories', 'category_names', 'labels_name'):
            if key not in data:
                continue
            raw = data[key]
            names = []
            for item in raw.flat:
                if isinstance(item, str):
                    names.append(item)
                elif hasattr(item, '__len__') and len(item) > 0:
                    names.append(str(item[0]) if not isinstance(item[0], str) else item[0])
            if len(names) >= n:
                return names[:n]
    except Exception:
        pass
    return None
