import numpy as np

def compute_stats(x):
	print('mean',np.mean(x),'max',np.max(x),'min',np.min(x),'std',np.std(x))

def uniformsimplex(n):
	# draw uniformly from a simplex of dimension n
	u = np.random.rand(n)
	e = -np.log(u)
	s = np.sum(e)
	return e / s

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed