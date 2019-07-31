import numpy as np

def ortho_init(shape, scale=1.0):
    shape = tuple(shape)
    if len(shape) == 2:
        flat_shape = shape
    else:
        raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

def relu(x):
    return np.float32(x > 0.0) * x + np.float32(x <= 0.0) * 0.0

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    #print('sigmoid',np.max(x),np.min(x))
    return 1.0 / (1.0 + np.exp(-x))

class fclayer(object):
    def __init__(self, nin, nout, act=lambda x:x, init_scale=1.0):
        init_scale /= np.sqrt(nin)
        self.w = ortho_init((nin, nout), scale=init_scale)
        self.b = np.zeros(nout)
        self.act = act
        self.nin = nin
        self.nout = nout

    def get_weights(self):
        return np.concatenate([self.w.flatten(), self.b])

    def __call__(self, x):
        #print(x.shape, self.b.shape, self.w.shape)
        # forward pass
        #print(x.shape, self.b.shape, self.w.shape)
        return self.act(self.b + np.dot(x, self.w))

    def update_weights(self, weights):
        #print(weights.shape, self.nin * self.nout, self.nout)
        assert weights.size == self.nin * self.nout + self.nout
        self.w = np.reshape(weights[:self.nin * self.nout].copy(), [self.nin, self.nout])
        self.b = weights[self.nin * self.nout:].copy()

class lstmlayer(object):
    def __init__(self, nin, nh, init_scale=1.0):
        self.wx = ortho_init((nin, nh*4), scale=init_scale/np.sqrt(nin))
        self.wb = ortho_init((nh, nh*4), scale=init_scale/np.sqrt(nh))
        self.b = np.zeros(nh*4)
        self.nin = nin
        self.nh = nh

    def get_weights(self):
        return np.concatenate([self.wx.flatten(), self.wb.flatten(), self.b])

    def __call__(self, s, x):
        #print(x.max(), x.min())
        c, h = np.split(s, 2, axis=-1)
        #print(c.shape, h.shape)
        z = np.dot(x, self.wx) + np.dot(h, self.wb) + self.b
        #print(z.shape)
        i, f, o, u = np.split(z, 4, axis=-1)
        i = sigmoid(i)
        f = sigmoid(f)
        o = sigmoid(o)
        u = tanh(u)
        c = f*c + i*u
        h = o*tanh(c)
        # combine s
        snew = np.concatenate([c, h], axis=-1)
        return h, snew

    def update_weights(self, weights):
        assert weights.size == self.nin * self.nh*4 + self.nh * self.nh*4 + self.nh*4
        self.wx = weights[:self.nin * self.nh*4].reshape(self.nin, self.nh*4)
        self.wb = weights[self.nin * self.nh*4:self.nin * self.nh*4 + self.nh * self.nh*4].reshape(self.nh, self.nh*4)
        self.b = weights[self.nin * self.nh*4 + self.nh * self.nh*4:self.nin * self.nh*4 + self.nh * self.nh*4 + self.nh*4]
