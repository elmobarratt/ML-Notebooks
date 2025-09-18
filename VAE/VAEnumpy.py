#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Variational Autoencoder implemented with NumPy only.

Features
- Arbitrary input dimensionality
- Configurable encoder/decoder MLP sizes
- Reparameterisation trick
- MSE reconstruction loss + KL divergence regulariser
- Basic Adam optimizer implemented in NumPy
- Mini-batch training loop

Usage example at the bottom shows training on toy data.

Note: This is educational code prioritising clarity over speed.
"""

import numpy as np

# ------------------ utility layers & activations ------------------

def glorot_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))


def zeros(shape):
    return np.zeros(shape)


class Linear:
    def __init__(self, in_dim, out_dim, name=None):
        self.W = glorot_init(in_dim, out_dim)
        self.b = zeros((out_dim,))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.name = name or f"Linear_{id(self)}"

    def __call__(self, x):
        # x: (batch, in_dim)
        return x.dot(self.W) + self.b


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def grad(self, x, grad_out):
        return grad_out * (x > 0)


class Sigmoid:
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def grad_from_out(self, out, grad_out):
        return grad_out * (out * (1 - out))


# ------------------ simple Adam optimizer ------------------

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # maps param id -> (m, v)
        self.m = {id(p): np.zeros_like(p) for p in params}
        self.v = {id(p): np.zeros_like(p) for p in params}
        self.t = 0

    def step(self, grads):
        # grads: list of gradient arrays corresponding to params
        self.t += 1
        out = []
        for p, g in zip(self.params, grads):
            pid = id(p)
            m = self.m[pid]
            v = self.v[pid]
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p -= step
            out.append(step)
        return out


# ------------------ Variational Autoencoder (NumPy) ------------------

class VAE:
    def __init__(self, input_dim, latent_dim, hidden_dims=None, lr=1e-3, seed=None):
        """
        input_dim: integer, dimensionality of flattened input (e.g. 784 for 28x28 images)
        latent_dim: integer, size of latent vector z
        hidden_dims: list of ints for encoder MLP widths. Decoder will be symmetric by default.
        lr: learning rate for Adam
        """
        if seed is not None:
            np.random.seed(seed)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.hidden_dims = hidden_dims

        # Build encoder layers
        dims = [input_dim] + hidden_dims
        self.enc_linears = [Linear(dims[i], dims[i+1], name=f"enc_lin_{i}") for i in range(len(dims)-1)]
        self.enc_acts = [ReLU() for _ in range(len(self.enc_linears))]

        # final projection to mean and logvar
        last = dims[-1]
        self.fc_mu = Linear(last, latent_dim, name="fc_mu")
        self.fc_logvar = Linear(last, latent_dim, name="fc_logvar")

        # Build decoder (mirror)
        dec_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.dec_linears = [Linear(dec_dims[i], dec_dims[i+1], name=f"dec_lin_{i}") for i in range(len(dec_dims)-1)]
        self.dec_acts = [ReLU() for _ in range(len(self.dec_linears)-1)]  # final layer is output

        # Collect parameter references for optimizer
        self.params = []
        for layer in (self.enc_linears + [self.fc_mu, self.fc_logvar] + self.dec_linears):
            self.params.append(layer.W)
            self.params.append(layer.b)

        self.lr = lr
        self.opt = Adam(self.params, lr=lr)

        # placeholders for saved forward caches (for simple backprop)
        self._cache = {}

    # ---------- forward pass (encoder -> sample -> decoder) ----------
    def encode(self, x):
        """Return mu, logvar for input x (batch, input_dim)"""
        h = x
        self._cache['enc_h'] = [h]
        for lin, act in zip(self.enc_linears, self.enc_acts):
            h = lin(h)
            self._cache['enc_h'].append(h)
            h = act(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        self._cache['enc_final_h'] = h
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + eps * std
        self._cache['z'] = z
        self._cache['eps'] = eps
        self._cache['std'] = std
        return z

    def decode(self, z):
        h = z
        self._cache['dec_h'] = [h]
        for i, lin in enumerate(self.dec_linears[:-1]):
            h = lin(h)
            self._cache['dec_h'].append(h)
            h = self.dec_acts[i](h)
        # final layer -> reconstruct
        out = self.dec_linears[-1](h)
        # For continuous data we'll use identity output (reconstruction directly)
        recon = out
        return recon

    # ---------- losses ----------
    def reconstruction_loss(self, x, x_recon):
        # MSE per batch
        recon_loss = np.mean(np.sum((x - x_recon)**2, axis=1))
        return recon_loss

    def kl_divergence(self, mu, logvar):
        # KL(N(mu, sigma) || N(0,1)) closed form
        kld = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1)
        return np.mean(kld)

    # ---------- training helpers: compute gradients (manual backprop) ----------
    def _zero_grads(self):
        for p in self.params:
            p[...] = p  # noop to show we don't store grads on params themselves
        # We'll maintain separate grad lists matching self.params order

    def _backprop(self, x, x_recon, mu, logvar):
        # We'll compute gradients for all parameters and return as a list matching self.params
        batch_size = x.shape[0]

        # Grad of reconstruction loss (MSE) w.r.t x_recon: dLoss/dx_recon = 2*(x_recon - x)/batch_size
        d_recon = (2.0 * (x_recon - x)) / batch_size  # shape (B, input_dim)

        grads = []

        # Backprop through decoder linear layers
        # We need to compute gradients for decoder weights and biases.
        # We'll store intermediate activations from self._cache['dec_h'] which had pre-activation values.
        dec_linears = self.dec_linears
        dec_h_pre = self._cache['dec_h']  # list of pre-act values (including z as first)

        # gradient w.r.t output layer input
        # out = dec_linears[-1](h_last) = h_last.dot(W) + b; d_recon is gradient w.r.t out
        grad = d_recon  # gradient w.r.t out pre-activation (identity)

        # grads for final dec layer (W, b)
        last_lin = dec_linears[-1]
        h_last = dec_h_pre[-1]
        dW_last = h_last.T.dot(grad)
        db_last = np.sum(grad, axis=0)

        # gradient w.r.t h_last (input to final linear)
        dh = grad.dot(last_lin.W.T)

        grads_dec = []
        grads_dec.append(dW_last)
        grads_dec.append(db_last)

        # propagate through remaining decoder layers (in reverse)
        for i in range(len(dec_linears)-2, -1, -1):
            lin = dec_linears[i]
            pre = dec_h_pre[i]  # pre-activation input to this layer
            # after linear we applied ReLU
            dh = self.dec_acts[i].grad(pre, dh) if hasattr(self.dec_acts[i], 'grad') else dh * (pre > 0)
            dW = dec_h_pre[i].T.dot(dh)  # dec_h_pre[i] is the input to lin (for i==0 it's z)
            db = np.sum(dh, axis=0)
            grads_dec.insert(0, db)  # keep order consistent
            grads_dec.insert(0, dW)
            # propagate gradient to previous layer input
            dh = dh.dot(lin.W.T)

        # Now dh is gradient w.r.t z (input to decoder) shape (B, latent_dim)
        dz = dh

        # Backprop through reparameterization: z = mu + eps * std
        # dz/dmu = 1  => dL/dmu += dz
        # dz/dlogvar = dz * eps * 0.5 * exp(0.5 * logvar)
        eps = self._cache['eps']
        std = self._cache['std']
        dmu_from_decoder = np.mean(dz, axis=0)  # but we must propagate per-sample, so compute per sample
        # compute per-sample gradients
        dmu_per_sample = dz  # (B, latent_dim)
        dlogvar_per_sample = dz * eps * 0.5 * std  # chain through std = exp(0.5 * logvar)

        # Now gradients from KL term
        # KL per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # dKL/dmu = -mu
        # dKL/dlogvar = -0.5 * (1 - exp(logvar)) -> careful: derivative wrt logvar is -0.5*(1 - exp(logvar))
        # But KL is averaged across batch in loss; ensure scaling consistent
        batch = x.shape[0]
        dKL_dmu = -mu / batch  # shape (B, latent_dim)
        dKL_dlogvar = -0.5 * (1 - np.exp(logvar)) / batch  # shape (B, latent_dim)

        # Gradients of total loss w.r.t mu and logvar are combination
        # dL/dmu = dL_recon/dmu (from decoder path) + dKL/dmu
        dmu_total = dmu_per_sample + dKL_dmu
        dlogvar_total = dlogvar_per_sample + dKL_dlogvar

        # Backprop through encoder's final linear layers (fc_mu and fc_logvar)
        # fc_mu: mu = h_enc.dot(W_mu) + b_mu
        # fc_logvar similar
        h_enc = self._cache['enc_final_h']  # (B, hidden_dim)
        dW_mu = h_enc.T.dot(dmu_total)
        db_mu = np.sum(dmu_total, axis=0)
        dW_logvar = h_enc.T.dot(dlogvar_total)
        db_logvar = np.sum(dlogvar_total, axis=0)

        # gradient w.r.t h_enc from both branches
        dh_enc = dmu_total.dot(self.fc_mu.W.T) + dlogvar_total.dot(self.fc_logvar.W.T)

        # Backprop through encoder hidden layers (in reverse)
        enc_grads = []
        enc_h_pre = self._cache['enc_h']  # list of pre-activation values including input
        # enc_linears: maps enc_h_pre[i] -> pre_act (we stored pre-activation before applying activation)
        for i in range(len(self.enc_linears)-1, -1, -1):
            lin = self.enc_linears[i]
            pre = enc_h_pre[i+1]
            # apply ReLU grad
            dh_enc = self.enc_acts[i].grad(pre, dh_enc) if hasattr(self.enc_acts[i], 'grad') else dh_enc * (pre > 0)
            dW = enc_h_pre[i].T.dot(dh_enc)
            db = np.sum(dh_enc, axis=0)
            enc_grads.insert(0, db)
            enc_grads.insert(0, dW)
            dh_enc = dh_enc.dot(lin.W.T)

        # Now assemble gradients matching self.params order
        grads = []
        # encoder linears
        for dW, db in zip(enc_grads[::2], enc_grads[1::2]):
            grads.append(dW)
            grads.append(db)
        # fc_mu and fc_logvar
        grads.append(dW_mu)
        grads.append(db_mu)
        grads.append(dW_logvar)
        grads.append(db_logvar)
        # decoder linears grads were prepared earlier in grads_dec, but need to match W,b sequence
        # grads_dec currently is [dW_first, db_first, dW_second, db_second, ..., dW_last, db_last]
        grads.extend(grads_dec)

        # As a safety check, ensure grads length matches params length
        if len(grads) != len(self.params):
            raise ValueError(f"Gradient/param length mismatch: {len(grads)} vs {len(self.params)}")

        return grads

    def train_step(self, x_batch):
        # Forward
        mu, logvar = self.encode(x_batch)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        # Loss
        recon_loss = self.reconstruction_loss(x_batch, x_recon)
        kld = self.kl_divergence(mu, logvar)
        loss = recon_loss + kld

        # Backprop
        grads = self._backprop(x_batch, x_recon, mu, logvar)

        # Optimiser step
        self.opt.step(grads)

        return loss, recon_loss, kld

    def train(self, data, batch_size=64, epochs=10, shuffle=True, verbose=True):
        n = data.shape[0]
        for epoch in range(1, epochs+1):
            if shuffle:
                perm = np.random.permutation(n)
                data = data[perm]
            losses = []
            r_losses = []
            kls = []
            for i in range(0, n, batch_size):
                batch = data[i:i+batch_size]
                loss, rloss, kld = self.train_step(batch)
                losses.append(loss)
                r_losses.append(rloss)
                kls.append(kld)
            if verbose:
                print(f"Epoch {epoch}/{epochs}  loss={np.mean(losses):.6f}  recon={np.mean(r_losses):.6f}  kld={np.mean(kls):.6f}")

    # ---------- convenience functions ----------
    def encode_single(self, x):
        mu, logvar = self.encode(x[None, :])
        return mu[0], logvar[0]

    def sample(self, n=1):
        z = np.random.randn(n, self.latent_dim)
        recon = self.decode(z)
        return recon


# ------------------ Example usage ------------------
if __name__ == '__main__':
    # toy dataset: 2D points arranged in two gaussian blobs, then flattened to vector form
    N = 1000
    D = 8  # input dimension (arbitrary)
    np.random.seed(42)
    # create two clusters
    a = np.random.randn(N//2, D) * 0.3 + 2.0
    b = np.random.randn(N//2, D) * 0.2 - 2.0
    X = np.vstack([a, b]).astype(np.float32)

    vae = VAE(input_dim=D, latent_dim=2, hidden_dims=[64, 32], lr=1e-3, seed=0)
    print("Starting training on toy data...")
    vae.train(X, batch_size=64, epochs=50)

    # encode some points
    mu, logvar = vae.encode(X[:10])
    print("Encoded means (first 10):\n", mu)

    # sample from prior and decode
    samples = vae.sample(5)
    print("Decoded samples shape:", samples.shape)

    # simple reconstruction check
    z = vae.reparameterize(*vae.encode(X[:5]))
    recon = vae.decode(z)
    print("Reconstruction (first 5) diff norms:", np.linalg.norm(recon - X[:5], axis=1))
