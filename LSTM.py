import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random

# LSTM layer
# forward & bptt
class LSTMLayer:
    def __init__(self, in_dim, h_dim):
        self.h_dim = h_dim
        self.W = np.random.randn(4 * h_dim, in_dim + h_dim) * np.sqrt(2 / (in_dim + h_dim))
        self.b = np.zeros((4 * h_dim, 1))
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x_seq):
        h, c = np.zeros((self.h_dim, 1)), np.zeros((self.h_dim, 1))
        h_seq, cache = [], []
        for x in x_seq:
            combined = np.vstack((x, h))
            z = self.W @ combined + self.b
            i = 1 / (1 + np.exp(-np.clip(z[:self.h_dim],            -15, 15)))
            f = 1 / (1 + np.exp(-np.clip(z[self.h_dim:2*self.h_dim],-15, 15)))
            g = np.tanh(z[2*self.h_dim:3*self.h_dim])
            o = 1 / (1 + np.exp(-np.clip(z[3*self.h_dim:],          -15, 15)))
            c_next = f * c + i * g
            h_next = o * np.tanh(c_next)
            cache.append((combined, c, c_next, i, f, g, o))
            h, c = h_next, c_next
            h_seq.append(h)
        return h_seq, cache

    def backward(self, dh_seq, cache):
        dW, db = np.zeros_like(self.W), np.zeros_like(self.b)
        dx_seq = []
        dh_next = np.zeros((self.h_dim, 1))
        dc_next = np.zeros((self.h_dim, 1))
        for t in reversed(range(len(cache))):
            comb, c_prev, c_curr, i, f, g, o = cache[t]
            dh = dh_seq[t] + dh_next
            tanh_c = np.tanh(c_curr)
            dc = dh * o * (1 - tanh_c**2) + dc_next
            dz = np.vstack([
                (dc * g)      * (i * (1 - i)),
                (dc * c_prev) * (f * (1 - f)),
                (dc * i)      * (1 - g**2),
                (dh * tanh_c) * (o * (1 - o))
            ])
            dW += dz @ comb.T
            db += dz
            d_comb = self.W.T @ dz
            dx_seq.insert(0, d_comb[:d_comb.shape[0] - self.h_dim])
            dh_next = d_comb[d_comb.shape[0] - self.h_dim:]
            dc_next = f * dc
        return dx_seq, (dW, db)

# Construct one LSTM with training/testing
class SingleLSTM:
    """Single LSTM layer → linear output head."""
    def __init__(self, h_dim=10, lr=1e-3):
        self.lr  = lr
        self.h_dim = h_dim
        self.lstm = LSTMLayer(1, h_dim)

        # Output dense layer
        self.Wy = np.random.randn(1, h_dim) * np.sqrt(2 / h_dim)
        self.by = np.zeros((1, 1))
        self.mWy, self.vWy = np.zeros_like(self.Wy), np.zeros_like(self.Wy)
        self.mby, self.vby = np.zeros_like(self.by),  np.zeros_like(self.by)
        self.t = 0

    def _adam(self, p, g, m, v):
        np.clip(g, -1, 1, out=g)
        m[:] = 0.9   * m + 0.1   * g
        v[:] = 0.999 * v + 0.001 * (g ** 2)
        m_hat = m / (1 - 0.9   ** self.t)
        v_hat = v / (1 - 0.999 ** self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def train_step(self, x_seq, y_true):
        self.t += 1
        h_seq, cache = self.lstm.forward(x_seq)
        y_pred = self.Wy @ h_seq[-1] + self.by
        dy     = y_pred - y_true

        dWy = dy @ h_seq[-1].T
        dby = dy

        dh_in = [np.zeros((self.h_dim, 1)) for _ in h_seq]
        dh_in[-1] = self.Wy.T @ dy
        _, grads = self.lstm.backward(dh_in, cache)

        self._adam(self.Wy, dWy, self.mWy, self.vWy)
        self._adam(self.by, dby, self.mby, self.vby)
        self._adam(self.lstm.W, grads[0], self.lstm.mW, self.lstm.vW)
        self._adam(self.lstm.b, grads[1], self.lstm.mb, self.lstm.vb)
        return 0.5 * float(dy ** 2)

    def predict(self, x_seq):
        h_seq, _ = self.lstm.forward(x_seq)
        return (self.Wy @ h_seq[-1] + self.by).item()


# Helper Functions
WINDOW = 11   # 1 solar cycle lookback

def get_windows(data, window=WINDOW):
    X, y = [], []
    for i in range(len(data) - window):
        X.append([data[j].reshape(1, 1) for j in range(i, i + window)])
        y.append(data[i + window].reshape(1, 1))
    return X, y


def forecast_sequence(model, seed_window, n_steps, scaler):
    """
    Autoregressive forecast.
    seed_window : list of exactly WINDOW scaled values (each shape (1,1))
    Returns     : inverse-scaled predictions (n_steps,)
    """
    assert len(seed_window) == WINDOW, \
        f"seed_window must have exactly {WINDOW} points, got {len(seed_window)}"
    buf = list(seed_window)
    preds_scaled = []
    for _ in range(n_steps):
        win = [np.array(v).reshape(1, 1) for v in buf[-WINDOW:]]
        p = model.predict(win)
        preds_scaled.append(p)
        buf.append(np.array([p]))
    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))


# Load data
df = pd.read_csv("SN_y_tot_V2.0.csv", delimiter=';', header=None)
yearly_values = df[1].values.astype(np.float32).reshape(-1, 1)

#plt.plot(range(len(yearly_values)), yearly_values, label=f'All SNN', color='steelblue')
#plt.xlabel('Year #')
#plt.ylabel('SNN')
#plt.title('All Availible SNN Data')
#plt.legend(fontsize=8)

#plt.savefig('all_snn.png', dpi=150)
#plt.show()

# Train on first 80 % of the full record
train_idx   = int(len(yearly_values) * 0.8)
train_raw   = yearly_values[:train_idx]
scaler_main = MinMaxScaler()
train_scaled = scaler_main.fit_transform(train_raw)

X_train, y_train = get_windows(train_scaled)

# Train
N_TRIALS = 5
fig, axes = plt.subplots(N_TRIALS, 2, figsize=(14, 4 * N_TRIALS))

for trial in range(N_TRIALS):

    model = SingleLSTM(h_dim=10, lr=1e-3)
    loss = []
    
    for epoch in range(1000):
        epoch_loss = 0
        for i in np.random.permutation(len(X_train)):
            epoch_loss += model.train_step(X_train[i], y_train[i])
        #if (epoch + 1) % 100 == 0:
            #print(f"Epoch {epoch+1}/1000 | Loss: {epoch_loss/len(X_train):.6f}")
        loss.append(epoch_loss)
    
    # SSN Forecast
    # past-2008
    CUTOFF = 2008
    pre_raw    = df[df[0] <= CUTOFF][1].values.reshape(-1, 1)
    scaler_exp = MinMaxScaler()
    pre_scaled = scaler_exp.fit_transform(pre_raw)
    
    # Seed = last 11 known years (fixed length — controls for cycle-length confound)
    seed_control = [pre_scaled[j] for j in range(-WINDOW, 0)]
    control_pred = forecast_sequence(model, seed_control, n_steps=11, scaler=scaler_exp)
    
    actual_post = df[df[0] > CUTOFF][1].values[:11].reshape(-1, 1)
    ssr_control = float(np.sum((control_pred - actual_post) ** 2) / 11)
    print(f"SSR: {ssr_control:.4f}")

    # left column — loss curve (same for every trial)
    axes[trial, 0].plot(range(1000), loss, color='orange', label='LSTM Error')
    axes[trial, 0].set_xlabel('Epoch #')
    axes[trial, 0].set_ylabel('Loss')
    axes[trial, 0].set_title(f'Trial {trial+1} — Training Loss')
    axes[trial, 0].legend(fontsize=8)

    #if ssr_control < 550 and ssr_control > 450:
    # right column — forecast comparison
    axes[trial, 1].plot(range(11), actual_post, label='Actual SSN', linestyle='dashed', color='black')
    axes[trial, 1].plot(range(11), control_pred, label=f'Predicted SSN (SSR={ssr_control:.1f})', color='green')
    axes[trial, 1].set_xlabel('Year # of Solar Cycle 24')
    axes[trial, 1].set_ylabel('SSN')
    axes[trial, 1].set_title(f'Trial {trial+1} — Cycle 24 Forecast')
    axes[trial, 1].legend(fontsize=8)

        #break

plt.tight_layout()
plt.savefig('example.png', dpi=150)
plt.show()
