"""Small positive conditional-mean networks; all preprocessing is fit-local."""
import time
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor
from .experiment import FEATURES


def positive(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Expected finite positive values; no clipping is permitted')
    return values


def prepare(training, rows):
    log_x = np.log(positive(training[FEATURES].to_numpy()))
    mean, std = log_x.mean(axis=0), log_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    scale = float(positive(training.target).mean())
    x = (log_x - mean) / std
    new_x = (np.log(positive(rows[FEATURES].to_numpy())) - mean) / std
    return x, positive(training.target.to_numpy()) / scale, new_x, {
        'log_feature_mean': mean.tolist(), 'log_feature_std': std.tolist(), 'target_scale': scale}


class VarianceNet(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.skip = torch.nn.Linear(3, 1)
        self.network = torch.nn.Sequential(torch.nn.Linear(3, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
        torch.nn.init.zeros_(self.skip.weight)
        torch.nn.init.zeros_(self.skip.bias)
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return (self.skip(x) + self.network(x)).squeeze(-1)


def qlike_objective(log_mean, scaled_target):
    return (log_mean + scaled_target * torch.exp(-log_mean)).mean()


def neural_forecast(training, rows, protocol, seed):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    x, y, new_x, metadata = prepare(training, rows)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    net = VarianceNet(protocol['hidden_width']).double()
    optimizer = torch.optim.AdamW(net.parameters(), lr=protocol['learning_rate'],
                                 weight_decay=protocol['weight_decay'])
    initial = None
    for _ in range(protocol['epochs']):
        optimizer.zero_grad()
        objective = qlike_objective(net(x), y)
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite neural training objective')
        if initial is None:
            initial = float(objective.detach())
        objective.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), protocol['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
    net.eval()
    with torch.no_grad():
        final = float(qlike_objective(net(x), y))
        values = positive(torch.exp(net(torch.from_numpy(new_x))).numpy() * metadata['target_scale'])
    if not np.isfinite(final):
        raise ValueError('Nonfinite final training loss')
    metadata.update(seed=seed, epochs=protocol['epochs'], initial_training_objective=initial,
                    final_training_objective=final,
                    parameters=sum(p.numel() for p in net.parameters()),
                    seconds=time.perf_counter()-started)
    return values, metadata


def glm_forecast(training, rows, protocol):
    started = time.perf_counter()
    x, y, new_x, metadata = prepare(training, rows)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = GammaRegressor(alpha=protocol['glm_alpha'], max_iter=2000, tol=1e-8).fit(x, y)
    values = positive(model.predict(new_x) * metadata['target_scale'])
    metadata.update(parameters=4, iterations=int(model.n_iter_), coefficients=model.coef_.tolist(),
                    intercept=float(model.intercept_), seconds=time.perf_counter()-started)
    return values, metadata
