import torch
from torchdyn.datasets import ToyDataset

d = ToyDataset()
def generate_moon_data(n_samples=1000, noise=1e-1):
    X,yn = d.generate(n_samples=n_samples, dataset_type='moons', noise=noise)
    X = (X - X.mean()) / X.std()
    return X,yn

def generate_spiral_data(n_samples=10000, noise=1e-1):
    X,yn = d.generate(n_samples=n_samples, dataset_type='spirals', noise=noise)
    X = (X - X.mean()) / X.std()
    return (X,yn)

def generate_gaussians_data(n_samples=10000,  n_gaussians=6, ):
    X,yn = d.generate(
        n_samples=n_samples//n_gaussians, 
        dataset_type='gaussians', 
        n_gaussians = n_gaussians,
        std_gaussians=.5, dim=2, radius=4)
    X = (X - X.mean()) / X.std()
    return (X,yn)

def generate_spheres_data(n_samples=10000, noise=5e-2):
    X,yn = d.generate(n_samples=n_samples, dataset_type='spheres', noise=noise, dim=2)
    X = (X - X.mean()) / X.std()
    return (X,yn)
# def generate_halfmoon_data():