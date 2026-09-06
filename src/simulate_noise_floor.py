import numpy as np

def simulate_noise_floor(k, n_spots, n_simulations=1000):
    np.random.seed(42)
    eces = []
    for _ in range(n_simulations):
        # perfectly calibrated probabilities: uniform dirichlet for spots
        true_probs = np.random.dirichlet(np.ones(k), size=n_spots)
        
        # sample ground truth classes based on probabilities
        y_true = np.array([np.random.choice(k, p=p) for p in true_probs])
        
        # predicted classes (argmax)
        y_pred = np.argmax(true_probs, axis=1)
        confidences = np.max(true_probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        # calculate ECE (10 bins)
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidences, bins) - 1
        
        ece = 0
        for b in range(10):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                acc = np.mean(accuracies[mask])
                conf = np.mean(confidences[mask])
                weight = np.sum(mask) / n_spots
                ece += weight * np.abs(acc - conf)
                
        eces.append(ece)
        
    return np.mean(eces), np.std(eces)

print(f"k=23, n=500: {simulate_noise_floor(23, 500)}")
print(f"k=4, n=500: {simulate_noise_floor(4, 500)}")
