"""Download the published animal re-ID model once for offline service use."""

from model import load_model


if __name__ == "__main__":
    model, target, _ = load_model()
    print(f"MegaDescriptor is ready: {model.num_features} features, device {target}")
