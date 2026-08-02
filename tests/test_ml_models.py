#test_ml_models.py

from src.tools.ml_models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
)

print("\n========== MODEL REGISTRY ==========\n")

print("Classification Models")

for model in CLASSIFICATION_MODELS:
    print("-", model)

print()

print("Regression Models")

for model in REGRESSION_MODELS:
    print("-", model)

assert len(CLASSIFICATION_MODELS) >= 8
assert len(REGRESSION_MODELS) >= 8

print("\nModel Registry: PASSED")