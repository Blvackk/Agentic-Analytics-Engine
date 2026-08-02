#ml_models.py

"""
Machine Learning Model Registry.

Contains all machine learning models
supported by the Agentic Analytics Engine.
"""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

from lightgbm import (
    LGBMClassifier,
    LGBMRegressor,
)

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)


CLASSIFICATION_MODELS = {

    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "decision_tree": DecisionTreeClassifier(
        random_state=42,
    ),

    "random_forest": RandomForestClassifier(
        random_state=42,
    ),

    "extra_trees": ExtraTreesClassifier(
        random_state=42,
    ),

    "gradient_boosting": GradientBoostingClassifier(
        random_state=42,
    ),

    "xgboost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),

    "lightgbm": LGBMClassifier(
        random_state=42,
        verbose=-1,
    ),

    "catboost": CatBoostClassifier(
        random_state=42,
        verbose=False,
    ),
}


REGRESSION_MODELS = {

    "linear_regression": LinearRegression(),

    "decision_tree_regressor": DecisionTreeRegressor(
        random_state=42,
    ),

    "random_forest_regressor": RandomForestRegressor(
        random_state=42,
    ),

    "extra_trees_regressor": ExtraTreesRegressor(
        random_state=42,
    ),

    "gradient_boosting_regressor": GradientBoostingRegressor(
        random_state=42,
    ),

    "xgboost_regressor": XGBRegressor(
        random_state=42,
        verbosity=0,
    ),

    "lightgbm_regressor": LGBMRegressor(
        random_state=42,
        verbose=-1,
    ),

    "catboost_regressor": CatBoostRegressor(
        random_state=42,
        verbose=False,
    ),
}


__all__ = [
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
]