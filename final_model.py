from scipy.stats import randint, uniform
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score, 
    train_test_split
)
from load_disease_data import load_dataset
from sklearn.ensemble import RandomForestClassifier as RandomForest


def main() -> None:
    X, y = load_dataset(use_full_data=False)

    # Filter out diseases that occur less than 4 times for cross-fold validation
    disease_counts = y.value_counts()
    valid_diseases = disease_counts[disease_counts >= 4].index
    valid_mask = y.isin(valid_diseases)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Split into test and training
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create random forest model (default params)
    baseline_model = RandomForest(random_state=42)

    # 4 fold cross validation to evaluate baseline performance
    baseline_scores = cross_val_score(
        baseline_model, X_train, y_train, cv=4, scoring="accuracy", n_jobs=-1
    )
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    ci = baseline_std * 2
    print(f"Baseline accuracy: {baseline_mean:.4f} (+/- {ci:.4f})")

    # Grid for testing hyperparameters
    # Narrowed down to best params from original parameters after first tests
    grid_param_grid = {
        "n_estimators": [100],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [3, 7, None],
        "criterion": ["gini", "log_loss"]
    }
    # Create random forest model, ensure a fixed random state
    model = RandomForest(random_state=42)

    # Perform GridSearch CV
    grid_search = GridSearchCV(
        model,
        grid_param_grid,
        cv=4,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    # Fit grid search on the training data
    grid_search.fit(X_train, y_train)
    # Print best parameters, and accuracy, as well as improvement
    print("GridSearchCV completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    improv = grid_search.best_score_ - baseline_mean
    print(f"Improvement over baseline: {improv:.4f}")

    # Parameter distributions for RandomizedSearchCV
    random_param = {
        "n_estimators": randint(50, 200),
        "max_features": ["sqrt", "log2", None],
        "max_depth": [None, randint(5, 100)],
        "criterion": ["gini", "log_loss"],
        "min_samples_leaf": randint(2, 6)
    }

    # New model fr testing randomized search,
    # ensures both start from same baseline
    model_random = RandomForest(random_state=42)

    # Test 40 random parameter combinations
    random_search = RandomizedSearchCV(
        model_random,
        random_param,
        n_iter=20,
        cv=4,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    # Fit the randomized search on training data
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV completed!")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    improv = random_search.best_score_ - baseline_mean
    print(f"Improvement over baseline: {improv:.4f}")

    # Best results from all search methods
    print(f"Baseline accuracy:           {baseline_mean:.4f}")
    print(f"GridSearchCV best accuracy:  {grid_search.best_score_:.4f}")
    print(f"RandomizedSearchCV accuracy: {random_search.best_score_:.4f}")


if __name__ == "__main__":
    main()
