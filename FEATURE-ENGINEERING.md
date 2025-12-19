# Feature Engineering Pipeline Documentation

## Objective
Transform cleaned Titanic data into model-ready features using encoding, scaling,
feature generation, and selection.

---

## Encoding Strategy
- Sex: Binary encoding
- Embarked: One-hot encoding
- Pclass: Ordinal numeric

---

## Feature Generation
- FamilySize
- IsAlone
- FarePerPerson
- IsChild
- IsElderly
- IsUpperClass
- IsLowerClass
- FemaleUpperClass
- MaleLowerClass
- HighFare

These features capture social, economic, and demographic patterns.

---

## Scaling
- StandardScaler applied to numerical features
- Fitted only on training data to avoid leakage

---

## Feature Selection
- RandomForest feature importance
- Top features selected based on contribution

---

## Outputs
- X_train, X_test
- y_train, y_test
- Feature registry saved in JSON

---

## Design Principles
- Modular
- Reproducible
- No data leakage
- Interview-ready
