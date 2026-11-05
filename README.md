# California Housing Price Prediction
 
An end-to-end machine learning project that predicts the **median house value of a California district** from census data. It covers data loading, stratified splitting, feature engineering, a scikit-learn pipeline, model comparison, hyperparameter tuning, and final evaluation.
 
> **Status:** learning project / prototype. Not production-ready (see [Limitations](#limitations)).
 
## Dataset
 
California Housing (1990 census), 20,640 districts, 9 features plus the target.
 
| Feature | Description |
|---|---|
| `longitude`, `latitude` | District location |
| `housing_median_age` | Median age of houses |
| `total_rooms`, `total_bedrooms` | Totals in the district |
| `population`, `households` | District population and household count |
| `median_income` | Median income (in tens of thousands of USD) |
| `ocean_proximity` | Categorical: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` |
| **`median_house_value`** | **Target (USD)** |
 
Source: https://github.com/ageron/data (downloaded automatically by the notebook).
 
## Approach
 
1. **Split:** 80/20 stratified split on income category (`StratifiedShuffleSplit`, `random_state=42`) so the test set mirrors the income distribution.
2. **Feature engineering:**
   - Ratios: bedrooms/rooms, rooms/household, people/household
   - Log transform on skewed features
   - Custom `ClusterSimilarity` transformer (KMeans on lat/long plus RBF similarity)
   - One-hot encoding for `ocean_proximity`
3. **Pipeline:** everything lives in one `ColumnTransformer` + `Pipeline`, so preprocessing is fit only on training data (no leakage).
4. **Models compared (10-fold CV, RMSE):**
   | Model | RMSE (USD) |
   |---|---|
   | Linear Regression (training error) | ~68,973 |
   | Decision Tree (CV) | ~66,091 |
   | Random Forest (CV) | ~47,210 |
5. **Tuning:** `RandomizedSearchCV` (10 iterations, 3-fold) over `n_clusters` and `max_features`.
6. **Final test RMSE: ~41,493 USD** on the held-out test set.
## Project structure
 
```
.
├── main.ipynb          # full workflow
├── datasets/           # downloaded automatically
├── requirements.txt
└── README.md
```
 
## Setup
 
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook main.ipynb
```
 
Requirements: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `joblib`, `jupyter`.
 
## Usage
 
The trained pipeline is saved with joblib. The custom `ClusterSimilarity` class must be importable when loading.
 
```python
import joblib
model = joblib.load("model.pkl")
model.predict(new_districts_df)   # DataFrame with the original 9 feature columns
```
 
## Limitations
 
- **Old data:** 1990 census. Prices do not reflect today's market.
- **District-level, not house-level:** it predicts a district's median value, not a single home's price.
- **Capped target:** values are capped at $500k in the dataset, so high-end predictions are unreliable.
- **Error size:** ~41k USD RMSE is large relative to typical values.
- **California only:** do not use for other regions.
## Next steps
 
- Extract the pipeline and custom transformers into a `src/` module
- Try gradient boosting (HistGradientBoosting, XGBoost, LightGBM)
- Wider hyperparameter search
- Add an inference API (FastAPI) and tests
- Add monitoring for data drift
## License
 
MIT