from flask import Flask, request, jsonify
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import joblib
import numpy as np
import pandas as pd
from collections import Counter

# === Setup ===
app = Flask(__name__)
CORS(app)

# === Load Models ===
models = {
    "Logistic Regression": joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\logistic_with_metrics.pkl")["model"],
    "Tuned SVM": joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\svm_with_metrics.pkl")["model"],
    "Tuned KNN": joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\knn_Grid_with_metrics.pkl")["model"],
    "Random Forest": joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\rf_grid_with_metrics.pkl")["model"],
    "XGBoost": joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\xgb_with_metrics.pkl")["model"],
    "LightGBM": joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\lgbm_with_metrics.pkl")["model"],
    "Stacking Ensemble": joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\stacking_with_metrics.pkl")["model"],
}

# === Load Transformers ===
imputer = joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\imputer.pkl")
scaler = joblib.load("C:\\Users\\Mai Badr\\Graduation_Project\\scaler.pkl")

# === Descriptors Setup ===
descriptor_names = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
MACCS_FP_SIZE = 167

# === Helper Functions ===
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_names)
    return calculator.CalcDescriptors(mol)

def compute_maccs_fingerprint(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((MACCS_FP_SIZE,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr.tolist()

# === Prediction Route (For Index + Detailed) ===
@app.route("/predict_majority", methods=["POST"])
def predict_majority_and_models():
    data = request.json
    if "smiles" not in data:
        return jsonify({
            "canonical_smiles": None,
            "majority_vote": "Missing SMILES input",
            "model_predictions": {}
        }), 400

    input_smiles = data["smiles"]
    canonical_smiles = canonicalize_smiles(input_smiles)

    if canonical_smiles is None:
        return jsonify({
            "canonical_smiles": None,
            "majority_vote": "Invalid SMILES",
            "model_predictions": {}
        }), 400

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return jsonify({
            "canonical_smiles": canonical_smiles,
            "majority_vote": "Invalid Molecule",
            "model_predictions": {}
        }), 400

    try:
        # === Feature extraction
        descriptors = calc_descriptors(canonical_smiles)
        maccs_fp = compute_maccs_fingerprint(mol)
        input_features = list(descriptors) + list(maccs_fp)

        # === Create DataFrame
        feature_df = pd.DataFrame(
            [input_features],
            columns=descriptor_names + [f"MACCS_{i}" for i in range(MACCS_FP_SIZE)]
        )
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # === Preprocessing
        X_imputed = imputer.transform(feature_df)
        X_scaled = scaler.transform(X_imputed)

        # === Prediction
        model_predictions = {}
        votes = []
        for name, model in models.items():
            pred = int(model.predict(X_scaled)[0])
            model_predictions[name] = pred
            votes.append(pred)

        # === Majority Vote
        majority_vote = Counter(votes).most_common(1)[0][0]

        return jsonify({
            "canonical_smiles": canonical_smiles,
            "majority_vote": int(majority_vote),
            "model_predictions": model_predictions
        })

    except Exception as e:
        return jsonify({
            "canonical_smiles": canonical_smiles,
            "majority_vote": f"Error: {str(e)}",
            "model_predictions": {}
        }), 500

# === Start the server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

