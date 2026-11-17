from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
import json
import os

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 STARTING SALES FORECASTING TRAINING PIPELINE")
    print("=" * 60)

    print("\n>>> Stage 1: Data Ingestion")
    dataingestion = DataIngestion()
    dataingestion.load_dataset()
    print("✅ Stage 1 complete: Data ingested.")

    print("\n>>> Stage 2: Data Transformation & Feature Engineering")
    datatransformation = DataTransformation()
    datatransformation.integrate_data()
    datatransformation.split_data(number_of_test_days=15)
    datatransformation.transform_data()
    print("✅ Stage 2 complete: Datasets transformed & time series built.")

    print("\n>>> Stage 3: Model Training (LightGBM)")
    modeltrainer = ModelTrainer()
    modeltrainer.train_model()
    print("✅ Stage 3 complete: Models trained and saved.")

    print("\n>>> Stage 4: Model Evaluation")
    modelevaluation = ModelEvaluation()
    train, targets, predictions = modelevaluation.generate_predictions()
    modelevaluation.evaluate_predictions(train, targets, predictions)
    print("✅ Stage 4 complete: Evaluation finished.")

    results_path = os.path.join("artifacts", "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            metrics = json.load(f)
        print("\n" + "=" * 60)
        print("🏆 PIPELINE RUN FINISHED SUCCESSFULLY!")
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  - {k}: {v:.5f}")
            elif isinstance(v, dict):
                print(f"  - {k}:")
                for sub_k, sub_v in v.items():
                    print(f"      • {sub_k}: {sub_v}")
            else:
                print(f"  - {k}: {v}")
        print("=" * 60 + "\n")
