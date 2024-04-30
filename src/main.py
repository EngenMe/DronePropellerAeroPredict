from data_loader import DataLoader
from random_forest_regression import RandomForestReg
from results import Results

def main():
    
    build_from_scratch = input("Do you want to build the model from scratch (Y/N)?")

    if build_from_scratch == "Y":
        # Load data and describe it
        dataloader = DataLoader()
        dataloader.data_description()

        # Get input keys from DataLoader
        input_keys = dataloader.input_keys

        # Train Decision Tree Regressor and evaluate error for CT
        random_forest = RandomForestReg(target_key='CT', input_keys=input_keys)
        random_forest.error()

        # Train Decision Tree Regressor and evaluate error for CP
        random_forest = RandomForestReg(target_key='CP', input_keys=input_keys)
        random_forest.error()

    # Plot results
    results = Results()
    results.plotter()

if __name__ == "__main__":
    main()
