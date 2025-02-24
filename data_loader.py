import pandas as pd

def load_data(benign_path: str, exploit_path: str):
    benign_df = pd.read_csv(benign_path, low_memory=False)
    exploit_df = pd.read_csv(exploit_path, low_memory=False)
    return benign_df, exploit_df