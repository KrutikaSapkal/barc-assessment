from data_loader import load_data
from feature_extraction import vectorize_user_agents
from model_trainer import train_models
from moder_saver import save_model
from config.config import benign_path, exploit_path

def main():
    print(benign_path)
    print(exploit_path)
    benign_df, exploit_df = load_data(benign_path=benign_path, exploit_path=exploit_path)
    benign_agents = benign_df['userAgentString'].dropna().unique()
    exploit_agents = exploit_df['userAgentString'].dropna().unique()
    vectorizer, X_benign, X_exploit = vectorize_user_agents(benign_agents, exploit_agents)
    best_model, best_model_name = train_models(X_benign, X_exploit)
    save_model(best_model, vectorizer)


if __name__ == "__main__":
    main()

