from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_user_agents(benign_agents, exploit_agents):
    vectorizer = TfidfVectorizer()
    X_benign = vectorizer.fit_transform(benign_agents)
    X_exploit = vectorizer.transform(exploit_agents)
    return vectorizer, X_benign, X_exploit