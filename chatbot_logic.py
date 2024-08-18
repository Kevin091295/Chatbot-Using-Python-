import numpy as np
from numpy.linalg import norm
from glove_embeddings import embeddings  # Importing the embeddings

# Sample FAQ database
faq_dict = {
    "how can i track my order": "You can track your order by logging into your account and clicking on 'Track Order'.",
    "what is the return policy": "You can return any item within 30 days of purchase.",
    "how do i contact customer service": "You can contact customer service by calling 1-800-555-1234.",
    "what payment methods are accepted": "We accept Visa, MasterCard, American Express, and PayPal.",
    "how long does shipping take": "Shipping typically takes 3-5 business days within the United States."
}

def find_similar_question(user_input, faq_dict, embeddings):
    user_vector = np.mean([embeddings[word] for word in user_input.split() if word in embeddings], axis=0)
    similarities = {}
    for question, answer in faq_dict.items():
        question_vector = np.mean([embeddings[word] for word in question.split() if word in embeddings], axis=0)
        similarity = np.dot(user_vector, question_vector) / (norm(user_vector) * norm(question_vector))
        similarities[question] = similarity
    return max(similarities, key=similarities.get)

def get_response(user_input, faq_dict, embeddings):
    similar_question = find_similar_question(user_input, faq_dict, embeddings)
    return faq_dict[similar_question]

# Example usage
if __name__ == "__main__":
    user_input = input("Ask a question: ")
    response = get_response(user_input, faq_dict, embeddings)
    print("Chatbot:", response)
