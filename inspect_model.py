import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the model
with open('policy_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

print("Model loaded successfully.")
print(f"Model type: {type(decision_tree_model)}")

# Create large figure
plt.figure(figsize=(40, 20), dpi=300)

plot_tree(
    decision_tree_model,
    feature_names=None,
    class_names=None,
    filled=True,
    rounded=True,
    fontsize=10
)

# Save high resolution image
plt.savefig("decision_tree_high_quality.png", dpi=300, bbox_inches="tight")

print("High-quality decision tree saved as decision_tree_high_quality.png")