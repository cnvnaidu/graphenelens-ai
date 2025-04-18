baseline_accuracy = 0.74  # Example accuracy for Baseline CNN, update with actual results
resnet_accuracy = 0.82    # Example accuracy for ResNet50 model, update with actual results
mlp_accuracy = 0.80       # Example accuracy for MLP, update with actual results

print(f"Baseline CNN Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"ResNet50 Accuracy: {resnet_accuracy * 100:.2f}%")
print(f"MLP Classifier Accuracy (PCA + SMOTE): {mlp_accuracy * 100:.2f}%")
