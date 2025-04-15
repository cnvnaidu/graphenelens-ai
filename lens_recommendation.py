def recommend_lens(disease, severity):
    if disease == "cataract" and severity == "high":
        return "Graphene Lens A"
    elif disease == "diabetic_retinopathy" and severity == "medium":
        return "Graphene Lens B"
    elif disease == "glaucoma":
        return "Graphene Lens C"
    else:
        return "No lens needed"

# Example usage
print(recommend_lens("glaucoma", "high"))
