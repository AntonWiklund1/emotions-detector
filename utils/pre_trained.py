def load_pretrained_weights(model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    # Filter out unnecessary keys
    matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # Update model's weights
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    # Calculate the number of matched weights
    num_matched = len(matched_dict)
    num_total = len(pretrained_dict)
    print(f"Number of matched weights: {num_matched}/{num_total}")

    return num_matched, num_total