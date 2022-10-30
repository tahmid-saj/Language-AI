tribid_embedding_model_probs = tribid_embedding_model.predict(val_pos_char_token_dataset, verbose=1)
tribid_embedding_model_probs

tribid_embedding_model_preds = tf.argmax(tribid_embedding_model_probs, axis=1)
tribid_embedding_model_preds

tribid_embedding_model_results = calculate_results(y_true=val_labels_encoded, y_pred=tribid_embedding_model_preds)
tribid_embedding_model_results

tribid_embedding_model.save("tribid_embedding_model")

LOADED_MODEL_PATH = "tribid_embedding_model"

loaded_tribid_embedding_model = tf.keras.models.load_model(LOADED_MODEL_PATH)