comb_pred_probs = tf.squeeze(lstm_model_pred_probs, axis=1) + tf.squeeze(bidir_lstm_model_pred_probs, axis=1)
comb_pred_probs = tf.round(comb_pred_probs / 2)
print(comb_pred_probs, comb_pred_probs.shape)

ensemble_model_metrics = evaluate_preds(y_true=val_label, y_pred=comb_pred_probs)
print(f"Ensemble model metrics: {ensemble_model_metrics}")

test_text = test_df[["clean_text"]].to_list()
test_samples = random.sample(test_text, 10)

for test_sample in test_samples:
    pred = make_ensemble_preds(ensemble_models, test_samples)
    print(f"Pred: {int(pred)}")
    print(f"Text: \n{test_sample}\n\n")

print(f"Ensemble model 0: {ensemble_models[0]}")

ensemble_models[0].save("/models/ensemble_models/dense_model", save_format='tf')
ensemble_models[1].save("/models/ensemble_models/lstm_model", save_format='tf')
ensemble_models[2].save("/models/ensemble_models/bidir_lstm_model", save_format='tf')

loaded_ensemble_dense_model = tf.keras.models.load_model("/models/ensemble_models/dense_model")
loaded_ensemble_lstm_model = tf.keras.models.load_model("/models/ensemble_models/lstm_model")
loaded_ensemble_bidir_lstm_model = tf.keras.models.load_model("/models/ensemble_models/bidir_lstm_model")

print(loaded_ensemble_dense_model.summary(), "\n")
print(loaded_ensemble_lstm_model.summary(), "\n")
print(loaded_ensemble_bidir_lstm_model.summary(), "\n")

val_text = ""
val_label = 0

loaded_ensemble_models = [loaded_ensemble_dense_model, loaded_ensemble_lstm_model, loaded_ensemble_bidir_lstm_model]

loaded_ensemble_models.evaluate_preds(val_text, val_label)