dense_model_obj = DenseModel(tokenizer=tokenizer)
dense_model = dense_model_obj.get_model()

lstm_model_obj = LSTMModel(tokenizer=tokenizer, vocab=vocab, mean_word_length=mean_word_length)
lstm_model = lstm_model_obj.get_model()

bidir_lstm_model_obj = BidirectionalLSTMModel(tokenizer=tokenizer, vocab=vocab, mean_word_length=mean_word_length)
bidir_lstm_model = bidir_lstm_model_obj.get_model()

ensemble_models = [dense_model, lstm_model, bidir_lstm_model]

ensemble_models = get_ensemble_models(models=ensemble_models, 
                                      train_data=train_text, 
                                      train_label=train_label, 
                                      val_data=(val_text, val_label), 
                                      num_iter=10, num_epochs=100)