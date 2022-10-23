# Tokenization

mean_word_length = int(mean([len(w.split()) for w in train_text]))

tokenizer = layers.TextVectorization(max_tokens=VOCAB_LENGTH, standardize="lower_and_strip_punctuation", split="whitespace", ngrams=None, output_mode="int", output_sequence_length=mean_word_length)
tokenizer.adapt(train_text)

vocab = tokenizer.get_vocabulary()
frequent_words = vocab[:10]
unfrequent_words = vocab[-10:]
print(f"Frequent words: {frequent_words}\n Unfrequent words: {unfrequent_words}")

# Dense model

class DenseModel():
    def __init__(self, tokenizer, n_units=1, activation="sigmoid"):
        self.tokenizer = tokenizer
        self.n_units = n_units
        self.activation = activation
    
    def get_model(self, model_name="dense_model"):
        dense_inputs = layers.Input(shape=(1, ), dtype="string")
        
        x = self.tokenizer(dense_inputs)
        dense_embedding = layers.Embedding(input_dim=len(vocab), 
                                           output_dim=128, 
                                           embeddings_initializer="uniform", 
                                           input_length=mean_word_length, 
                                           name="embedding")
        x = dense_embedding(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        dense_outputs = layers.Dense(units=self.n_units, activation=self.activation)(x)
        dense_model = tf.keras.Model(dense_inputs, dense_outputs, name=model_name)
        
        return dense_model

# LSTM model

class LSTMModel():
    def __init__(self, tokenizer, vocab, mean_word_length, embedding_output_dim=128, n_lstm_units=64, n_dense_units=1, activation="sigmoid"):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.mean_word_length = mean_word_length
        self.embedding_output_dim = embedding_output_dim
        self.n_lstm_units = n_lstm_units
        self.n_dense_units = n_dense_units
        self.activation = activation
        
        self.embedding_input_dim = len(vocab)
        
    def get_model(self, model_name="lstm_model"):
        tf.random.set_seed(42)
        
        lstm_embedding = layers.Embedding(input_dim=self.embedding_input_dim, 
                                          output_dim=self.embedding_output_dim, 
                                          embeddings_initializer="uniform",
                                          input_length=self.mean_word_length,
                                          name="embedding-lstm")
        
        lstm_inputs = layers.Input(shape=(1, ), dtype="string")
        x = self.tokenizer(lstm_inputs)
        x = lstm_embedding(x)
        x = layers.LSTM(units=self.n_lstm_units)(x)
        lstm_outputs = layers.Dense(units=self.n_dense_units, activation=self.activation)(x)
        lstm_model = tf.keras.Model(lstm_inputs, lstm_outputs, name=model_name)
        
        return lstm_model

# Bidirectional LSTM model

class BidirectionalLSTMModel():
    def __init__(self, tokenizer, vocab, mean_word_length, embedding_output_dim=128, n_lstm_units=64, n_dense_units=1, activation="sigmoid"):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.mean_word_length = mean_word_length
        self.embedding_output_dim = embedding_output_dim
        self.n_lstm_units = n_lstm_units
        self.n_dense_units = n_dense_units
        self.activation = activation
        
        self.embedding_input_dim = len(vocab)
        
    def get_model(self, model_name="bidirectional_lstm_model"):
        tf.random.set_seed(42)
        
        bidir_lstm_embedding = layers.Embedding(input_dim=self.embedding_input_dim,
                                               output_dim=self.embedding_output_dim,
                                               embeddings_initializer="uniform",
                                               input_length=self.mean_word_length,
                                               name="embedding-bidirectional-lstm")
        bidir_lstm_inputs = layers.Input(shape=(1, ), dtype="string")
        x = self.tokenizer(bidir_lstm_inputs)
        x = bidir_lstm_embedding(x)
        x = layers.Bidirectional(layers.LSTM(units=self.n_lstm_units))(x)
        bidir_lstm_outputs = layers.Dense(units=1, activation="sigmoid")(x)
        bidir_lstm_model = tf.keras.Model(bidir_lstm_inputs, bidir_lstm_outputs, name=model_name)
        
        return bidir_lstm_model

# Ensemble model

def get_ensemble_models(models, train_data, train_label, val_data, num_iter=10, num_epochs=100, loss_funcs=["binary_crossentropy"]):
    """
    Returns a list of num_iter models each trained on binary_crossentropy loss functions by default.
    
    For instance, if num_iter = 10, a list of 60 trained models will be returned.
    10 * len(loss_funcs) * len(models) = 60 
    
    Parameters
    ----------
    models: NLP models passed.
    train_data: Training text dataset before tokenization and embedding.
    train_label: Training label dataset.
    val_data: List of validation dataset before tokenization and embedding.
    """
    ensemble_models = []
    
    for n_iter in range(num_iter):
        for model in models:
            for loss_func in loss_funcs:
                print(f"Reducing: {loss_func} for epochs: {num_epochs}, num_iter: {n_iter}, model: {model.name}")
                
                model.compile(loss=loss_func, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                
                model.fit(train_data, train_label, epochs=num_epochs, verbose=2, validation_data=val_data,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])
                
                ensemble_models.append(model)
    
    return ensemble_models