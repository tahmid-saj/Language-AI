print(train_df['line_number'].value_counts())

train_df.line_number.plot.hist()

# One-hot encoding of line numbers

train_line_numbers_one_hot = tf.one_hot(train_df['line_number'].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df['line_number'].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df['line_number'].to_numpy(), depth=15)

print(train_line_numbers_one_hot.shape, train_line_numbers_one_hot[:20])

print(train_df['total_lines'].value_counts())

train_df.total_lines.plot.hist()

# One-hot encoding of total lines
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)

print(train_total_lines_one_hot.shape, train_total_lines_one_hot[:10])