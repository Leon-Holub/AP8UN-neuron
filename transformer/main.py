import tensorflow as tf
import tensorflow_datasets as tfds

# Distribuční strategie (MirroredStrategy pro více GPU)
strategy = tf.distribute.MirroredStrategy()

# Dataset – wiki_dialog
(ds_train, ds_val), ds_info = tfds.load(
    "wiki_dialog",
    split=["train[:1%]", "train[1%:2%]"],
    with_info=True,
    data_dir="./data"
)

# Předzpracování
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=8000,
    output_mode='int',
    output_sequence_length=50
)

train_text = ds_train.map(lambda x, y: x)
tokenizer.adapt(train_text)

def preprocess(x, y):
    x = tokenizer(x)
    y = tokenizer(y)
    return (x, y)

train_ds = ds_train.map(preprocess).padded_batch(32)
val_ds = ds_val.map(preprocess).padded_batch(32)
