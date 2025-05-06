import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# === Parametry ===
MAX_LEN = 80
VOCAB_SIZE = 12000
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 512
BATCH_SIZE = 32
EPOCHS = 6

# === Načtení tiny_shakespeare ===
dataset = tfds.load("tiny_shakespeare", split="train", data_dir="./data")
raw_text = next(iter(dataset))['text'].numpy().decode()

# === Rozdělení textu na kratší bloky ===
chunks = [raw_text[i:i + MAX_LEN] for i in range(0, len(raw_text) - MAX_LEN, MAX_LEN)]

# === TextVectorization ===
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_LEN,
    standardize=None,
    split="whitespace"
)
tokenizer.adapt(chunks)

# === Preprocessing ===
def preprocess(text):
    tokens = tokenizer(text)
    return (tokens[:-1], tokens[:-1]), tokens[1:]

train_ds = tf.data.Dataset.from_tensor_slices(chunks).map(preprocess).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = train_ds.take(5)

# === Pozicové kódování ===
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super().__init__()
        pos = np.arange(maxlen)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# === Transformer blok ===
def transformer_block(embed_dim, num_heads, ff_dim):
    inputs = tf.keras.Input(shape=(None, embed_dim))
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(embed_dim)
    ])
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn + ffn(attn))
    return tf.keras.Model(inputs, outputs)

# === Transformer model ===
def build_model():
    inp = tf.keras.Input(shape=(None,), dtype='int64')
    tgt = tf.keras.Input(shape=(None,), dtype='int64')
    embed = tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)
    x = embed(inp)
    y = embed(tgt)
    x = PositionalEncoding(MAX_LEN, EMBED_DIM)(x)
    y = PositionalEncoding(MAX_LEN, EMBED_DIM)(y)
    enc = transformer_block(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
    dec_attn = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM)(y, enc)
    dec_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(y + dec_attn)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(FF_DIM, activation='relu'),
        tf.keras.layers.Dense(VOCAB_SIZE)
    ])
    output = ffn(dec_out)
    return tf.keras.Model([inp, tgt], output)

# === Kompilace ===
model = build_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("🔁 Trénuji.")
model.fit(train_ds, epochs=EPOCHS)
print("✅ Trénink dokončen!")

# === Generace ===
vocab = tokenizer.get_vocabulary()
index_to_word = dict(enumerate(vocab))

def detokenize(tokens):
    return ' '.join([index_to_word.get(int(token), '[UNK]') for token in tokens])

def generate(prompt, max_len=20, temperature=0.8):
    input_ids = tokenizer(tf.convert_to_tensor([prompt]))[:, :MAX_LEN-1]
    output_ids = input_ids[:, :1]
    for _ in range(max_len):
        pred = model([input_ids, output_ids])
        logits = pred[:, -1, :] / temperature
        logits = tf.tensor_scatter_nd_update(
            logits,
            indices=tf.constant([[0, 0]]),  # Zakaž padding token (0)
            updates=tf.constant([-1e9])
        )
        next_token = tf.random.categorical(logits, num_samples=1, dtype=tf.int64)[:, 0]
        output_ids = tf.concat([output_ids, tf.expand_dims(next_token, 1)], axis=1)
        print("🔄 Generuji tokeny:", output_ids.numpy()[0])
    return detokenize(output_ids[0].numpy())

print("📝 Vygenerovaný text:", generate("to be or not", max_len=30))
