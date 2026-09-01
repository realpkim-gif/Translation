# Attention in Encoder–Decoder Transformers

## Translation Example

**English input:** `I love cats`  
**French output:** `J’aime les chats`

An encoder–decoder Transformer uses three main attention operations:

1. Encoder self-attention
2. Masked decoder self-attention
3. Encoder–decoder cross-attention

---

## 1. Encoder Self-Attention

The encoder processes the complete English sentence:

`I | love | cats`

Each English token can attend to every other English token. This allows the encoder to understand relationships such as:

- `I` is the subject.
- `love` is the action.
- `cats` is the object.

Encoder self-attention is **not causally masked**, because the full source sentence is available before translation begins.

---

## 2. Masked Decoder Self-Attention

The decoder generates the French translation from left to right:

`J’ → aime → les → chats`

Each position may attend only to earlier French tokens, not future French tokens.

| Decoder input | Target token |
|---|---|
| `<START>` | `J’` |
| `<START> J’` | `aime` |
| `<START> J’ aime` | `les` |
| `<START> J’ aime les` | `chats` |

During training, the complete correct French sentence is available to the system. However, a **causal mask** prevents each position from seeing future French tokens.

For example, while predicting `aime`, the decoder can see:

- Earlier French tokens: `J’`
- The complete English sentence: `I love cats`

It cannot see:

- Future French tokens: `les chats`

During inference, future French tokens do not exist yet. The same causal generation rule therefore applies naturally.

---

## 3. Encoder–Decoder Cross-Attention (only during training)

Cross-attention connects the decoder to the encoder’s representation of the English sentence.

For example:

| French token being generated | Relevant English token |
|---|---|
| `J’` | `I` |
| `aime` | `love` |
| `les chats` | `cats` |

The decoder uses:

- **Queries** from its current French-side representations
- **Keys and values** from the English encoder output

Cross-attention can access the **entire English sentence** at every generation step. It does not need a causal mask over the English input because that sentence is already fully known.

The masked decoder state used as the cross-attention query still contains no information from future French tokens.

---

## Training vs. Inference

### During Training

The correct French translation is supplied using **teacher forcing**. All target positions can be processed in parallel, but causal masking ensures that each prediction uses only earlier target tokens.

### During Inference

The correct French translation is unavailable. The model generates one token at a time and feeds each generated token back into the decoder:

1. Generate `J’`
2. Use `J’` to generate `aime`
3. Use `J’aime` to generate `les`
4. Use `J’aime les` to generate `chats`

---

## Overall Information Flow

```text
English sentence: "I love cats"
        |
        v
Encoder self-attention
Understands the complete English sentence
        |
        v
English contextual representations
        |
        +--------------------------+
                                   |
Previous French tokens             |
        |                          |
        v                          |
Masked decoder self-attention      |
Cannot see future French tokens    |
        |                          |
        v                          v
        Encoder–decoder cross-attention
        Connects French context to English
                    |
                    v
             Next French token