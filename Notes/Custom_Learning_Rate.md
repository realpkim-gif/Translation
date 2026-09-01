# Transformer Decoder Output and Learning Rate Schedule

## Decoder Output: All Vectors vs Last Vector

In a Transformer, the decoder returns one output vector for each target-token position.

If

$$
\text{dec output} \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

then:

- $B$ is the batch size
- $T$ is the target sequence length
- $d_{\text{model}}$ is the model dimension

So `dec_output` contains all final decoder-layer vectors:

$$
h_1, h_2, \dots, h_T
$$

It is not just the last vector.

---

## Vocabulary Projection

The final dense layer projects decoder vectors into vocabulary space:

```python
final_output = self.final_layer(dec_output)
```

Mathematically:

$$
Z = H W_{\text{vocab}} + b
$$

where:

$$
H \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

$$
W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times V}
$$

So:

$$
Z \in \mathbb{R}^{B \times T \times V}
$$

This means every decoder vector is multiplied by the vocabulary matrix, producing one vocabulary distribution per target position.

After softmax:

$$
P(y_t \mid y_{\lt t}, x) = \text{softmax}(Z_t)
$$

---

## During Training: Use All Decoder Vectors

During training, the target sentence is shifted.

Example:

```text
decoder input:  <sos> je suis content
target output:  je    suis content <eos>
```

The model predicts every next token in parallel:

```text
vector for <sos>     -> predict "je"
vector for "je"      -> predict "suis"
vector for "suis"    -> predict "content"
vector for "content" -> predict "<eos>"
```

The loss uses all positions:

$$
\mathcal{L}
= -\sum_{t=1}^{T}
\log P(y_t \mid y_{\lt t}, x)
$$

So during training, we use:

```python
final_output = self.final_layer(dec_output)
```

with the full shape:

$$
B \times T \times d_{\text{model}}
\rightarrow
B \times T \times V
$$

If we only used the last decoder vector during training, the model would get only one prediction per sentence and would throw away most of the learning signal.

---

## During Inference: Use Only the Last Vector

During inference, we generate one token at a time.

If the current generated prefix is:

```text
<sos> je suis
```

then we only need the next token after `"suis"`.

So we take the last position:

```python
next_token_logits = final_output[:, -1, :]
```

Mathematically:

$$
Z_{\text{last}}
= h_T W_{\text{vocab}} + b
$$

and:

$$
P(y_{T+1} \mid y_{\le T}, x)
= \text{softmax}(Z_{\text{last}})
$$

Summary:

```text
training  -> use all decoder output vectors
inference -> use only the last decoder output vector
```

---

## Encoder Output vs Vocabulary Projection

The decoder does not directly use the encoder output to predict vocabulary tokens.

There are two separate dot-product operations:

```text
decoder vectors with encoder output -> cross-attention
decoder vectors with vocab matrix   -> word prediction
```

Flow:

```text
encoder input
  -> encoder output

decoder input
  -> masked self-attention
  -> cross-attention with encoder output
  -> feed-forward
  -> decoder output vectors
  -> vocab projection
  -> softmax
```

The encoder output influences the decoder through cross-attention. Then the final decoder vectors are projected into vocabulary space.

---

# Transformer Learning Rate Schedule

## Formula

The Transformer learning rate schedule is:

$$
LR(s)
= d_{\text{model}}^{-\frac{1}{2}}
\min
\left(
s^{-\frac{1}{2}},
s \cdot w^{-\frac{3}{2}}
\right)
$$

where:

$$
s = \text{step}
$$

$$
w = \text{warmup steps}
$$

$$
d_{\text{model}} = \text{model dimension}
$$

Equivalently:

$$
LR(s)
= \frac{1}{\sqrt{d_{\text{model}}}}
\min
\left(
\frac{1}{\sqrt{s}},
\frac{s}{w^{3/2}}
\right)
$$

---

## Warmup Phase

For:

$$
s \le w
$$

the learning rate uses the linear warmup term:

$$
LR(s)
= \frac{1}{\sqrt d}
\cdot
\frac{s}{w^{3/2}}
$$

So:

$$
LR(s)
= \frac{s}{\sqrt d \, w^{3/2}}
$$

Its derivative is:

$$
\frac{d}{ds}LR(s)
= \frac{1}{\sqrt d \, w^{3/2}}
$$

This is constant, meaning the learning rate increases linearly during warmup.

---

## Peak Learning Rate

At:

$$
s = w
$$

the learning rate reaches its maximum:

$$
LR(w)
= \frac{1}{\sqrt d}
\cdot
\frac{w}{w^{3/2}}
$$

Since:

$$
\frac{w}{w^{3/2}}
= \frac{1}{\sqrt w}
$$

we get:

$$
LR_{\max}
= \frac{1}{\sqrt{dw}}
$$

---

## Decay Phase

For:

$$
s \ge w
$$

the schedule uses inverse-square-root decay:

$$
LR(s)
= \frac{1}{\sqrt d}
\cdot
\frac{1}{\sqrt s}
$$

So:

$$
LR(s)
= \frac{1}{\sqrt{ds}}
$$

The derivative is:

$$
\frac{d}{ds}LR(s)
= -\frac{1}{2\sqrt d \, s^{3/2}}
$$

Since this derivative is negative, the learning rate decreases after warmup.

Because the denominator contains $s^{3/2}$, the magnitude of the derivative gets smaller over time. So the curve gradually flattens.

---

## Piecewise Form

The whole schedule can be written as:

$$
LR(s)
= \begin{cases}
\frac{s}{\sqrt d \, w^{3/2}}, & s \le w \\[8pt]
\frac{1}{\sqrt{ds}}, & s \ge w
\end{cases}
$$

So:

```text
before warmup_steps -> learning rate rises linearly
at warmup_steps     -> learning rate is maximum
after warmup_steps  -> learning rate decays like 1 / sqrt(step)
```

---

## Comparing the Derivatives

During warmup:

$$
\left|
\frac{d}{ds}LR(s)
\right|
= \frac{1}{\sqrt d \, w^{3/2}}
$$

After warmup:

$$
\left|
\frac{d}{ds}LR(s)
\right|
= \frac{1}{2\sqrt d \, s^{3/2}}
$$

At $s=w$:

$$
\left|
\frac{d}{ds}LR(w)
\right|_{\text{after}}
= \frac{1}{2\sqrt d \, w^{3/2}}
$$

while:

$$
\left|
\frac{d}{ds}LR(w)
\right|_{\text{before}}
= \frac{1}{\sqrt d \, w^{3/2}}
$$

Therefore:

$$
\left|
\frac{d}{ds}LR(w)
\right|_{\text{after}}
= \frac{1}{2}
\left|
\frac{d}{ds}LR(w)
\right|_{\text{before}}
$$

So the inverse-square-root decay changes more slowly than the linear warmup, and it gets flatter as $s$ increases.

---

## Intuition

The schedule does not start with a large or erratic learning rate.

Instead:

```text
small cautious steps
-> linear warmup to a peak
-> gradual inverse-square-root decay
```

Warmup helps prevent unstable early updates when the model weights are random. Decay later helps the model refine its parameters with smaller updates.