### **Cross-Attention Mechanism**

* **Step 1:** The decoder already has representations of the translated words so far (after masked self-attention).
* **Step 2:** Those decoder vectors become **Queries**.
* **Step 3:** The encoder outputs become **Keys** and **Values**.
* **Step 4:** Each decoder **Query** takes the dot product with every encoder **Key**.
* **Step 5:** **Softmax** converts the scores into attention weights.
* **Step 6:** Those weights are applied to the encoder **Values**.
* **Step 7:** The weighted **Values** are summed to produce one new vector for that decoder token.



```text
Encoder
───────────────

"The"

"cat"

"sat"

↓

Keys
Values

───────────────
Decoder


Query₂ (decoder query)

↓

Compare with Keys from encoder (dot product)

↓

Attention Weights

↓

Weighted Sum of Values (1,n) * attention weights (n,64)

↓

Cross-Attention Output₂ (1,64) usually per head

↓

Concatenate heads

↓

Wᴼ

↓

Residual + LayerNorm

↓

Feed-Forward Network


↓

Linear Layer

↓

Softmax

↓

Predict next word ("chat")


```



# Decoder Output After Cross-Attention
```text
Cross-Attention Output
+
Decoder Input
↓
Residual + LayerNorm
```

Next, the MLP processes each decoder token separately:

```text
(1×512)

↓

Linear: 512 → 2048

↓

ReLU

↓

Linear: 2048 → 512

↓

MLP Output
(1×512)
```

The MLP refines the combined information from:

- 1. previous decoder tokens before decoder's value matrix output (residual + norm)
- 2. relevant encoder information -- value matrix output of decoder, (residual + nrom)
- 3. grammar and deeper contextual meaning (MLP)


i & ii: 
Information about previously generated words + Relevant information from the encoder from residual + norm


- 4. Another residual connection (strengthen i,ii)
```text
The MLP output is added to its input:

Combined Representation
(1×512)

+

MLP Output
(1×512)

↓

Final Decoder-Layer Output
(1×512)
```

### After all of this final Decoder has all the data to predict the next word
Its pretty much a vector that has the data for all the complex meaning and attributes the next word needs
```text
Final Decoder Vector
(1×512)

×

Vocabulary Matrix
(512×Vocabulary Size)

↓ (dot product gives how similar it is to the final decoder vector)

One score for every vocabulary token **(logit)**

↓

Softmax

↓

Next-token probabilities

```

#### Example:
```
chat     0.91
chien    0.03
maison   0.01
...
```

Predict probablilty