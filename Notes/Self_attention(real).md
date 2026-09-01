# Multi-Head Attention Matrix Operations (n Tokens)

Suppose the input contains **n tokens**.

---

## Step 1: Create Queries, Keys, and Values

```text
Input (Encoder or Decoder):    (n×512)

↓

Q = XWQ                        (n×64)

K = XWK                        (n×64)

V = XWV                        (n×64)
```

Each **row** is the Query, Key, or Value vector for one token.

---

## Step 2: Compute Attention Scores

Transpose the Key matrix:

```text
Kᵀ (64×n)
```

Now compute:

```text
Q (n×64)

×

Kᵀ (64×n)

↓

Attention Scores (n×n)
```

Each entry is:

```text
Scoreᵢⱼ = Qᵢ · Kⱼ
```

After scaling and softmax:

```text
Attention Weights (n×n)
```

Each **row** sums to **1**.

Example for **5 tokens**:

```text
          V₁   V₂   V₃   V₄   V₅
Q₁ →     0.7  0.1  0.1  0.0  0.1
Q₂ →     0.2  0.4  0.1  0.2  0.1
Q₃ →     ...
...
```

---

## Step 3: Apply the Values

The Value matrix is:

```text
V (n×64)
```

Each row is one Value vector:

```text
V₁
V₂
...
Vₙ
```

Now multiply:

```text
Attention Weights (n×n)

×

V (n×64)

↓

Output (n×64)
```

The inner dimensions match, so the multiplication is valid.

---

## What Does One Row Mean?

Take **one row** of the attention weights for example:

```text
0.1  0.3  0.5  0.0  0.1
```

Multiply by the Value matrix:

```text
0.1V₁
+
0.3V₂
+
0.5V₃
+
0.0V₄
+
0.1V₅

↓

Output₃ (1×64)
```

This is exactly the weighted-sum equation learned earlier.

The matrix multiplication simply computes this for **every Query simultaneously**:

## Why is the Output (1×64)?

The key is that **\(V_i\) is not a number—it is a 64-dimensional vector.**

Suppose:

```text
V₁ = [v₁₁, v₁₂, ..., v₁₆₄]
V₂ = [v₂₁, v₂₂, ..., v₂₆₄]
...
V₅ = [v₅₁, v₅₂, ..., v₅₆₄]
```

Each Value has shape:

```text
(1×64)
```

Now compute:

```text
0.1V₁
+
0.3V₂
+
0.5V₃
+
0.0V₄
+
0.1V₅
```

This really means:

```text
[0.1v₁₁, 0.1v₁₂, ..., 0.1v₁₆₄]

+

[0.3v₂₁, 0.3v₂₂, ..., 0.3v₂₆₄]

+

...

↓

[
0.1v₁₁ + 0.3v₂₁ + 0.5v₃₁ + 0.0v₄₁ + 0.1v₅₁,

0.1v₁₂ + 0.3v₂₂ + 0.5v₃₂ + 0.0v₄₂ + 0.1v₅₂,

...

0.1v₁₆₄ + 0.3v₂₆₄ + 0.5v₃₆₄ + 0.0v₄₆₄ + 0.1v₅₆₄
]
```

The result still has **64 components**, so its shape is:

```text
Output₃ (1×64)
```

---

## Intuition

The attention weights are simply saying:

- Take **10%** of **V₁**
- Take **30%** of **V₂**
- Take **50%** of **V₃**
- Take **0%** of **V₄**
- Take **10%** of **V₅**

Then add those **64-dimensional vectors** together.

Since you're adding vectors that are all **(1×64)**, the result is also **(1×64)**.

```text
Attention Weights (1×5)

↓

[0.1  0.3  0.5  0.0  0.1]

×

Value Matrix (5×64)

↓

Weighted average of the 5 Value vectors

↓

Output₃ (1×64)
```

So the attention weights determine **how much of each token's 64-dimensional Value vector contributes to the new 64-dimensional representation for token 3**.


```text
Output₁
Output₂
Output₃
...
Outputₙ
```

giving

```text
Output (n×64)
```

---

# Intuition Behind the Attention Output

The attention block produces a **new representation for every token** by taking a **weighted sum of all the Value vectors**, where the weights are determined by how strongly that token's **Query** matches every **Key**.

For example, for **token 3**:

```text
Token 3

↓

How related is token 3 to every token?

↓

Attention Weights

↓

Use those weights to mix together all Value vectors

↓

New representation for token 3
```

This process is repeated for **every token**:

```text
Token 1 → Weighted sum of all Value vectors with attention scores

Token 2 → Weighted sum of all Value vectors with attention scores

...

Token n → Weighted sum of all Value vectors with attention scores

(You always find the relation of each word for every token by multiplying each query by 
every key.)
```

giving the output:

```text
Output (n×64)
```

## Mental Model

Each token creates its **own custom weighted combination of all the Value vectors in the sentence**.

The weights are determined by the relationships (**Query–Key similarities**) between that token and every other token.

The result is a new representation for every token that contains information from the most relevant parts of the sentence.



## Complete Matrix Flow

```text
Input (n×512)

↓

Q = XWQ (n×64)

K = XWK (n×64)

V = XWV (n×64)

↓

QKᵀ

(n×64)

×

(64×n)

↓

Attention Scores (n×n)

↓

÷√64

↓

Softmax

↓

Attention Weights (n×n)

↓

Attention Weights × V

(n×n)

×

(n×64)

↓

Output of One Head (n×64)
```

---

## Multi-Head Output

```text
Head 1 Output   (n×64)

Head 2 Output   (n×64)

...

Head 8 Output   (n×64)

↓

Concatenate

↓

(n×512)

↓

Wᴼ (512×512) → Output Projection Matrix (learned matrix that mixes concatenated value matrix
into one final (1,512) representation)

↓

Final Output (n×512)
```

---

## Intuition

- **Rows = tokens** (there are **n** of them).
- **Columns = features** (64 per head, 512 after concatenating 8 heads).

Examples:

- **5 tokens → Output = (5×64)** per head.
- **10 tokens → Output = (10×64)** per head.
- **100 tokens → Output = (100×64)** per head.

Each **row** of the output is the updated representation for one token after attending to all the other tokens.

The Transformer computes all Query–Key dot products and all weighted sums of the Value vectors **simultaneously** using matrix multiplication instead of processing one token at a time.


---

## Why is projection Matrix a square?

```text
mathematically:

Attention Output (1×512)

×

Wᴼ (512×512)

↓

Projected Output (1×512)
```
