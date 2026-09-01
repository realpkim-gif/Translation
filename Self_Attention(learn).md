# Multi-Head Self-Attention (Original Transformer)

## Tokens vs. Heads

Example:

```text
How are you doing today I hope you're feeling great
```

**10 tokens** → Encoder outputs:

```text
E₁ E₂ E₃ ... E₁₀

(each is 1×512)
```

Each head has its own learned matrices:

```text
WQ (512×64)
WK (512×64)
WV (512×64)
```

The **same matrices are applied to every token**:

```text
E₁ (1×512) → Q₁ (1×64)   K₁ (1×64)   V₁ (1×64)

E₂ (1×512) → Q₂ (1×64)   K₂ (1×64)   V₂ (1×64)

...

E₁₀ (1×512) → Q₁₀ (1×64)   K₁₀ (1×64)   V₁₀ (1×64)
```

So each head produces:

- 10 Queries
- 10 Keys
- 10 Values

This repeats independently for all 8 heads.

---

## Attention inside one head

For token 3:

```text
Q₃ (1×64)

↓

Q₃·K₁ᵀ (1×1)
Q₃·K₂ᵀ (1×1)
...
Q₃·K₁₀ᵀ (1×1)

↓

10 Attention Scores (10×1)

↓

÷√64 (10×1)

↓

Softmax (10×1)

↓

10 Attention Weights (10×1)

↓

Weight₁V₁ + Weight₂V₂ + ... + Weight₁₀V₁₀ (1×64)

↓

Output₃ (1×64)
```

Repeat for every Query, producing one **1×64 output per token**.

---

## Final combination

For each token:

```text
Head1 Output₃ (1×64)

Head2 Output₃ (1×64)

...

Head8 Output₃ (1×64)

↓

Concatenate

↓

(1×512)

↓

Wᴼ (512×512)
(learned output projection matrix)

↓

Final Output₃ (1×512)
```

### Mental model

> Every head processes every token. The same learned matrices (WQ), (WK), and (WV) are applied to every encoder vector. Each Query compares against every Key to produce attention weights, which combine the Values into a new 64-dimensional vector. The 8 head outputs are concatenated into a 512-dimensional vector, then the learned output projection matrix (WO) mixes information from all heads to produce the final 512-dimensional output.