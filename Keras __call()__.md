## Keras `__call__()` vs. `call()`

A Keras model can be used like a function:

```python
predictions = transformer(inputs)
```

Python interprets this as:

```python
predictions = transformer.__call__(inputs)
```

Keras provides the `__call__()` method. It manages the model execution and then runs the `call()` method defined in the model class:

```text
transformer(...)
        ↓
Keras __call__()
        ↓
your call(...)
        ↓
predictions
```

For example:

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)
```

Use the model like this:

```python
outputs = model(inputs, training=True)
```

Conceptually, Keras performs something similar to:

```python
def __call__(self, inputs, **kwargs):
    validate_inputs(inputs)
    build_model_if_needed(inputs)
    outputs = self.call(inputs, **kwargs)
    track_model_information()
    return outputs
```

This is a simplified illustration. Keras's actual implementation is more complex.

The two methods have different jobs:

```text
__call__() → supplied by Keras; manages model execution
call()     → written by you; defines the model's forward pass
```

Keras's `__call__()` helps with:

- Input handling and validation
- Creating model weights when needed
- Training and inference behavior
- Mask handling
- Tracking model losses and updates
- Calling your custom `call()` method

Therefore, use:

```python
predictions = transformer(
    encoder_input,
    decoder_input,
    training=True,
    enc_padding_mask=enc_padding_mask,
    look_ahead_mask=combined_mask,
    dec_padding_mask=dec_padding_mask
)
```

Usually, do not call `call()` directly:

```python
predictions = transformer.call(...)
```

Calling `.call()` directly can bypass some of Keras's model-management behavior.

The shorthand works because Keras models are callable objects—not because the model has only one method.
