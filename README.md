# Logistic Neural Network
Another yet simple neural network implementation written in Python with NumPy.

## Getting Started
 * Package comes with pre-trained models. You can train your own models, or can try out pre-trained models. (see #Demo)

### Dataset Format
`FCNN` class expects inputs as `(m, n)` shaped matrix (NumPy array) where `m` is data count and `n` is the feature count.

For labels it expects a `(1, m)` matrix.

For example, that is the dataset for `and` model;
```python
logic_and = {
    "x": np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]),
    "y": np.array([[0, 0, 0, 1]])
}
```

### Initializing Network
You can set number of input features and units of hidden layer in constructor.
```python
nn = FCNN(input_features=2, hidden_layer_units=4)
```

### Training
After initialized the network, one can train network with given iteration limit and learning rate.
```python
nn.train(x, y, iterations=10 ** 5, learning_rate=0.01)
```

### Saving & Loading a Model
Before or after training, one can save current model to `*.csv` files. Currently model saving does not have much formatting, but it is a planned feature.

Both `dump_model` and `load_model` methods requires a file path.
#### Saving Model
```python
nn.dump_model(os.path.join(MODEL_DIR, model["name"]))
```
#### Loading / Retrieving a Model
Loading model will automatically set number of input features and number of hidden layer units.
```python
nn.load_model(os.path.join(MODEL_DIR, model["name"]))
```
 
## Demo

### Usage
`main.py` can be used for just trying out or learning how to use `FCNN` class.

#### Running the Demo

```
main.py train|evaluate
```

### Results from Pretrained Models
I prepared 5 models for basic logical operators.

#### NOT
```
Input   Expected        Guessed
[0]     1               1.0 (0.9995)
[1]     0               0.0 (0.0003)
```
#### XOR
```
Input   Expected        Guessed
[0 0]   0               0.0 (0.4995)
[0 1]   1               1.0 (0.5005)
[1 0]   1               1.0 (0.5005)
[1 1]   0               0.0 (0.4995)
```
#### AND
```
Input   Expected        Guessed
[0 0]   0               0.0 (0.0000)
[0 1]   0               0.0 (0.0015)
[1 0]   0               0.0 (0.0015)
[1 1]   1               1.0 (0.9979)
```
#### OR
```
Input   Expected        Guessed
[0 0]   0               0.0 (0.0009)
[0 1]   1               1.0 (0.9997)
[1 0]   1               1.0 (0.9997)
[1 1]   1               1.0 (1.0000)
```
#### IMPLIES
```
Input   Expected        Guessed
[0 0]   1               1.0 (0.9994)
[0 1]   1               1.0 (1.0000)
[1 0]   0               0.0 (0.0007)
[1 1]   1               1.0 (0.9997)
```

## Contributing
See CONTRIBUTING.md

## License
See GNU Public License v3.

## Changelog

 * 30/11/2017
    * Initial release.