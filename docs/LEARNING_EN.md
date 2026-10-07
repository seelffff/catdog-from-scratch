# Learn machine learning through this project

[Project and commands](../README.md) · [Detailed Russian textbook](LEARNING_RU.md) · [Model card](MODEL_CARD.md)

This guide explains the reasoning behind the implementation. Work through the calculations and experiments, then inspect the corresponding source. Installation commands are in the README. The recorded 97.5% test accuracy describes one documented experiment; your goal as a learner is to understand how that claim can be checked.

## 1. What the model learns

Supervised learning starts with examples `(image, label)`. Here, `0` means cat and `1` means dog. A function with adjustable parameters transforms an image into two scores. Training adjusts those parameters so that the correct score tends to be larger on unseen examples.

The dataset contains observations, not the model's parameters. The architecture determines how many parameters exist before the first image is seen. In this network there are 11,286,882 trainable numbers, initialized randomly. Gradient descent learns their values.

Imagine recognizing an animal using its ears, muzzle, texture and overall shape. We do not manually write those rules into the classifier. Layers learn useful combinations of visual patterns because those combinations reduce prediction errors on training examples.

**Exercise:** Explain the difference between one image, one label, one parameter, one batch and one epoch. An epoch is one pass through the training sampler; validation examples are not used for parameter updates.

## 2. An image is a tensor

A color image is an array of red, green and blue intensities. A batch passed to this PyTorch model has shape `[B, 3, H, W]`: batch, channels, height, width. At inference, `H = W = 256`.

Pixel values originally range from 0 to 255. The project maps them approximately into `[-1, 1]`:

```python
normalized = (pixel / 255.0 - 0.5) / 0.5
```

Zero becomes `-1`, 255 becomes `1`, and 128 is close to zero. This numerical scale helps optimization. The transformation must match during training and prediction.

The image is converted to RGB, resized while preserving aspect ratio, and placed on a square gray canvas. This is letterboxing. Stretching an animal into a square would change its proportions; center cropping could remove its head or tail. The final preprocessing is recorded in the checkpoint.

See [data.py](../src/catsdogs/data.py), [inference.py](../src/catsdogs/inference.py) and the independent serving adapter [scratch_v2.py](../backend/app/model/scratch_v2.py). An otherwise good model can fail if serving uses a different channel order, normalization or class mapping.

**Exercise:** Compute the normalized values of 0, 128 and 255. Then explain why a tensor of shape `[1, 256, 256, 3]` is not the expected input here.

## 3. Convolution and parameters

A convolution slides a small learned filter over the image. At each position it multiplies nearby values by filter weights, sums them, and produces a feature activation. Sharing the same filter across positions allows a pattern to be recognized in different parts of the image.

One output channel is produced by one filter spanning every input channel. For a convolution without bias:

```text
number of weights = output_channels × input_channels × kernel_height × kernel_width
```

The first layer uses 32 filters, RGB input and 3×3 kernels: `32 × 3 × 3 × 3 = 864` weights. Its filters are learned; they are not 32 photographs stored in the model. The later convolutions have more channels, so their parameter counts grow quickly.

Stride controls how far the filter moves. A stride of two reduces spatial resolution. ReLU keeps positive activations and replaces negative ones with zero. Without nonlinear operations, stacking linear layers would still produce a linear transformation and greatly restrict what the model can represent.

Pooling reduces spatial size. Global average pooling averages each final channel over its spatial positions, converting a feature grid into one number per channel. The final linear layer maps those 512 numbers to two class scores.

**Exercise:** A 3×3 convolution maps 64 input channels to 128 output channels without bias. It has `128 × 64 × 9 = 73,728` weights. How would adding one bias per output channel change the answer?

## 4. Residual blocks and channel attention

A residual block learns a correction `F(x)` and adds a shortcut: `y = F(x) + shortcut(x)`. When shapes agree, the shortcut can be the identity. When channels or resolution change, a projection makes the shapes compatible. These paths make it easier to optimize deeper networks.

Squeeze-and-excitation first summarizes each channel, then learns gates that scale the channels. This helps the network adjust which feature channels matter for a particular input. These gates are not a human explanation of its decision.

This project's `robust_resnet18` is a custom residual architecture. Its stem, pooling and squeeze-and-excitation blocks differ from the standard torchvision ResNet-18. Reading the class name alone is insufficient to reconstruct the network. See [model.py](../src/catsdogs/model.py) and [the architecture description](MODEL_CARD.md).

## 5. Scores, probabilities and loss

The last layer produces logits: unrestricted real-valued scores. Softmax turns two logits into positive values summing to one:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

For logits `[1, 2]`, the dog probability is approximately 0.731. A binary classifier always assigns its probability mass between the two available classes, even if you upload a car. Therefore 99% softmax output is not proof that the image contains an animal or that the answer is correct.

Cross-entropy penalizes assigning low probability to the correct class. With a hard label, the loss is `-log(p_correct)`: about 0.105 at probability 0.9 and 2.303 at probability 0.1. In code, pass raw logits to `CrossEntropyLoss`; it already performs the necessary log-softmax operation.

Label smoothing slightly softens training targets. Mixup blends two training images and their targets. Both can discourage excessive confidence, but they do not replace evaluation on real images.

**Exercise:** Why does predicting the wrong answer confidently cost more than predicting it uncertainly? Why should you not apply softmax twice?

## 6. Gradients and a training step

A gradient measures how the loss changes when a parameter changes slightly. Backpropagation applies the chain rule through the computation graph to obtain those gradients. An optimizer uses them to update the parameters.

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

PyTorch accumulates gradients unless cleared. `backward()` calculates derivatives; `step()` changes parameters. A forward pass alone does not train the network. Validation uses `model.eval()` and disabled gradient tracking, with no optimizer update.

Learning rate sets the update scale. Too large can destabilize training; too small can make progress slow. This experiment uses AdamW, weight decay, a warmup and cosine learning-rate schedule. Warmup introduces larger updates gradually, and the later schedule lowers their scale.

An exponential moving average, or EMA, maintains smoothed copies of the learned parameters. The released checkpoint uses EMA weights selected on validation at epoch 67 of a 90-epoch run. Completing more epochs does not imply that the final epoch is best.

See [train.py](../src/catsdogs/train.py) and [the recorded configuration](../reports/v2/training/robust224_w64/config.json).

## 7. Generalization and overfitting

Training loss measures fit to examples used for updates. Validation estimates behavior on held-out examples from the same collection. If training keeps improving while validation worsens, the network may be overfitting.

Augmentation generates transformed versions of training images: changes in crop, orientation, color or composition. It should preserve the intended label. Augment only the training pipeline; evaluation should use a fixed, documented transformation.

Dropout, weight decay, mixup, label smoothing and early stopping are different tools with different effects. Adding every regularizer at its maximum strength can cause underfitting. Change one meaningful factor at a time and compare it on validation.

**Exercise:** Sketch training and validation loss curves for underfitting, useful learning and overfitting. What evidence would justify increasing model capacity?

## 8. Splits, duplicates and leakage

This collection contains 9,972 valid images. The committed split has 7,975 training, 997 validation and 1,000 test examples. Classes are approximately balanced.

Training data updates parameters. Validation selects the architecture, checkpoint, inference resolution and calibration. Test estimates performance after those choices are frozen. Repeatedly using test scores to select changes turns the test into another validation set.

Near-duplicate images must stay in the same partition: almost the same photograph appearing in training and test can inflate the score. The audit records zero exact duplicates and four near-duplicate pairs; the split handles those groups together.

The split manifests provide an explicit filename-to-partition mapping. A random seed helps reproduce a procedure, while a committed manifest records its actual result. The [methodology](METHODOLOGY.md) also explains why the V2 model was randomly initialized after the test split changed.

**Exercise:** Is initializing a new model from weights trained on an earlier split safe if some earlier training images are now in the new test split? No: information about the new test may already be in those weights.

## 9. Calibration and evaluation

Accuracy is `correct / total`. The recorded final model predicts 975 of 1,000 test images correctly. The confusion matrix distinguishes types of errors:

| True / predicted | Cat | Dog |
|---|---:|---:|
| Cat | 483 | 14 |
| Dog | 11 | 492 |

Precision asks how often a prediction of a class is correct. Recall asks how many actual examples of that class are found. F1 combines precision and recall. Macro F1 averages the class-specific F1 values equally. ROC AUC evaluates ranking across decision thresholds; a high AUC does not directly establish probability calibration.

Temperature scaling divides logits by a positive value `T` before softmax. The selected value is about 0.87055. A single shared positive temperature changes probabilities without changing which logit is largest. Fit it on validation, not test. At serving time the dog threshold is 0.5 and flip test-time augmentation is disabled.

The 95% Wilson interval for the recorded accuracy is approximately 96.34%–98.30%, under binomial sampling assumptions. This describes sampling uncertainty, not protection against a new data distribution. Different breeds, artistic images or unusual lighting can produce a different error rate.

See [evaluate.py](../src/catsdogs/evaluate.py), [evaluation.json](../reports/v2/evaluation.json), and the recorded [confusion matrix](figures/confusion_matrix.png) and [reliability diagram](figures/reliability.png).

## 10. From experiment to website

Serving loads frozen weights once, applies the saved preprocessing, runs inference without gradients, and returns probabilities and real early feature maps. It must not run an optimizer. The model file includes parameters and inference metadata; a checksum identifies the exact portable artifact.

The API controls file size, decoded image dimensions and concurrent inference. The UI sends a photo and renders the response. These safeguards affect service stability; they do not increase classification accuracy.

The first convolution, activation and pooling visualizations use the actual model. Later animated illustrations explain the pipeline but are not all intermediate activations. A feature map shows where one channel activates; it does not establish a causal explanation or an anatomical detector.

## 11. A practical learning plan

1. Calculate normalization, parameter counts, softmax and cross-entropy by hand.
2. Trace one image through preprocessing and inspect tensor shapes.
3. Read one residual block and explain its shortcut.
4. Run the tests, then predict a photo with the released checkpoint.
5. On training data only, try overfitting a tiny subset as a pipeline diagnostic. Success does not measure generalization.
6. Train a small baseline and record training/validation curves.
7. Reproduce the documented recipe in a new experiment directory.
8. Change one augmentation or regularization choice; compare validation results and keep a written hypothesis.
9. Freeze the chosen pipeline before a fresh final test evaluation.
10. Explain the remaining mistakes and limitations, rather than promising universal recognition.

Before claiming that a model improved, identify the comparison dataset, selected metric, selection procedure and whether both models were evaluated on the same untouched examples. Careful experimental reasoning is as much a part of ML as implementing the network.
