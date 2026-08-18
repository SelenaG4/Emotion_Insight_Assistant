# happy
Best-performing class in the trained model: precision 0.81, recall 0.91, f1 0.85 on
the 128-image held-out test set. High recall means the model rarely misses a genuine
'happy' expression; precision of 0.81 means roughly 1 in 5 'happy' predictions is a
misclassification of another class.

# sad
Weakest class by recall: precision 0.70, recall 0.59, f1 0.64. This is the class most
likely to be under-detected -- a real 'sad' expression is more likely to be missed
(labeled as something else) than for the other three classes. Any client-facing summary
should flag 'sad' counts as a lower bound, not an exact count.

# neutral
Precision 0.64, recall 0.72, f1 0.68. Second-weakest class; precision of 0.64 means
roughly 1 in 3 'neutral' predictions is actually a misclassified other expression, most
plausibly a subtle or borderline 'sad' expression given the visual similarity between
low-intensity sad and neutral faces.

# surprise
Strongest class by precision: precision 0.97, recall 0.88, f1 0.92. When the model says
'surprise' it is almost always right; the training set was also the most imbalanced
class (3,173 images vs. ~3,980 for the other three), which is a plausible partial cause
of the (still good) recall gap relative to precision.

# model_choice
The deployed model is "Model 3" from the original capstone: a 5-convolutional-block CNN
(64-128-512-512-128 filters) with batch normalization and LeakyReLU, trained from
scratch on grayscale 48x48 faces, no ImageNet pretraining. It reached 77.3% overall test
accuracy, beating a simpler baseline CNN (66.4%), a deeper batchnorm CNN (71.1%), and
three transfer-learning heads built on VGG16 (50.0%), ResNet101 (25.0%, chance level),
and EfficientNetV2B2 (25.0%, chance level). The transfer-learning models underperformed
because they were pretrained at ~224x224 resolution; resizing them down to this
dataset's native 48x48 images destroyed most of the pretrained features they rely on.

# dataset_and_limits
Trained on 15,109 images, validated on 4,977, tested on a held-out set of only 128
images (32 per class). A 128-image test set is small enough that the reported
percentages have real sampling noise -- treat any single-decimal-point comparison
between models as indicative, not exact. The four classes are happy, sad, neutral, and
surprise; there is no 'angry', 'fear', or 'disgust' class in this dataset, so those
expressions will be forced into one of the four available labels.

# confidence_and_ethics
Emotion predictions from this model are outputs of a pattern-matching classifier on
facial geometry, not a measurement of how someone actually feels -- facial expression
and internal emotional state are not the same thing, and this gap is well documented in
the affective-computing literature. Any client-facing report should present detections
as "the model observed an expression consistent with X, confidence Y%", never as a
factual claim about a person's emotional state.
