# Comprehensive ML/DL/LLM Interview Questions 2025
## An Exhaustive Collection Covering the Full Breadth of Machine Learning

---

## Table of Contents
1. [Fundamentals & Statistics](#fundamentals--statistics)
2. [Classical Machine Learning](#classical-machine-learning)
3. [Deep Learning Basics](#deep-learning-basics)
4. [Computer Vision](#computer-vision)
5. [Natural Language Processing](#natural-language-processing)
6. [Large Language Models (LLMs)](#large-language-models-llms)
7. [Transformers & Attention](#transformers--attention)
8. [Optimization & Training](#optimization--training)
9. [Regularization & Generalization](#regularization--generalization)
10. [Model Evaluation & Metrics](#model-evaluation--metrics)
11. [Feature Engineering](#feature-engineering)
12. [Ensemble Methods](#ensemble-methods)
13. [Unsupervised Learning](#unsupervised-learning)
14. [Reinforcement Learning](#reinforcement-learning)
15. [MLOps & Production](#mlops--production)
16. [Advanced Topics](#advanced-topics)

---

## Fundamentals & Statistics

### Basic Probability & Statistics

**1. What is the difference between probability and likelihood?**
- Probability: Given a model, what's the chance of observing specific data? P(data|model)
- Likelihood: Given observed data, how probable is a particular model? L(model|data)
- Probability integrates to 1 over outcomes; likelihood doesn't necessarily integrate to 1 over parameters

**2. Explain the Central Limit Theorem and its importance in ML**
- The distribution of sample means approaches normal distribution as sample size increases, regardless of the population distribution
- Enables hypothesis testing and confidence intervals
- Justifies using normal approximations in many algorithms
- Critical for understanding gradient descent convergence

**3. What's the difference between frequentist and Bayesian approaches?**
- Frequentist: Parameters are fixed, data is random; uses point estimates and confidence intervals
- Bayesian: Parameters are random variables with distributions; incorporates prior knowledge; updates beliefs with data
- Bayesian provides full posterior distributions; Frequentist provides point estimates

**4. Explain Maximum Likelihood Estimation (MLE) vs Maximum A Posteriori (MAP)**
- MLE: Finds parameter that maximizes P(data|θ); doesn't use priors
- MAP: Finds parameter that maximizes P(θ|data) ∝ P(data|θ)P(θ); incorporates prior
- MAP reduces to MLE with uniform prior
- MAP is less prone to overfitting due to regularization from prior

**5. What is the bias-variance trade-off?**
- Bias: Error from incorrect assumptions; high bias = underfitting
- Variance: Error from sensitivity to training data fluctuations; high variance = overfitting
- Total Error = Bias² + Variance + Irreducible Error
- Complex models: low bias, high variance; Simple models: high bias, low variance

**6. Explain covariance vs correlation**
- Covariance: Measures how two variables change together; units depend on variables
- Correlation: Normalized covariance (-1 to 1); dimensionless
- Correlation = Cov(X,Y) / (σx * σy)
- Correlation captures linear relationships only

**7. What are Type I and Type II errors?**
- Type I (False Positive): Rejecting true null hypothesis (α = significance level)
- Type II (False Negative): Failing to reject false null hypothesis (β)
- Power = 1 - β
- Trade-off controlled by significance threshold

**8. Explain p-values and their limitations**
- P-value: Probability of observing data at least as extreme as observed, assuming null hypothesis is true
- Not the probability that null is true
- Sensitive to sample size (large samples → small p-values even for trivial effects)
- Subject to misinterpretation and p-hacking

**9. What is the law of large numbers?**
- Sample average converges to expected value as sample size increases
- Weak LLN: Convergence in probability
- Strong LLN: Convergence almost surely
- Foundation for Monte Carlo methods

**10. Explain conditional probability and Bayes' Theorem**
- P(A|B) = P(A∩B) / P(B)
- Bayes: P(A|B) = P(B|A)P(A) / P(B)
- Foundation of Bayesian inference
- Used in Naive Bayes, Bayesian networks

### Linear Algebra Essentials

**11. What is the difference between eigenvalues and singular values?**
- Eigenvalues: For square matrices A, Av = λv (direction preserved)
- Singular values: For any matrix A = UΣV^T, diagonal of Σ (always non-negative)
- Every matrix has singular values; only square matrices have eigenvalues
- Singular values = square root of eigenvalues of A^T A

**12. Explain matrix rank and its importance in ML**
- Rank: Maximum number of linearly independent columns/rows
- Full rank: Invertible matrix; unique solutions
- Rank deficiency: Redundant features, multicollinearity
- Affects dimensionality reduction, matrix factorization

**13. What is the Moore-Penrose pseudoinverse?**
- Generalization of inverse for non-square or singular matrices
- A^+ = (A^T A)^-1 A^T for full column rank
- Used in least squares: θ = (X^T X)^-1 X^T y
- Handles underdetermined and overdetermined systems

**14. Explain matrix norms (L1, L2, Frobenius)**
- L1 norm: Sum of absolute values of elements
- L2 norm: Largest singular value (spectral norm)
- Frobenius norm: √(sum of squared elements) = √(sum of squared singular values)
- Used in regularization and measuring matrix "size"

**15. What is Principal Component Analysis geometrically?**
- Finds orthogonal axes that maximize variance
- First PC: direction of maximum variance
- Subsequent PCs: orthogonal directions of decreasing variance
- Rotation to align data with axes of maximum spread

**16. Explain the difference between determinant and trace**
- Determinant: Product of eigenvalues; measures volume scaling
- Trace: Sum of eigenvalues = sum of diagonal elements
- det(A) = 0 → singular matrix
- Both are invariant under change of basis

---

## Classical Machine Learning

### Linear Models

**17. Derive the closed-form solution for linear regression**
- Minimize: ||y - Xθ||²
- Gradient: ∇θ = -2X^T(y - Xθ) = 0
- Solution: θ = (X^T X)^-1 X^T y (Normal Equation)
- Requires X^T X to be invertible

**18. What's the difference between Ridge and Lasso regression?**
- Ridge (L2): Minimizes ||y - Xθ||² + λ||θ||²₂; shrinks coefficients
- Lasso (L1): Minimizes ||y - Xθ||² + λ||θ||₁; enforces sparsity
- Ridge: All features included; Lasso: Feature selection (some θ = 0)
- Ridge has closed form; Lasso requires iterative solution

**19. Why does Lasso provide feature selection but Ridge doesn't?**
- L1 penalty has corners at axes (non-differentiable at 0)
- Contours of L1 ball are diamond-shaped; L2 ball is circular
- Optimization more likely to hit axes with L1
- Gradient of L1 is constant, pushing small weights to exactly zero

**20. What is Elastic Net and when would you use it?**
- Combines L1 and L2: ||y - Xθ||² + λ₁||θ||₁ + λ₂||θ||²₂
- Gets sparsity from L1 and stability from L2
- Better than Lasso when features are highly correlated
- Handles p >> n better than Ridge or Lasso alone

**21. Explain the probabilistic interpretation of linear regression**
- Assume y = Xθ + ε, where ε ~ N(0, σ²)
- Equivalent to assuming y|X ~ N(Xθ, σ²)
- MLE leads to least squares solution
- Adding Gaussian prior on θ gives Ridge regression (MAP)

**22. What are the assumptions of linear regression?**
- Linearity: Relationship between X and y is linear
- Independence: Observations are independent
- Homoscedasticity: Constant variance of errors
- Normality: Errors are normally distributed
- No multicollinearity: Features aren't highly correlated

**23. How do you interpret logistic regression coefficients?**
- Coefficient β: log-odds change per unit increase in feature
- exp(β): Odds ratio (multiplicative effect on odds)
- For probability: ∂P/∂x = βP(1-P)
- Can't directly interpret as probability change (non-linear)

**24. What's the difference between logistic regression and linear regression?**
- Output: Logistic (probabilities 0-1), Linear (continuous)
- Loss: Logistic (cross-entropy), Linear (MSE)
- Assumptions: Logistic (Bernoulli outcome), Linear (Gaussian errors)
- Link function: Logistic (logit), Linear (identity)

**25. Why use sigmoid function in logistic regression?**
- Maps real numbers to (0, 1) for probabilities
- Smooth, differentiable everywhere
- Symmetry: σ(-x) = 1 - σ(x)
- Natural from log-odds: log(p/(1-p)) = z → p = 1/(1+e^-z)

### Support Vector Machines

**26. Explain the intuition behind SVMs**
- Find hyperplane that maximizes margin between classes
- Margin: Distance from hyperplane to nearest points (support vectors)
- Only support vectors matter for decision boundary
- Robust to outliers far from boundary

**27. What is the kernel trick and why is it useful?**
- Implicitly maps data to higher-dimensional space without computing transformation
- K(x, x') = φ(x)^T φ(x') computed directly
- Avoids curse of dimensionality in feature computation
- Makes non-linear classification possible with linear methods

**28. Explain common kernels: Linear, RBF, Polynomial**
- Linear: K(x,x') = x^T x' (no transformation)
- RBF/Gaussian: K(x,x') = exp(-γ||x-x'||²) (infinite dimensions)
- Polynomial: K(x,x') = (x^T x' + c)^d (degree d polynomial features)
- RBF most common; polynomial can overfit

**29. What are support vectors?**
- Training points that lie on the margin boundaries
- Only these points determine the decision boundary
- Removing non-support vectors doesn't change model
- Typically small subset of training data

**30. What's the difference between hard-margin and soft-margin SVM?**
- Hard-margin: Requires perfect separation; no slack allowed
- Soft-margin: Allows misclassification with penalty (C parameter)
- Hard-margin fails with non-separable data
- Soft-margin more practical; C controls regularization

**31. How does the C parameter affect SVM?**
- Large C: Smaller margin, lower training error (more overfitting risk)
- Small C: Larger margin, more misclassifications (more regularization)
- C = ∞ approximates hard-margin SVM
- Trade-off between margin width and classification errors

### Decision Trees & Tree-Based Methods

**32. How does a decision tree make splits?**
- Greedy top-down approach
- Classification: Gini impurity or Entropy (Information Gain)
- Regression: MSE or MAE reduction
- Choose split that maximizes reduction in impurity/error

**33. What is Gini impurity vs Entropy?**
- Gini: 1 - Σp²ᵢ (probability of misclassification)
- Entropy: -Σpᵢlog(pᵢ) (information theory measure)
- Both measure node purity; Entropy more computationally expensive
- Gini ranges [0, 0.5]; Entropy [0, log(k)] for k classes

**34. What is Information Gain?**
- IG = Entropy(parent) - Weighted_Average(Entropy(children))
- Measures reduction in uncertainty from split
- Biased toward features with many values
- Addressed by Gain Ratio (normalizing by split information)

**35. How do you prevent decision trees from overfitting?**
- Pre-pruning: max_depth, min_samples_split, min_samples_leaf, max_features
- Post-pruning: Grow full tree, then remove branches
- Cost-complexity pruning: Balance tree size and accuracy
- Ensemble methods: Random forests, boosting

**36. What are the advantages and disadvantages of decision trees?**
- Advantages: Interpretable, non-linear, handles mixed data, no scaling needed
- Disadvantages: High variance, overfitting prone, biased to dominant classes, greedy
- Unstable: Small data changes → different tree
- Can't extrapolate beyond training range

**37. How does Random Forest work?**
- Ensemble of decision trees with bagging
- Each tree: Random subset of data (bootstrap) + random subset of features
- Predictions: Majority vote (classification) or average (regression)
- Reduces variance while maintaining low bias

**38. What is bagging and how does it reduce variance?**
- Bootstrap Aggregating: Train models on bootstrap samples
- Combine predictions by averaging/voting
- Variance reduction: σ²/n for independent models
- Effective for high-variance, low-bias models (deep trees)

**39. What's the difference between Random Forest and ExtraTrees?**
- Random Forest: Best split among random features
- ExtraTrees (Extremely Randomized Trees): Random thresholds for splits
- ExtraTrees: More randomness, faster training, often similar performance
- ExtraTrees may reduce variance further

**40. What is feature importance in Random Forests?**
- Measure 1: Average decrease in impurity across all trees
- Measure 2: Permutation importance (performance drop when feature shuffled)
- Higher importance → more relevant feature
- Can be biased toward high-cardinality features

### Gradient Boosting

**41. Explain the intuition behind gradient boosting**
- Sequentially add weak learners to correct previous errors
- Each new model fits the residuals (errors) of ensemble so far
- Additive model: F(x) = Σ αᵢhᵢ(x)
- Gradient descent in function space

**42. What's the difference between AdaBoost and Gradient Boosting?**
- AdaBoost: Reweights samples based on errors; exponential loss
- Gradient Boosting: Fits residuals; flexible loss functions
- AdaBoost: Weights focus on hard examples; GBM: general gradient framework
- GBM more general and commonly used

**43. How does XGBoost improve upon traditional gradient boosting?**
- Regularization: L1/L2 penalties on leaf weights
- Handling missing values: Learns optimal default direction
- Parallel tree building: Speed optimization
- Second-order approximation: Uses Hessian for better optimization
- Built-in cross-validation

**44. What is LightGBM and how does it differ from XGBoost?**
- Leaf-wise growth vs level-wise (XGBoost)
- Gradient-based One-Side Sampling (GOSS): Keeps large gradients
- Exclusive Feature Bundling (EFB): Bundles mutually exclusive features
- Faster training, lower memory for large datasets

**45. What is CatBoost's main innovation?**
- Ordered target encoding: Prevents target leakage
- Symmetric trees: Faster inference, less overfitting
- Native categorical feature handling
- Reduced need for extensive hyperparameter tuning

**46. Explain the learning rate in gradient boosting**
- Scales contribution of each tree: F ← F + η * h
- Small η: More trees needed, better generalization, slower training
- Large η: Fewer trees, faster training, more overfitting risk
- Typical values: 0.01-0.3
- Often combined with early stopping

**47. What is early stopping in gradient boosting?**
- Monitor validation performance during training
- Stop when performance stops improving for n rounds
- Prevents overfitting without manually tuning n_estimators
- Requires validation set

### Clustering

**48. How does K-means work? What are its limitations?**
- Algorithm: (1) Initialize k centroids (2) Assign points to nearest centroid (3) Update centroids (4) Repeat
- Limitations: Requires predefined k, assumes spherical clusters, sensitive to initialization and outliers
- Uses Euclidean distance (assumes isotropic variance)
- Converges to local optima

**49. How do you choose the number of clusters (k)?**
- Elbow method: Plot within-cluster sum of squares vs k
- Silhouette score: Measures separation quality
- Gap statistic: Compares within-cluster dispersion to null reference
- Domain knowledge often most important

**50. What is the Silhouette score?**
- s(i) = (b(i) - a(i)) / max(a(i), b(i))
- a(i): Average distance to points in same cluster
- b(i): Average distance to points in nearest other cluster
- Range [-1, 1]; higher is better; negative means likely wrong cluster

**51. Explain DBSCAN and when to use it**
- Density-based clustering: Groups closely packed points
- Parameters: ε (neighborhood radius), min_samples (minimum points)
- Can find arbitrary shapes, identifies outliers as noise
- Doesn't require k; struggles with varying densities

**52. What's the difference between K-means and hierarchical clustering?**
- K-means: Partitional, requires k, faster, flat clusters
- Hierarchical: Agglomerative/divisive, creates dendrogram, slower, nested clusters
- Hierarchical: Can visualize at multiple granularities
- K-means: Better for large datasets

**53. Explain linkage methods in hierarchical clustering**
- Single linkage: Minimum distance between clusters (chaining problem)
- Complete linkage: Maximum distance (breaks large clusters)
- Average linkage: Average of all distances (balanced)
- Ward's method: Minimizes within-cluster variance (common choice)

**54. What is Gaussian Mixture Model (GMM)?**
- Probabilistic model: Data generated from mixture of Gaussians
- Each point has probability of belonging to each cluster
- Soft clustering (vs K-means hard assignment)
- Fitted using Expectation-Maximization (EM) algorithm

**55. Explain the EM algorithm for GMMs**
- E-step: Compute probability of each point belonging to each Gaussian
- M-step: Update Gaussian parameters (mean, covariance, mixing coefficients)
- Iterates until convergence
- Generalizes K-means (hard EM = K-means)

---

## Deep Learning Basics

### Neural Network Fundamentals

**56. What is a neural network? Explain the universal approximation theorem**
- Network of parameterized functions (layers) composed together
- UAT: NN with single hidden layer can approximate any continuous function (given enough units)
- Doesn't specify required width or how to find weights
- Deeper networks more efficient for complex functions

**57. Explain forward propagation vs backpropagation**
- Forward: Compute outputs layer by layer from input to output
- Backward: Compute gradients layer by layer from output to input using chain rule
- Backprop enables efficient gradient computation in O(n) time
- Automatic differentiation generalizes this

**58. Derive the backpropagation equations for a simple network**
- For layer l: a^l = σ(W^l a^(l-1) + b^l)
- δ^L = (a^L - y) ⊙ σ'(z^L) (output layer error)
- δ^l = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^l) (backprop error)
- ∂C/∂W^l = δ^l (a^(l-1))^T, ∂C/∂b^l = δ^l

**59. What is the vanishing gradient problem?**
- Gradients become exponentially small in deep networks
- Caused by repeated multiplication of small derivatives (σ' ∈ [0, 0.25] for sigmoid)
- Earlier layers learn very slowly
- Solutions: ReLU, skip connections, batch normalization, careful initialization

**60. What is the exploding gradient problem?**
- Gradients become exponentially large in deep networks
- Causes numerical instability and divergence
- Common in RNNs with long sequences
- Solutions: Gradient clipping, careful initialization, LSTM/GRU

**61. Why is ReLU better than sigmoid/tanh?**
- No vanishing gradient for positive inputs (gradient = 1)
- Computationally efficient (simple threshold)
- Sparse activation (negative inputs = 0)
- Empirically faster convergence
- Issues: Dead ReLU problem (neurons stuck at 0)

**62. What are variants of ReLU?**
- Leaky ReLU: f(x) = max(αx, x) where α ~ 0.01 (addresses dead ReLU)
- Parametric ReLU (PReLU): α is learned
- ELU: Smooth for negative values; f(x) = x if x>0, else α(e^x - 1)
- GELU: Gaussian Error Linear Unit; used in transformers
- Swish/SiLU: f(x) = x * sigmoid(x); smooth, non-monotonic

**63. What is the dead ReLU problem?**
- Neurons output 0 for all inputs (never activate)
- Caused by large negative bias or learning rate
- Gradients always 0 → no learning
- Solutions: Leaky ReLU, careful initialization, lower learning rate

**64. Explain weight initialization strategies**
- Zero initialization: Symmetry problem (all neurons learn same thing)
- Random small values: Risk of vanishing/exploding gradients
- Xavier/Glorot: Var(W) = 1/n_in (good for sigmoid/tanh)
- He initialization: Var(W) = 2/n_in (good for ReLU)
- Goal: Maintain activation and gradient variance across layers

**65. What is batch normalization and why does it work?**
- Normalizes layer inputs to zero mean, unit variance per mini-batch
- BN(x) = γ((x - μ) / σ) + β where γ, β are learned
- Benefits: Faster training, higher learning rates, regularization, reduces sensitivity to initialization
- Why: Reduces internal covariate shift (though debate on mechanism)

**66. What are alternatives to batch normalization?**
- Layer Normalization: Normalizes across features (better for RNNs, transformers)
- Instance Normalization: Normalizes per sample per channel (style transfer)
- Group Normalization: Normalizes groups of channels (small batch sizes)
- Weight Normalization: Reparameterizes weights (v/||v||)

**67. What is dropout and how does it work?**
- Randomly sets fraction p of activations to zero during training
- Each forward pass uses different sub-network
- At inference: Scale by (1-p) or use all weights * (1-p)
- Acts as ensemble of networks; prevents co-adaptation of features
- Bayesian interpretation: Approximate variational inference

**68. Why doesn't dropout work well with batch normalization?**
- BN and dropout have conflicting interactions
- Dropout's randomness affects batch statistics
- Training-test mismatch in statistics
- Often one or the other is used, not both
- When combined, dropout typically placed after BN

**69. What is the difference between parameters and hyperparameters?**
- Parameters: Learned from data (weights, biases)
- Hyperparameters: Set before training (learning rate, architecture, epochs)
- Parameters updated by optimization; hyperparameters tuned by validation
- # parameters affects model capacity; hyperparameters affect learning

**70. How do you count parameters in a neural network?**
- Fully connected: (n_in × n_out) + n_out (weights + biases)
- Convolutional: (k_h × k_w × c_in × c_out) + c_out
- Remember to account for each layer
- Total = sum across all layers

---

## Computer Vision

### Convolutional Neural Networks

**71. What is a convolution operation?**
- Slides filter/kernel over input, computing element-wise products and summing
- Output(i,j) = Σ Σ Input(i+m, j+n) × Kernel(m,n)
- Preserves spatial structure
- Parameter sharing and local connectivity

**72. What determines the output size of a conv layer?**
- Formula: O = (I - K + 2P)/S + 1
- I: Input size, K: Kernel size, P: Padding, S: Stride
- Example: 32×32 input, 5×5 kernel, stride=1, pad=0 → 28×28 output

**73. What is padding and why use it?**
- Adds zeros (or other values) around input borders
- Valid: No padding (output shrinks)
- Same: Padding so output size = input size (P = (K-1)/2 for S=1)
- Preserves spatial dimensions
- Prevents information loss at borders

**74. What is stride in convolution?**
- Step size when sliding filter
- Stride = 1: Filter moves one pixel at a time
- Stride > 1: Downsamples spatially (reduces computation)
- Larger stride = smaller output

**75. What is dilated/atrous convolution?**
- Inserts spaces between kernel elements
- Receptive field grows without adding parameters
- Rate r: kernel element spacing
- Effective kernel size: K + (K-1)(r-1)
- Useful for dense prediction (segmentation)

**76. What is the receptive field?**
- Region of input that affects a particular output neuron
- Grows with depth (layers) and kernel size
- Formula: RF(l) = RF(l-1) + (K-1) × Πᵢ Sᵢ (product of strides)
- Larger RF allows seeing more context

**77. What is 1×1 convolution and why use it?**
- Pointwise convolution across channels
- Reduces/increases channel dimensions
- Adds non-linearity without changing spatial size
- Computationally cheap (vs larger kernels)
- Used in Inception, MobileNet

**78. Explain pooling layers (max, average)**
- Downsamples spatial dimensions (reduces parameters, computation)
- Max pooling: Takes maximum in window (preserves strongest features)
- Average pooling: Takes average (smoother, less aggressive)
- Common: 2×2 window, stride 2 (halves dimensions)
- Non-learnable (no parameters)

**79. What is global average pooling?**
- Averages each feature map to single value
- Replaces fully connected layers at end
- Reduces parameters dramatically
- More robust to spatial translations
- Used in many modern architectures

**80. What is transposed convolution (deconvolution)?**
- Upsamples spatial dimensions
- Learns upsampling (vs bilinear interpolation)
- Not true inverse of convolution
- Used in segmentation, GANs, autoencoders
- Can cause checkerboard artifacts

**81. Explain separable convolutions**
- Depthwise separable: Depthwise (each channel separately) + pointwise (1×1)
- Spatial separable: Separate horizontal and vertical (K×1 then 1×K)
- Reduces parameters and computation
- Used in MobileNet, Xception

**82. What is batch size's effect on CNN training?**
- Larger batch: More stable gradients, better hardware utilization, less regularization
- Smaller batch: More noise in gradients (implicit regularization), better generalization
- Memory constraint: Limits maximum batch size
- Learning rate often scaled with batch size

### CNN Architectures

**83. Explain the key innovation in AlexNet**
- First successful deep CNN on ImageNet (2012)
- ReLU activation (vs sigmoid/tanh)
- Dropout for regularization
- Data augmentation
- GPU training with parallelization
- 8 layers (5 conv, 3 fc)

**84. What is VGGNet's design principle?**
- Depth matters: 16-19 layers
- Small 3×3 filters throughout
- Stacks of conv layers before pooling
- Simple, homogeneous architecture
- Very large # parameters (138M in VGG-16)

**85. Why does VGG use 3×3 filters?**
- Two 3×3 convs have same receptive field as one 5×5
- Three 3×3 convs = one 7×7
- Fewer parameters: 3(3²C²) vs 5²C² or 7²C²
- More non-linearities (more layers)
- Better feature learning

**86. Explain ResNet and skip connections**
- Residual connections: y = F(x) + x (shortcut identity mapping)
- Solves vanishing gradient problem
- Enables training very deep networks (100+ layers)
- Easier to optimize: Learn residual instead of unreferenced function
- Identity shortcuts cost-free

**87. What is the degradation problem ResNet solves?**
- Very deep plain networks perform worse than shallower ones (not just overfitting)
- Optimization difficulty, not representation capacity
- Residual learning makes it easier to learn identity mapping
- Skip connections create "gradient highways"

**88. Explain the Inception module**
- Multiple parallel conv paths (1×1, 3×3, 5×5, pooling)
- Concatenates outputs (multi-scale features)
- 1×1 convs before larger filters (dimension reduction)
- "Network in network" design
- Efficient multi-scale feature extraction

**89. What is the bottleneck architecture in ResNet?**
- 1×1 (reduce) → 3×3 → 1×1 (expand)
- Reduces parameters and computation
- First 1×1 reduces channels (e.g., 256→64)
- 3×3 operates on reduced dimensions
- Final 1×1 expands back to match input

**90. Explain MobileNet's core idea**
- Depthwise separable convolutions
- Trades accuracy for efficiency (mobile/embedded)
- Width multiplier α and resolution multiplier ρ
- Dramatically fewer parameters and FLOPs
- MobileNetV2: Inverted residuals, linear bottlenecks

**91. What is EfficientNet's scaling strategy?**
- Compound scaling: Simultaneously scales depth, width, resolution
- Grid search to find optimal scaling coefficients
- Depth: Network depth (layers)
- Width: Channel width
- Resolution: Input image size
- More balanced scaling than increasing one dimension

**92. Explain the architecture of DenseNet**
- Each layer connects to all previous layers
- Feature reuse and gradient flow
- Concatenation (not addition like ResNet)
- Reduces parameters (no need to relearn features)
- Can cause memory issues

**93. What is squeeze-and-excitation (SE) block?**
- Channel attention mechanism
- Global pool → fc (reduce) → fc (expand) → sigmoid → scale
- Learns channel-wise importance
- Small parameter overhead
- Consistent improvements across architectures

### Object Detection

**94. What is object detection vs classification?**
- Classification: Assign label to entire image
- Detection: Find and classify multiple objects (bounding boxes + labels)
- More challenging: Localization + recognition
- Multiple objects, various scales

**95. Explain R-CNN and its limitations**
- Region proposals (Selective Search) → CNN features → SVM classifier
- Limitations: Slow (CNN per region ~2000), multi-stage, large disk space
- Training separate components (doesn't end-to-end)

**96. How does Fast R-CNN improve on R-CNN?**
- Single CNN for entire image (not per region)
- RoI pooling: Extract features for proposals from feature map
- Multi-task loss: Classification + bounding box regression
- Faster training and inference (~10x speedup)
- Still uses external region proposals

**97. Explain Faster R-CNN's key innovation**
- Region Proposal Network (RPN): Learns proposals (end-to-end)
- RPN shares features with detection network
- Anchor boxes: Multiple scales and ratios
- Significantly faster than Fast R-CNN
- State-of-art accuracy for two-stage detectors

**98. What are anchor boxes?**
- Pre-defined boxes at various scales and aspect ratios
- Placed at each location on feature map
- RPN predicts objectness and refinements for each anchor
- Handles multiple scales without image pyramids
- Typical: 3 scales × 3 ratios = 9 anchors per location

**99. How does YOLO (You Only Look Once) work?**
- Single-stage detector: One pass through network
- Divides image into grid (e.g., 7×7)
- Each cell predicts B bounding boxes + class probabilities
- Extremely fast (real-time detection)
- Trade-off: Lower accuracy on small objects

**100. What's the difference between one-stage and two-stage detectors?**
- Two-stage (Faster R-CNN): Proposals → Classification (more accurate, slower)
- One-stage (YOLO, SSD): Direct prediction (faster, less accurate)
- One-stage: Struggles with class imbalance
- Two-stage: RoI pooling provides better features

**101. Explain SSD (Single Shot Detector)**
- Multi-scale feature maps for detection
- Default boxes (like anchors) at each location, multiple scales
- Predictions from multiple layers (different resolutions)
- Handles various object sizes better than original YOLO
- Good speed-accuracy trade-off

**102. What is Focal Loss and why is it important?**
- Addresses class imbalance in one-stage detectors
- FL(pₜ) = -α(1-pₜ)^γ log(pₜ)
- Down-weights easy examples (high confidence)
- Focuses training on hard examples
- Enables RetinaNet to match two-stage accuracy

**103. What is Non-Maximum Suppression (NMS)?**
- Post-processing to remove duplicate detections
- Algorithm: (1) Sort by confidence (2) Keep highest, remove overlapping boxes (IoU > threshold) (3) Repeat
- Greedy algorithm, not optimal
- Variants: Soft-NMS, NMS by class

**104. What is IoU (Intersection over Union)?**
- Overlap metric for bounding boxes
- IoU = Area of Overlap / Area of Union
- Range [0, 1]; higher = better overlap
- Used in NMS, matching predictions to ground truth
- Threshold (e.g., 0.5) defines positive match

**105. Explain the concept of Feature Pyramid Networks (FPN)**
- Multi-scale feature extraction with top-down pathway
- High-level features: Semantic, low resolution
- Low-level features: Spatial detail, high resolution
- Lateral connections combine both
- Improves detection at all scales

### Image Segmentation

**106. What's the difference between semantic and instance segmentation?**
- Semantic: Label every pixel with class (same class = same label)
- Instance: Distinguish individual objects (person1 ≠ person2)
- Panoptic: Combines both (semantic for stuff, instance for things)

**107. How does FCN (Fully Convolutional Network) work?**
- Replaces FC layers with 1×1 convs
- Preserves spatial information
- Upsampling to original resolution (transposed conv)
- Skip connections from encoder to decoder
- End-to-end training for segmentation

**108. Explain U-Net architecture**
- Encoder-decoder with symmetric skip connections
- Encoder: Downsampling (max pooling)
- Decoder: Upsampling (transposed conv)
- Skip connections: Concatenate encoder and decoder features (same level)
- Excellent for medical imaging and small datasets

**109. What is the purpose of skip connections in U-Net?**
- Recover spatial information lost in downsampling
- Combine high-resolution, low-level features with low-resolution, high-level features
- Better localization
- Gradient flow

**110. Explain Mask R-CNN**
- Extends Faster R-CNN for instance segmentation
- Adds mask branch (FCN) parallel to classification and bbox
- RoIAlign (vs RoI pooling) for pixel-accurate masks
- Predicts binary mask for each detected object
- State-of-art instance segmentation

**111. What is RoIAlign and why is it better than RoI pooling?**
- RoI pooling: Quantization (rounding) causes misalignment
- RoIAlign: Bilinear interpolation, no quantization
- Improves mask prediction accuracy
- Properly aligns features with input pixels

**112. What is DeepLab and atrous spatial pyramid pooling (ASPP)?**
- Atrous convolution for larger receptive fields
- ASPP: Parallel atrous convs at multiple rates + pooling
- Captures multi-scale context
- Dense prediction without resolution loss
- Used for semantic segmentation

### Advanced Vision Topics

**113. What are Vision Transformers (ViT)?**
- Applies transformer architecture to images
- Split image into patches (16×16), linear projection
- Patch embeddings + positional encoding
- Standard transformer encoder
- Matches/exceeds CNN performance with large data

**114. How does ViT compare to CNNs?**
- ViT: Global attention (vs local receptive fields)
- ViT: Less inductive bias (needs more data)
- ViT: More interpretable attention patterns
- ViT: Computationally expensive (quadratic in sequence length)
- CNNs: Better sample efficiency on small datasets

**115. What is the CLIP model?**
- Contrastive Language-Image Pre-training
- Learns joint embedding space for images and text
- Trained on image-text pairs from internet
- Zero-shot classification (text prompts as classifiers)
- Enables multi-modal understanding

**116. Explain data augmentation in computer vision**
- Increases effective dataset size without new data
- Common: Flip, rotation, crop, color jitter, cutout, mixup
- Reduces overfitting, improves generalization
- Task-specific: Augmentations should preserve semantics
- AutoAugment: Learns augmentation policy

**117. What is transfer learning in computer vision?**
- Use pre-trained model (e.g., ImageNet) for new task
- Fine-tuning: Continue training on new data
- Feature extraction: Freeze early layers, train only top
- Works because low-level features are general
- Crucial for small datasets

**118. When should you fine-tune all layers vs just the top?**
- Small dataset + similar task → Freeze most, train top
- Large dataset + similar task → Fine-tune all with small LR
- Small dataset + different task → Freeze more, train more layers
- Large dataset + different task → Fine-tune all or train from scratch

---

## Natural Language Processing

### Text Preprocessing & Representation

**119. What is tokenization and why is it important?**
- Splitting text into units (tokens): words, subwords, characters
- Foundation for all NLP processing
- Choices affect vocabulary size, OOV handling, model performance
- Language-dependent: Spaces, punctuation, morphology

**120. Explain different tokenization strategies**
- Word-level: Split on whitespace/punctuation (large vocab, OOV issues)
- Character-level: Individual characters (no OOV, very long sequences)
- Subword: BPE, WordPiece, Unigram (balance vocab size and OOV)
- Sentence-level: Sentences as units

**121. What is Byte-Pair Encoding (BPE)?**
- Iteratively merges most frequent character/subword pairs
- Starts with characters, builds vocabulary bottom-up
- Handles OOV by breaking into known subwords
- Used in GPT, RoBERTa
- Balances vocabulary size and coverage

**122. What is WordPiece tokenization?**
- Similar to BPE but maximizes likelihood on training corpus
- Used in BERT
- Prefixes like ## indicate subword continuation
- Slightly different merge criterion than BPE

**123. What is SentencePiece?**
- Language-agnostic tokenization (treats input as raw UTF-8)
- No pre-tokenization required (handles spaces)
- Unigram language model or BPE
- Used in T5, XLNet
- Better for languages without clear word boundaries

**124. What is stemming vs lemmatization?**
- Stemming: Crude heuristic chopping (running → run; studies → studi)
- Lemmatization: Dictionary-based, returns valid word (studies → study)
- Stemming faster but less accurate
- Lemmatization needs POS tagging for best results

**125. What are stop words and should you remove them?**
- Common words with little semantic meaning (the, is, at)
- Traditional NLP: Often removed (reduce noise, improve efficiency)
- Deep learning: Usually kept (models learn relevance)
- Context-dependent: "not" is stop word but critical for sentiment

**126. Explain TF-IDF**
- Term Frequency - Inverse Document Frequency
- TF(t,d) = count(t in d) / total words in d
- IDF(t) = log(N / df(t)) where N = total docs, df = docs containing t
- TF-IDF = TF × IDF
- Balances term importance in document and corpus

**127. What is the bag-of-words model?**
- Represents text as unordered set of words (ignores grammar, order)
- Document vector: Word counts or binary indicators
- Simple, interpretable
- Loses word order, context
- Foundation for classical NLP

**128. What is one-hot encoding for words?**
- Binary vector, size = vocabulary
- Single 1 at word's index, rest 0s
- No semantic similarity (all pairs orthogonal)
- Extremely high-dimensional
- Not used in modern NLP (word embeddings instead)

**129. What are word embeddings?**
- Dense, low-dimensional vectors representing words
- Learned from data, capture semantic relationships
- Similar words have similar vectors (cosine similarity)
- Enable arithmetic: king - man + woman ≈ queen
- Foundation of modern NLP

**130. Explain Word2Vec (Skip-gram and CBOW)**
- Skip-gram: Predict context words from target word
- CBOW: Predict target word from context words
- Shallow neural network, trained on large corpus
- Learns embeddings as side effect of prediction task
- Negative sampling for efficiency

**131. How does negative sampling work in Word2Vec?**
- Training on all words infeasible (softmax over large vocabulary)
- Instead: Discriminate true context words from random words
- K negative samples per positive (typically k=5-20)
- Turns into binary classification (sigmoid)
- Computationally efficient

**132. What is GloVe?**
- Global Vectors: Matrix factorization on word co-occurrence
- Objective: Word vectors whose dot product = log co-occurrence probability
- Combines benefits of count-based (LSA) and prediction-based (Word2Vec)
- Pre-computes co-occurrence matrix
- Often similar performance to Word2Vec

**133. What is FastText?**
- Extension of Word2Vec to subword level
- Represents word as bag of character n-grams
- Example: "where" → "<wh", "whe", "her", "ere", "re>"
- Handles OOV words (compose from subwords)
- Better for morphologically rich languages

**134. What are contextualized embeddings?**
- Word representations depend on context (vs static like Word2Vec)
- Same word, different contexts → different embeddings
- Examples: ELMo, BERT, GPT embeddings
- Captures polysemy (bank = financial vs river)
- State-of-art for most NLP tasks

### Sequence Models

**135. Why are RNNs suitable for sequential data?**
- Process sequences of variable length
- Share parameters across time steps
- Maintain hidden state (memory)
- Can model temporal dependencies
- Natural for text, speech, time series

**136. Explain the vanishing/exploding gradient problem in RNNs**
- Backpropagation through time multiplies gradients
- Vanishing: Gradients → 0 (can't learn long dependencies)
- Exploding: Gradients → ∞ (numerical instability)
- Worse with long sequences
- Due to repeated matrix multiplication and activation derivatives

**137. What is LSTM and how does it solve vanishing gradients?**
- Long Short-Term Memory: Gated RNN cell
- Gates: Input, forget, output (control information flow)
- Cell state: Separate path for gradient flow
- Additive updates (vs multiplicative in vanilla RNN)
- Can learn dependencies over 100+ timesteps

**138. Explain the gates in LSTM**
- Forget gate: fₜ = σ(Wf[hₜ₋₁, xₜ] + bf) - What to remove from cell state
- Input gate: iₜ = σ(Wi[hₜ₋₁, xₜ] + bi), C̃ₜ = tanh(Wc[hₜ₋₁, xₜ] + bc) - What to add
- Output gate: oₜ = σ(Wo[hₜ₋₁, xₜ] + bo) - What to output
- Cell state: Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ

**139. What is GRU and how does it differ from LSTM?**
- Gated Recurrent Unit: Simpler than LSTM
- Two gates: Reset, update (vs three in LSTM)
- No separate cell state
- Fewer parameters, faster training
- Often similar performance to LSTM

**140. What is bidirectional RNN?**
- Two RNNs: Forward (left-to-right) and backward (right-to-left)
- Concatenate hidden states at each timestep
- Captures context from both directions
- Cannot be used for real-time prediction (needs full sequence)
- Used in BERT, ELMo

**141. Explain seq2seq models**
- Encoder-decoder architecture
- Encoder: Encodes input sequence into fixed context vector
- Decoder: Generates output sequence from context
- Used for machine translation, summarization, dialogue
- Bottleneck: Single context vector

**142. What is the attention mechanism?**
- Allows decoder to "attend" to different encoder positions
- Computes weighted average of encoder states
- Attention weights: Learned importance of each encoder position
- Solves information bottleneck of vanilla seq2seq
- Enables modeling long-range dependencies

**143. How is attention computed?**
- Score: e_ij = score(decoder_state_i, encoder_state_j)
- Common: dot product, scaled dot product, additive
- Weights: α_ij = softmax(e_ij) (attention distribution)
- Context: c_i = Σ α_ij * encoder_state_j
- Decoder uses context vector for prediction

**144. What is self-attention?**
- Attention within single sequence (vs encoder-decoder attention)
- Each position attends to all positions in sequence
- Captures dependencies regardless of distance
- Foundation of transformers
- Parallelizable (vs sequential RNN)

**145. Explain teacher forcing in seq2seq training**
- Use ground truth as decoder input (vs model's prediction)
- Faster convergence, more stable training
- Problem: Exposure bias (model never sees own errors during training)
- Inference: Model must use own predictions
- Scheduled sampling: Gradually use model predictions during training

---

## Large Language Models (LLMs)

### Transformer Architecture

**146. What is the Transformer architecture?**
- Attention-based model without recurrence or convolution
- Encoder-decoder structure (original)
- Multi-head self-attention + position-wise feed-forward
- Positional encodings for sequence order
- Highly parallelizable, captures long-range dependencies

**147. Explain multi-head attention**
- Multiple attention mechanisms in parallel
- Each head: Different learned projection of Q, K, V
- Concat outputs and project: MultiHead(Q,K,V) = Concat(head₁,...,headₕ)Wᴼ
- Allows attending to different representation subspaces
- Typical: 8-16 heads

**148. What is scaled dot-product attention?**
- Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
- Q: Queries, K: Keys, V: Values
- Scaling by √d_k prevents softmax saturation for large d_k
- O(n²d) complexity in sequence length n
- Dot product measures similarity

**149. Why do we scale by √d_k in attention?**
- Dot product magnitude grows with dimension
- Large values → softmax saturation (small gradients)
- Scaling normalizes dot product variance
- Maintains stable gradients
- Empirically improves training

**150. What are positional encodings and why are they needed?**
- Transformers have no inherent notion of position (permutation invariant)
- Add position information to input embeddings
- Sinusoidal: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d))
- Learned: Treated as parameters
- Enable model to use sequence order

**151. What is the feed-forward network in transformers?**
- Position-wise: Applied independently to each position
- Two linear transformations with activation: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
- Typically: 4× dimension expansion then back
- Adds capacity after attention
- Same parameters at all positions

**152. Explain layer normalization in transformers**
- Normalizes across features (vs batch norm across batch)
- LN(x) = γ((x - μ)/σ) + β
- Applied before/after attention and FFN (depending on variant)
- Stabilizes training
- Better for variable sequence lengths than batch norm

**153. What is the difference between encoder-only, decoder-only, and encoder-decoder transformers?**
- Encoder-only (BERT): Bidirectional context, masked language modeling, good for understanding
- Decoder-only (GPT): Unidirectional (causal) attention, next token prediction, good for generation
- Encoder-decoder (T5): Full transformer, seq2seq tasks
- Choice depends on task requirements

**154. What is causal (masked) self-attention?**
- Each position only attends to earlier positions (and itself)
- Implemented via attention mask (set future positions to -∞)
- Preserves autoregressive property
- Necessary for language generation
- Used in GPT models

**155. What is cross-attention?**
- Attention between two different sequences
- Q from one sequence (e.g., decoder), K,V from another (e.g., encoder)
- Used in encoder-decoder transformers
- Allows decoder to attend to encoder outputs
- Multiple cross-attention layers possible

### Pre-training & Fine-tuning

**156. What is pre-training in NLP?**
- Training on large unlabeled corpus with self-supervised task
- Learns general language representations
- Captures syntax, semantics, world knowledge
- Followed by fine-tuning on downstream task
- Enables transfer learning at scale

**157. Explain masked language modeling (MLM)**
- BERT's pre-training objective
- Randomly mask 15% of tokens
- Predict masked tokens from context (bidirectional)
- Forces model to understand context
- 80% [MASK], 10% random word, 10% unchanged (prevents overfitting to [MASK])

**158. What is causal language modeling?**
- GPT's pre-training objective
- Predict next token given previous tokens
- Unidirectional (left-to-right)
- Natural for text generation
- No special tokens needed

**159. What is next sentence prediction (NSP)?**
- BERT's second pre-training objective (now often omitted)
- Binary classification: Are two sentences consecutive?
- Helps with sentence-pair tasks
- Later research suggests minimal benefit
- RoBERTa removes this objective

**160. Explain sentence order prediction (SOP)**
- Alternative to NSP (used in ALBERT)
- Predict if sentences are in correct order
- More challenging than NSP
- Better for learning inter-sentence coherence
- Positive: Consecutive sentences; Negative: Swapped sentences

**161. What is permutation language modeling?**
- XLNet's pre-training objective
- Factorizes likelihood over all permutations
- Avoids [MASK] token (no pre-train/fine-tune mismatch)
- Captures bidirectional context while maintaining autoregressive property
- More complex than MLM

**162. What is replaced token detection?**
- ELECTRA's pre-training objective
- Generator produces fake tokens (small MLM model)
- Discriminator classifies tokens as original or replaced
- More sample-efficient than MLM (learns from all tokens, not just masked)
- Requires two models (generator discarded after pre-training)

**163. What is continual pre-training?**
- Further pre-training on domain-specific or newer data
- Adapts general model to specialized domain
- Cheaper than training from scratch
- Risk of catastrophic forgetting
- Used for domain adaptation (e.g., BioBERT, SciBERT)

**164. Explain different fine-tuning strategies**
- Full fine-tuning: Update all parameters
- Feature extraction: Freeze base, train task head only
- Gradual unfreezing: Unfreeze layers progressively
- Discriminative learning rates: Different LRs for different layers
- Adapter layers: Insert small trainable modules, freeze rest

**165. What is catastrophic forgetting?**
- Fine-tuning on new task degrades performance on original tasks
- Neural networks overwrite previous knowledge
- Solutions: Multi-task learning, elastic weight consolidation, rehearsal
- Worse with more dissimilar tasks
- Key challenge in continual learning

**166. What are adapter modules?**
- Small trainable layers inserted into frozen pre-trained model
- Typically: Down-project, non-linearity, up-project (bottleneck)
- Only adapters are updated during fine-tuning
- Maintains pre-trained knowledge
- Efficient: Few parameters per task

### Parameter-Efficient Fine-Tuning

**167. What is parameter-efficient fine-tuning (PEFT)?**
- Updating small subset of parameters during fine-tuning
- Reduces memory and computation
- Easier to manage multiple tasks (small task-specific modules)
- Often matches full fine-tuning performance
- Methods: LoRA, adapters, prefix tuning, prompt tuning

**168. Explain LoRA (Low-Rank Adaptation)**
- Freezes pre-trained weights
- Adds trainable low-rank decomposition: ΔW = BA (B: d×r, A: r×k where r << d,k)
- W' = W + ΔW applied during forward pass
- Drastically reduces trainable parameters (r is small, e.g., 8)
- No inference latency (can merge ΔW into W)

**169. What is prefix tuning?**
- Prepends trainable "prefix" vectors to key and value in each layer
- Prefix length typically 10-200 tokens
- Only prefix parameters are trained
- Model sees prefix as virtual tokens
- Used for conditional generation

**170. What is prompt tuning?**
- Learns soft prompts (continuous vectors) prepended to input
- Only prompt embeddings are trained
- Simpler than prefix tuning (only input layer)
- Scales better: Performance approaches fine-tuning with size
- Separate prompts per task

**171. How does prompt tuning differ from prefix tuning?**
- Prompt tuning: Only input embeddings
- Prefix tuning: Keys and values at every layer
- Prompt tuning: Simpler, fewer parameters
- Prefix tuning: Often better performance
- Both avoid modifying model parameters

**172. What is BitFit?**
- Only fine-tunes bias terms
- All other parameters frozen
- Surprisingly effective despite minimal parameters
- Very efficient
- Sometimes matches full fine-tuning

### Prompting & In-Context Learning

**173. What is few-shot prompting?**
- Provide examples in the prompt, no weight updates
- Model infers pattern from examples
- Emerged as capability in large models (GPT-3)
- No training required
- Performance improves with model size

**174. What is zero-shot prompting?**
- Task description only, no examples
- Relies on model's pre-trained knowledge
- Example: "Translate English to French: Hello → "
- More challenging than few-shot
- Requires clear instruction

**175. What is chain-of-thought (CoT) prompting?**
- Provide reasoning steps in examples
- Encourages model to "think step by step"
- Dramatically improves reasoning tasks
- Can be zero-shot: "Let's think step by step"
- Especially effective for math, logic

**176. What is self-consistency in prompting?**
- Generate multiple reasoning paths
- Take majority vote on final answers
- Reduces effect of single reasoning errors
- Improves robustness
- Requires multiple samples (higher cost)

**177. Explain instruction tuning**
- Fine-tuning on diverse instruction-following tasks
- Format: (instruction, input, output) triples
- Improves zero-shot performance on new tasks
- Examples: InstructGPT, FLAN, T0
- Critical for helpful assistants

**178. What is RLHF (Reinforcement Learning from Human Feedback)?**
- Aligns model with human preferences
- Steps: (1) Supervised fine-tuning (2) Reward model training (3) RL optimization (PPO)
- Reward model: Trained on human preference comparisons
- Policy: Optimized to maximize reward while staying close to SFT model
- Used in ChatGPT, Claude

**179. What is Constitutional AI?**
- Uses AI feedback instead of human feedback (scalable oversight)
- Model critiques own outputs against principles ("constitution")
- Revises responses to be more helpful and harmless
- Reduces need for human labeling
- Developed by Anthropic

**180. What is the difference between supervised fine-tuning and RLHF?**
- SFT: Maximizes likelihood of demonstration data
- RLHF: Optimizes reward from preference model
- SFT: Can learn undesirable patterns from data
- RLHF: Aligns with preferences even when demonstrations imperfect
- RLHF typically follows SFT

### Model Architectures & Variants

**181. What are key differences between GPT and BERT?**
- GPT: Decoder-only, causal LM, unidirectional, generation-focused
- BERT: Encoder-only, MLM, bidirectional, understanding-focused
- GPT: Autoregressive; BERT: Denoising autoencoder
- GPT: Larger (175B vs 340M), more compute
- Different strengths: GPT for generation, BERT for classification/extraction

**182. Explain GPT model evolution (GPT → GPT-2 → GPT-3 → GPT-4)**
- GPT (117M): Demonstrated pre-training + fine-tuning
- GPT-2 (1.5B): Zero-shot transfer, larger scale
- GPT-3 (175B): Few-shot learning, emergent abilities, scale is key
- GPT-4: Multimodal, better reasoning, safer, mixture of experts (rumored)
- Trend: Scaling, better training data, alignment

**183. What is T5 and how does it unify NLP tasks?**
- Text-to-Text Transfer Transformer
- All tasks as text-to-text: Input text → Output text
- Examples: "translate English to German: ..." → translation
- Enables unified training across tasks
- Pretrained on C4 corpus with various objectives

**184. What is BART?**
- Bidirectional and Auto-Regressive Transformer
- Encoder-decoder (like T5)
- Pre-training: Denoising autoencoder with various corruptions
- Good for generation and understanding
- Similar to T5 but different pre-training

**185. What is ELECTRA?**
- Efficiently Learning an Encoder that Classifies Token Replacements Accurately
- Generator-discriminator setup (not GAN)
- Discriminator trained on replaced token detection
- More sample-efficient than MLM
- Smaller models match larger BERT

**186. Explain RoBERTa improvements over BERT**
- Removes NSP objective
- Larger batches, more data, longer training
- Dynamic masking (vs static in BERT)
- No segment embeddings
- Better hyperparameters
- Significantly better performance

**187. What is ALBERT?**
- A Lite BERT: Parameter reduction techniques
- Factorized embedding: E → hidden smaller than H
- Cross-layer parameter sharing
- SOP instead of NSP
- Fewer parameters but similar performance

**188. What is DeBERTa?**
- Decoupling Enhanced BERT
- Disentangled attention: Content and position separately
- Enhanced mask decoder: Position information in decoding
- Improved performance over RoBERTa
- State-of-art on SuperGLUE (briefly)

**189. What is XLNet?**
- Combines benefits of autoregressive (GPT) and autoencoding (BERT)
- Permutation language modeling
- Two-stream self-attention
- Relative positional encodings
- Addresses BERT's [MASK] pre-train/fine-tune mismatch

**190. What are retrieval-augmented models?**
- Combines pre-trained LM with retrieval system
- Retrieves relevant documents during generation
- Examples: REALM, RAG, RETRO
- Reduces hallucination, incorporates up-to-date info
- Scalable alternative to ever-larger models

**191. What is RAG (Retrieval-Augmented Generation)?**
- Seq2seq model with neural retriever
- Retrieves documents from corpus (e.g., Wikipedia)
- Generator conditions on input + retrieved docs
- Trained end-to-end
- Improves factuality, handles knowledge-intensive tasks

**192. Explain Llama model series**
- Meta's open-source LLMs
- LLaMA: 7B-65B, trained on 1.4T tokens, strong performance
- LLaMA 2: Improved, chat-tuned variants, commercial license
- Code Llama: Specialized for code
- Focus on efficiency and accessibility

**193. What is Mistral 7B?**
- 7B parameter model, open-source
- Outperforms larger models (e.g., 13B Llama 2)
- Uses sliding window attention and grouped-query attention
- Very efficient inference
- Strong performance-to-size ratio

**194. What are mixture of experts (MoE) models?**
- Uses multiple "expert" networks
- Router determines which experts to activate per input
- Only subset of parameters active per forward pass
- Enables larger capacity with similar compute
- Examples: Switch Transformer, GLaM

**195. How does sparse attention work?**
- Attention to subset of positions (vs full O(n²))
- Patterns: Local, strided, random
- Reduces complexity to O(n√n) or O(n log n)
- Enables longer sequences
- Examples: Longformer, BigBird

**196. What is Longformer?**
- Transformer with attention patterns for long documents
- Combines local sliding window + global attention + task-specific attention
- O(n) complexity
- Handles 4096+ tokens
- Pre-trained with MLM on long documents

**197. What is FlashAttention?**
- IO-aware attention algorithm
- Fuses attention operations, reduces memory accesses
- Exact attention (not approximation)
- 2-4x speedup and lower memory
- Enables longer sequences and larger batches

### Advanced LLM Topics

**198. What are emergent abilities in LLMs?**
- Capabilities not present in small models, appear suddenly at scale
- Examples: Few-shot learning, chain-of-thought reasoning, instruction following
- Not smoothly increasing with scale
- May be artifact of evaluation metrics (debate ongoing)
- Drives scaling laws research

**199. What are scaling laws in language models?**
- Performance predictably improves with scale (parameters, data, compute)
- Loss follows power law: L(N) ∝ N^-α
- Optimal allocation: Balance model size, data, compute
- Chinchilla: Compute-optimal model is smaller + more data than previously thought
- Guides training decisions

**200. What is the Chinchilla scaling law?**
- Found models are typically undertrained
- Optimal: Equal scaling of model size and training tokens
- Chinchilla (70B): Outperforms larger models (Gopher 280B) with more data
- Implications: Previous models (GPT-3) inefficiently large
- Shifted focus to data quality and quantity

**201. What is model compression for LLMs?**
- Reduces size/compute without much performance loss
- Methods: Pruning, quantization, distillation
- Goals: Deployment on limited hardware, faster inference
- Trade-offs: Accuracy vs efficiency
- Critical for practical deployment

**202. Explain quantization in LLMs**
- Reduces precision of weights/activations (e.g., FP32 → INT8)
- Post-training quantization (PTQ): No retraining
- Quantization-aware training (QAT): Train with quantization
- Can reduce size by 4-8× with minimal loss
- Techniques: Symmetric, asymmetric, per-channel, mixed-precision

**203. What is knowledge distillation?**
- Transfer knowledge from large teacher to small student
- Student trained to mimic teacher's outputs (soft labels)
- Can dramatically reduce size (e.g., DistilBERT: 40% smaller, 97% performance)
- Soft labels provide richer signal than hard labels
- Can combine with other compression techniques

**204. What is pruning in neural networks?**
- Removes unnecessary weights/neurons
- Unstructured: Individual weights (sparse matrices)
- Structured: Entire channels/layers (dense, hardware-friendly)
- Magnitude-based: Remove small weights
- Can achieve 50-90% sparsity with minimal loss

**205. What is the lottery ticket hypothesis?**
- Dense networks contain sparse subnetworks (winning tickets)
- These subnetworks train to comparable accuracy in isolation
- Found via iterative magnitude pruning + retraining from init
- Suggests over-parameterization helps optimization, not just representation
- Sparse networks can be as effective as dense

**206. What are hallucinations in LLMs?**
- Model generates plausible but incorrect/fabricated information
- Causes: Training data noise, lack of knowledge, overconfidence
- Types: Factual errors, nonsensical outputs, inconsistencies
- Mitigation: Retrieval augmentation, citations, uncertainty estimation, RLHF
- Major challenge for deployment

**207. How do you evaluate LLM hallucinations?**
- Factuality benchmarks: TruthfulQA, HaluEval
- Human evaluation: Annotators rate accuracy
- Automatic metrics: Entailment checking, consistency scoring
- Challenge: Defining and detecting subtle errors
- Need for task-specific evaluation

**208. What is model alignment?**
- Making models behave according to human values and intentions
- Challenges: Defining values, scalable oversight, robust generalization
- Techniques: RLHF, Constitutional AI, red teaming
- Increasingly important with powerful models
- Active research area

**209. What is context length and why does it matter?**
- Maximum sequence length model can process
- Limits: Quadratic attention complexity, positional encoding constraints
- Longer context: More information, better coherence
- Models: GPT-3 (2048), GPT-4 (8k/32k), Claude (100k+)
- Requires architectural innovations (sparse attention, ALiBi)

**210. What is ALiBi (Attention with Linear Biases)?**
- Relative positional encoding method
- Adds linear bias to attention scores based on distance
- No positional embeddings needed
- Better extrapolation to longer sequences than absolute positions
- Simple and effective

**211. What is inference-time compute scaling?**
- Using more compute at inference (vs training) for better performance
- Methods: Multiple samples + reranking, search, verification
- Example: Self-consistency uses multiple reasoning paths
- Trade-off: Quality vs cost
- Emerging paradigm for improved outputs

**212. What are tool-using LLMs?**
- Models that can invoke external tools (calculators, search, code execution)
- Examples: Toolformer, ReAct, ChatGPT plugins
- Expands capabilities beyond text generation
- Reduces hallucination for factual queries
- Requires teaching when to use tools

**213. Explain multi-modal LLMs**
- Process multiple modalities (text, image, audio, video)
- Examples: GPT-4V, Gemini, LLaVA
- Architectures: Cross-modal attention, adapter modules, unified embeddings
- Enables richer understanding and generation
- Challenges: Data alignment, modality fusion

**214. What is LLaVA?**
- Large Language and Vision Assistant
- Connects vision encoder (CLIP) to LLM (Vicuna)
- Trained on GPT-4 generated image-text data
- Strong visual instruction following
- Efficient training with open models

**215. What are AI agents with LLMs?**
- LLMs as reasoning engines for autonomous agents
- Components: Planning, memory, tool use, action
- Examples: AutoGPT, BabyAGI
- Challenges: Reliability, safety, credit assignment
- Potential for complex task automation

---

## Transformers & Attention

**216. What is query, key, and value in attention?**
- Query: What am I looking for?
- Key: What do I contain?
- Value: What do I actually output?
- Analogy: Database retrieval (query searches keys, returns values)
- Learned projections: Q=XW_q, K=XW_k, V=XW_v

**217. Why is attention called "attention"?**
- Selectively focuses on relevant parts of input
- Weighted combination based on relevance
- Mimics human selective attention
- Not all inputs equally important
- Weights indicate where model is "attending"

**218. What is the complexity of attention and how to reduce it?**
- Standard: O(n²d) time, O(n²) space for sequence length n
- Bottleneck for long sequences
- Reductions: Sparse attention patterns, low-rank approximation, locality-sensitive hashing
- Kernel methods: Linear attention
- Flash Attention: IO optimization

**219. What is linear attention?**
- Uses kernel trick to avoid explicit softmax
- Complexity: O(nd²) vs O(n²d)
- Enables longer sequences
- Often sacrifice some performance
- Examples: Performer, Linear Transformer

**220. Explain the Performer model**
- Approximates softmax attention with random features
- Uses FAVOR+ algorithm
- O(nd log d) complexity
- Maintains quality close to standard attention
- Enables very long sequences

**221. What is axial attention?**
- Factorizes attention over multiple axes
- For images: Separate row and column attention
- Reduces n² to 2n for 2D
- Generalizes to higher dimensions
- Used in vision transformers for efficiency

**222. What is the Perceiver architecture?**
- Cross-attention to fixed-size latent array
- Decouples input size from computation
- Handles large, high-dimensional inputs efficiently
- Iterative refinement with latent array
- Versatile for different modalities

**223. How do transformers handle variable-length sequences?**
- Padding to max length in batch + attention mask
- Mask: 0 for real tokens, -∞ for padding
- Prevents attention to padding
- Batching efficiency vs computation waste
- Dynamic batching: Group similar lengths

---

## Optimization & Training

### Gradient Descent Variants

**224. Explain batch, mini-batch, and stochastic gradient descent**
- Batch GD: Uses entire dataset per update (slow, memory-intensive)
- Stochastic GD: Uses single sample per update (noisy, fast)
- Mini-batch GD: Uses subset (e.g., 32-256) per update (balanced)
- Mini-batch: Most common (efficient, stable)

**225. What is the learning rate and how to choose it?**
- Step size in parameter updates: θ ← θ - α∇L
- Too large: Divergence; Too small: Slow convergence
- Choice: Grid search, learning rate finder
- Typical range: 1e-5 to 1e-1
- Should decrease over training

**226. Explain momentum in optimization**
- Accumulates velocity in gradient direction
- v ← βv + ∇L, θ ← θ - αv
- Accelerates convergence, dampens oscillations
- Helps escape local minima and navigate ravines
- Typical β = 0.9

**227. What is Nesterov accelerated gradient?**
- Look-ahead momentum: Compute gradient at predicted position
- v ← βv + ∇L(θ - βv), θ ← θ - αv
- Often converges faster than standard momentum
- Better anticipation of gradient changes
- Used in many optimizers

**228. Explain AdaGrad**
- Adaptive learning rate per parameter
- θ ← θ - α/(√G + ε) ⊙ ∇L, where G = Σ(∇L)²
- Larger updates for infrequent features, smaller for frequent
- Good for sparse data
- Problem: Learning rate monotonically decreases (G always grows)

**229. What is RMSProp?**
- Fixes AdaGrad's monotonic decrease
- Exponential moving average of squared gradients
- E[g²] ← βE[g²] + (1-β)g², θ ← θ - α/√(E[g²] + ε) ⊙ g
- Typical β = 0.9
- Works well in practice

**230. Explain Adam optimizer**
- Combines momentum and RMSProp
- First moment: m ← β₁m + (1-β₁)g
- Second moment: v ← β₂v + (1-β₂)g²
- Bias correction: m̂ = m/(1-β₁^t), v̂ = v/(1-β₂^t)
- Update: θ ← θ - α * m̂/(√v̂ + ε)
- Most popular optimizer; default choice

**231. What are the typical hyperparameters for Adam?**
- α (learning rate): 1e-3 (default), tune 1e-4 to 1e-2
- β₁ (momentum): 0.9
- β₂ (RMSProp): 0.999
- ε (numerical stability): 1e-8
- Often only learning rate is tuned

**232. What is AdamW?**
- Adam with decoupled weight decay
- Weight decay applied separately from gradient update
- Original Adam: Weight decay coupled with gradient
- AdamW: Better regularization
- Now standard in transformer training

**233. What are learning rate schedules?**
- Time-varying learning rate
- Common: Step decay, exponential decay, cosine annealing, warm-up
- Helps convergence: Large LR early (exploration), small LR late (refinement)
- Warmup: Gradual increase from small LR (stabilizes early training)

**234. What is learning rate warm-up?**
- Gradually increase LR from 0 to target over initial steps
- Prevents instability from large gradients early in training
- Especially important for large batch sizes and transformers
- Typical: Linear or exponential increase over 1-10% of training

**235. What is cosine annealing?**
- LR follows cosine curve from max to min
- α(t) = α_min + 0.5(α_max - α_min)(1 + cos(πt/T))
- Smooth decay, no manual schedule
- Can restart (cosine annealing with restarts)
- Popular in modern training

**236. What is cyclical learning rate?**
- LR oscillates between bounds
- Helps escape local minima
- Can speed up training
- Super-convergence: 1cycle policy (increase then decrease)
- Requires careful tuning

**237. What is gradient clipping?**
- Caps gradient norm to prevent exploding gradients
- Norm clipping: g ← g * threshold/||g|| if ||g|| > threshold
- Value clipping: g ← clip(g, -threshold, threshold)
- Essential for RNNs and transformers
- Typical threshold: 1.0

**238. What is gradient accumulation?**
- Accumulate gradients over multiple forward passes before update
- Simulates larger batch size than fits in memory
- Update: θ ← θ - α * Σ∇L_i / K
- Linear trade-off: K accumulation steps = K× memory for batch
- Batch norm statistics computed per mini-batch, not accumulated batch

**239. What is mixed precision training?**
- Uses both FP16 and FP32 during training
- FP16 for forward/backward (faster, less memory)
- FP32 for parameter updates and certain operations
- Loss scaling to prevent FP16 underflow
- 2-3× speedup with minimal accuracy impact

**240. What is the vanishing learning rate problem in Adam?**
- Adaptive learning rate can become too small
- Caused by accumulated second moments
- Especially problematic in later training stages
- Solutions: Learning rate schedules, AdaBound, RAdam

**241. What is second-order optimization?**
- Uses Hessian (second derivatives) for better curvature information
- Newton's method: θ ← θ - H⁻¹∇L
- Pros: Faster convergence (fewer iterations)
- Cons: Hessian computation/inversion expensive (O(n³))
- Approximations: L-BFGS, natural gradient

**242. What is natural gradient descent?**
- Accounts for parameter space geometry
- Uses Fisher information matrix (approximation of Hessian)
- θ ← θ - αF⁻¹∇L
- More invariant to parameterization
- K-FAC: Practical approximation for deep learning

### Training Techniques

**243. What is curriculum learning?**
- Training order from easy to hard examples
- Mimics human learning
- Can speed up convergence and improve final performance
- Challenge: Defining "difficulty"
- Examples: Sentence length, task complexity

**244. What is self-supervised learning?**
- Learning from unlabeled data using pretext tasks
- Labels derived from data structure
- Examples: MLM, next token prediction, contrastive learning
- Foundation of modern pre-training
- Scales with unlabeled data

**245. What is contrastive learning?**
- Learn by contrasting positive and negative pairs
- Positive: Augmented versions of same sample
- Negative: Different samples
- Objective: Bring positives closer, push negatives apart
- Examples: SimCLR, MoCo, CLIP

**246. Explain SimCLR**
- Simple Framework for Contrastive Learning
- Two augmented views of same image
- Contrastive loss in latent space (NT-Xent)
- Large batch sizes crucial (more negatives)
- Strong data augmentation critical

**247. What is knowledge distillation?**
- Training small student to mimic large teacher
- Student learns from teacher's soft predictions (temperature-scaled softmax)
- Soft labels contain more information than hard labels
- Can distill ensemble into single model
- Applications: Model compression, transfer learning

**248. What is label smoothing?**
- Replaces one-hot labels with soft labels
- y_smooth = (1-ε)y + ε/K where K = # classes
- Prevents overconfidence
- Regularization effect
- Typical ε = 0.1

**249. What is mixup?**
- Creates synthetic training examples by interpolating
- x̃ = λxᵢ + (1-λ)xⱼ, ỹ = λyᵢ + (1-λ)yⱼ
- λ ~ Beta(α,α), typically α=0.2
- Regularization: Smooths decision boundaries
- Works across domains (vision, NLP)

**250. What is cutmix?**
- Mixes images by cutting and pasting patches
- Label mixed proportionally to patch area
- Stronger augmentation than mixup
- More localized mixing
- Popular in computer vision

**251. What is data augmentation?**
- Artificially expanding training set with transformed data
- Vision: Crop, flip, rotate, color jitter
- NLP: Back-translation, word substitution, paraphrasing
- Reduces overfitting, improves generalization
- Task-specific: Augmentations must preserve semantics

**252. What is AutoAugment?**
- Learns augmentation policy via reinforcement learning
- Searches over augmentation operations and probabilities
- Dataset-specific policies
- Expensive search but transferable
- Improved: RandAugment (simpler, fewer hyperparameters)

**253. What is hard negative mining?**
- Select difficult negative examples for training
- Common in metric learning, retrieval, detection
- Improves discrimination on hard cases
- Risk: Focus too much on outliers/noise
- Balance with random negatives

**254. What is focal loss?**
- Addresses class imbalance by focusing on hard examples
- FL(pₜ) = -α(1-pₜ)^γ log(pₜ)
- γ: Focusing parameter (typical: 2)
- Down-weights easy examples (high pₜ)
- Designed for object detection, useful elsewhere

**255. What is noise contrastive estimation (NCE)?**
- Approximates softmax by binary classification
- Distinguish data from noise distribution
- Reduces complexity from O(V) to O(k) where k = noise samples
- Used in word embeddings (Word2Vec)
- Related to negative sampling

**256. What is importance sampling?**
- Sample proportionally to importance (not uniformly)
- Reduces variance in gradient estimation
- Biases toward informative examples
- Used in off-policy RL, attention mechanisms
- Requires careful reweighting

---

## Regularization & Generalization

**257. What is L1 regularization (Lasso)?**
- Penalty: λΣ|θᵢ|
- Encourages sparsity (many weights exactly 0)
- Feature selection property
- Non-differentiable at 0 (use subgradient)
- Tends to produce few large weights

**258. What is L2 regularization (Ridge)?**
- Penalty: λΣθᵢ²
- Encourages small weights (distributed)
- Equivalent to Gaussian prior on weights
- Smooth, differentiable everywhere
- Tends to produce many small weights

**259. What is elastic net regularization?**
- Combines L1 and L2: λ₁Σ|θᵢ| + λ₂Σθᵢ²
- Gets benefits of both: Sparsity + grouping effect
- Better when features are correlated
- Two hyperparameters to tune
- Often better than L1 or L2 alone

**260. What is weight decay?**
- Multiplicative weight shrinkage: θ ← (1-λ)θ
- Equivalent to L2 regularization for SGD
- Not exactly equivalent for Adam (hence AdamW)
- Prevents weights from growing unbounded
- Typical λ: 1e-4 to 1e-2

**261. What is early stopping?**
- Stop training when validation performance stops improving
- Implicit regularization (limits effective capacity)
- Requires validation set and patience parameter
- Simple, effective
- Risk: May stop too early or oscillate

**262. What is dropout as regularization?**
- Randomly dropping units prevents co-adaptation
- Equivalent to training ensemble of subnetworks
- Reduces overfitting
- Bayesian interpretation: Approximate posterior inference
- Typical rate: 0.5 for hidden layers, 0.1-0.2 for input

**263. What is dropconnect?**
- Drops weights instead of activations (vs dropout)
- More aggressive regularization
- Less commonly used than dropout
- Can be combined with dropout
- Similar benefits but more expensive

**264. What is spectral normalization?**
- Normalizes weights by spectral norm (largest singular value)
- W_SN = W / σ(W)
- Stabilizes training, especially for GANs
- Controls Lipschitz constant
- Preserves information through layers

**265. What is data-dependent regularization?**
- Regularization strength varies with input
- Examples: Adaptive dropout, confidence penalty
- Can be more effective than fixed regularization
- Requires careful design
- More complex to implement

**266. What is the double descent phenomenon?**
- Test error decreases, increases, then decreases again with model capacity
- Contradicts classical bias-variance trade-off
- Occurs in overparameterized regime
- Related to implicit regularization of gradient descent
- Active research topic

**267. What is implicit regularization?**
- Regularization from optimization, not explicit penalty
- SGD biases toward flat minima
- Architecture choices (e.g., depth) regularize
- Large batch size reduces implicit regularization
- Helps explain generalization in deep learning

**268. What is the lottery ticket hypothesis and pruning?**
- Dense networks contain sparse subnetworks (winning tickets)
- These subnetworks can train to full accuracy
- Implies over-parameterization aids optimization
- Iterative magnitude pruning finds tickets
- Challenges conventional wisdom on network size

---

## Model Evaluation & Metrics

### Classification Metrics

**269. What is accuracy and when is it misleading?**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Misleading with class imbalance (e.g., 95% accuracy on 95% majority class)
- Doesn't distinguish error types
- Better: Balanced accuracy, F1, AUC
- Good for balanced classes only

**270. What are precision and recall?**
- Precision = TP / (TP + FP) - Of predicted positives, how many are correct?
- Recall = TP / (TP + FN) - Of actual positives, how many are found?
- Trade-off: Can improve one at expense of other
- Threshold-dependent for probabilistic classifiers
- Application-specific importance (medical: high recall; spam: high precision)

**271. What is the F1 score?**
- Harmonic mean of precision and recall
- F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN)
- Balances precision and recall equally
- Better than accuracy for imbalanced data
- F-beta: Weighs precision/recall differently

**272. What is the confusion matrix?**
- Table of predicted vs actual classes
- Diagonal: Correct predictions; Off-diagonal: Errors
- Provides detailed error analysis
- Multi-class extension straightforward
- Foundation for many metrics

**273. What is the ROC curve?**
- Receiver Operating Characteristic
- Plots TPR (recall) vs FPR at various thresholds
- FPR = FP / (FP + TN)
- Shows performance across all thresholds
- Threshold-independent evaluation

**274. What is AUC (Area Under ROC Curve)?**
- Summarizes ROC curve with single number
- AUC = 0.5: Random classifier; AUC = 1.0: Perfect
- Interpretation: Probability random positive ranks higher than random negative
- Robust to class imbalance
- Commonly used in binary classification

**275. What is the PR (Precision-Recall) curve?**
- Plots precision vs recall at various thresholds
- More informative than ROC for imbalanced data
- AUPRC: Area under PR curve
- Baseline: Random classifier at positive rate
- Preferred for rare positive class

**276. When to use ROC-AUC vs PR-AUC?**
- Balanced data or care about both classes: ROC-AUC
- Imbalanced data or focus on positive class: PR-AUC
- ROC-AUC more optimistic with imbalance
- PR-AUC more sensitive to improvements in positive class
- Can use both for comprehensive evaluation

**277. What is log loss (binary cross-entropy)?**
- Measures calibration of predicted probabilities
- LogLoss = -Σ[y log(p) + (1-y)log(1-p)]
- Penalizes confident wrong predictions heavily
- Lower is better
- Differentiable (used for training)

**278. What is Cohen's Kappa?**
- Agreement between predictions and truth, accounting for chance
- κ = (p_o - p_e) / (1 - p_e)
- p_o: Observed agreement; p_e: Expected agreement by chance
- Range: [-1, 1]; 1 = perfect, 0 = chance
- Used for inter-rater reliability

**279. What is Matthews Correlation Coefficient (MCC)?**
- Correlation between predicted and actual binary classifications
- MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
- Range: [-1, 1]; 1 = perfect, 0 = random, -1 = complete disagreement
- Robust to class imbalance
- Accounts for all four confusion matrix categories

**280. What is macro vs micro vs weighted averaging?**
- Macro: Compute metric per class, then average (equal weight to classes)
- Micro: Aggregate contributions, then compute metric (equal weight to instances)
- Weighted: Macro weighted by class frequency
- Micro favors majority class; Macro treats all equal
- Choice depends on whether all classes equally important

### Regression Metrics

**281. What is Mean Squared Error (MSE)?**
- MSE = (1/n)Σ(yᵢ - ŷᵢ)²
- Penalizes large errors heavily (squared term)
- Same units as target squared
- Sensitive to outliers
- Commonly used loss function

**282. What is Root Mean Squared Error (RMSE)?**
- RMSE = √MSE
- Same units as target
- Interpretable scale
- Still sensitive to outliers
- Common evaluation metric

**283. What is Mean Absolute Error (MAE)?**
- MAE = (1/n)Σ|yᵢ - ŷᵢ|
- Linear penalty (vs squared in MSE)
- More robust to outliers
- Same units as target
- Less emphasis on large errors

**284. What is R² (coefficient of determination)?**
- R² = 1 - SS_res/SS_tot
- Proportion of variance explained
- Range: (-∞, 1]; 1 = perfect, 0 = predicts mean
- Can be negative (worse than mean predictor)
- Popular for interpretability

**285. What is adjusted R²?**
- Penalizes additional features
- R²_adj = 1 - (1-R²)(n-1)/(n-p-1)
- Never increases with irrelevant features
- Better for model comparison
- Used in feature selection

**286. What is Mean Absolute Percentage Error (MAPE)?**
- MAPE = (100/n)Σ|yᵢ - ŷᵢ|/|yᵢ|
- Percentage error (scale-independent)
- Interpretable
- Problem: Undefined when yᵢ = 0, biased toward underestimates
- Asymmetric penalty

**287. What is Huber loss?**
- Combines MSE and MAE: Quadratic for small errors, linear for large
- δ parameter controls transition
- More robust than MSE, smoother than MAE
- Used in regression and RL
- Good for noisy data with outliers

### Model Comparison & Selection

**288. What is cross-validation?**
- Splits data into k folds, trains k times (each fold as validation once)
- Estimates generalization performance
- K-fold (common: k=5,10), leave-one-out (k=n)
- Reduces variance in performance estimate
- Computationally expensive

**289. What is stratified k-fold?**
- Maintains class distribution in each fold
- Important for imbalanced data
- Ensures each fold is representative
- Reduces variance in CV estimates
- Standard for classification

**290. What is nested cross-validation?**
- Outer loop: Performance estimation
- Inner loop: Hyperparameter tuning
- Avoids optimistic bias from tuning on test set
- More accurate generalization estimate
- Expensive: k_outer × k_inner × models

**291. What is the bias-variance decomposition of CV?**
- K-fold: Lower bias than train-test split, higher than leave-one-out
- Leave-one-out: Low bias, high variance (highly correlated folds)
- 5-10 fold: Good bias-variance trade-off
- Choice balances accuracy and variance

**292. What is a validation set vs test set?**
- Validation: Used during training for hyperparameter tuning, early stopping
- Test: Held out until final evaluation (never influences training)
- Train/validation/test split: E.g., 60/20/20 or 70/15/15
- Test set estimates real-world performance
- Validation set part of development process

**293. What is data leakage?**
- Information from test set influencing training
- Examples: Preprocessing on full data, temporal leakage, feature leakage
- Causes overly optimistic performance estimates
- Prevented by strict train-test separation
- Common mistake in practice

**294. What is temporal validation for time series?**
- Train on past, validate/test on future
- No random shuffling (breaks temporal structure)
- Walk-forward validation: Rolling origin, expanding window
- Accounts for non-stationarity
- Critical for forecasting

**295. What is the bootstrap method?**
- Resampling with replacement
- Creates multiple datasets from original
- Estimates sampling distribution
- Confidence intervals: Percentile, BCa
- Out-of-bag samples can serve as validation

**296. What is statistical significance testing for model comparison?**
- Tests: Paired t-test, Wilcoxon signed-rank
- Multiple comparisons: Bonferroni, Benjamini-Hochberg correction
- Effect size: Cohen's d
- P-value doesn't measure practical importance
- Consider confidence intervals

**297. What is the McNemar test?**
- Tests whether two classifiers have same error rate
- Uses contingency table of correct/incorrect on each sample
- Paired test (same test set)
- χ² distribution with 1 degree of freedom
- Common for model comparison

**298. What is a learning curve?**
- Plots performance vs training set size
- Diagnoses bias (underfitting) vs variance (overfitting)
- High bias: Curves converge, both low
- High variance: Large gap between train and validation
- Informs whether more data helps

**299. What is model calibration?**
- Predicted probabilities match true frequencies
- Calibrated: If predict 70%, correct 70% of time
- Calibration plot: Predicted probability vs observed frequency
- Metrics: Expected Calibration Error (ECE), Brier score
- Can calibrate post-hoc: Platt scaling, isotonic regression

**300. What is Expected Calibration Error (ECE)?**
- Measures miscalibration
- ECE = Σ(n_b/n)|acc(b) - conf(b)|
- Bins predictions by confidence, compares to accuracy
- Lower is better
- Important for decision-making applications

---

## Feature Engineering

**301. What is feature scaling and when is it necessary?**
- Rescaling features to similar ranges
- Necessary: Distance-based (KNN, SVM, K-means), gradient descent (NNs), regularization
- Not necessary: Tree-based methods
- Types: Normalization, standardization
- Prevents features with large scale from dominating

**302. What is normalization (min-max scaling)?**
- x' = (x - x_min) / (x_max - x_min)
- Scales to [0, 1] (or specified range)
- Preserves relationships
- Sensitive to outliers (affects range)
- Common in neural networks

**303. What is standardization (z-score normalization)?**
- x' = (x - μ) / σ
- Zero mean, unit variance
- Not bounded to specific range
- Less affected by outliers than min-max
- Assumes roughly normal distribution

**304. What is robust scaling?**
- Uses median and IQR instead of mean and std
- x' = (x - median) / IQR
- Robust to outliers
- Good for data with extreme values
- Doesn't assume distribution

**305. What is log transformation?**
- x' = log(x) or log(x+1)
- Reduces right skewness
- Makes multiplicative relationships additive
- Stabilizes variance
- Only for positive values

**306. What is Box-Cox transformation?**
- Parametric family of power transformations
- Finds optimal λ to normalize distribution
- y^(λ) if λ≠0, log(y) if λ=0
- More flexible than simple log
- Requires positive values

**307. What is one-hot encoding?**
- Creates binary column for each category
- Example: [red, blue, green] → [1,0,0], [0,1,0], [0,0,1]
- Increases dimensionality (# categories)
- No ordinal assumption
- Dummy variable trap: Use k-1 columns (drop one reference)

**308. What is label encoding?**
- Maps categories to integers: [red, blue, green] → [0, 1, 2]
- Reduces dimensionality
- Implies ordinality (may mislead algorithms)
- Good for ordinal data, tree-based methods
- Avoid for non-ordinal with distance-based methods

**309. What is target encoding (mean encoding)?**
- Replace category with target mean for that category
- Example: City → Average income in city
- Reduces dimensionality for high-cardinality features
- Risk of overfitting: Use smoothing, cross-validation
- Captures relationship with target

**310. What is frequency encoding?**
- Replace category with its frequency/count
- Preserves information about rarity
- Doesn't capture target relationship
- Simple, no overfitting risk
- Good for high-cardinality features

**311. What is feature hashing (hashing trick)?**
- Hash categorical features to fixed-size vector
- Avoids storing explicit mapping
- Hash collisions are acceptable (projected to same dimension)
- Memory-efficient for high-cardinality
- Used in online learning, NLP

**312. What is feature interaction?**
- Creating features by combining existing ones
- Multiplicative: x₁ × x₂; Polynomial: x₁² + x₁x₂ + x₂²
- Captures non-linear relationships
- Increases dimensionality (curse of dimensionality)
- Can be learned by neural networks

**313. What is polynomial feature expansion?**
- Generate polynomial terms up to degree d
- Example (d=2): [x₁, x₂] → [1, x₁, x₂, x₁², x₁x₂, x₂²]
- Increases model flexibility
- Exponential feature growth with degree
- Regularization crucial

**314. What is binning (discretization)?**
- Convert continuous to categorical
- Equal-width: Bins of equal range
- Equal-frequency: Bins with equal counts (quantiles)
- Can improve robustness, capture non-linear patterns
- Loss of information, sensitivity to bin boundaries

**315. What are time-based features?**
- Hour, day, month, year, day_of_week
- Cyclical encoding: sin/cos for periodic features
- Time since event, time until event
- Rolling statistics: Moving average, std
- Lag features for time series

**316. What is feature selection vs feature extraction?**
- Selection: Choose subset of original features
- Extraction: Transform features to new space (e.g., PCA)
- Selection: Interpretability preserved; Extraction: New features may not be interpretable
- Selection: Addresses curse of dimensionality while keeping original features

**317. What are filter methods for feature selection?**
- Rank features by statistical scores (independent of model)
- Examples: Correlation, mutual information, chi-squared test, ANOVA
- Fast, scalable
- Ignores feature interactions
- Independent of learning algorithm

**318. What are wrapper methods for feature selection?**
- Use model performance to evaluate feature subsets
- Examples: Forward selection, backward elimination, recursive feature elimination (RFE)
- Accounts for feature interactions and model specifics
- Computationally expensive
- Risk of overfitting to validation set

**319. What are embedded methods for feature selection?**
- Feature selection during model training
- Examples: L1 regularization (Lasso), tree feature importance
- More efficient than wrapper methods
- Model-specific
- Good balance of performance and computation

**320. What is recursive feature elimination (RFE)?**
- Iteratively removes least important features
- Train model, rank features, remove lowest, repeat
- Uses model's internal feature importance
- Can find good subset with fewer evaluations than forward/backward
- Requires model that provides feature importance

---

## Ensemble Methods

**321. What is ensemble learning?**
- Combining multiple models for better performance
- Wisdom of crowds: Diverse models correct each other's errors
- Reduces variance, improves robustness
- Types: Bagging, boosting, stacking
- Often wins competitions

**322. What is the difference between bagging and boosting?**
- Bagging: Parallel, reduces variance, independent models (Random Forest)
- Boosting: Sequential, reduces bias, models focus on errors of previous (XGBoost)
- Bagging: Sample with replacement; Boosting: Reweight samples
- Bagging: Equal weight; Boosting: Weighted combination

**323. Explain bootstrap aggregating (bagging)**
- Train models on bootstrap samples (sampling with replacement)
- Aggregate predictions (vote/average)
- Reduces variance (especially high-variance models like deep trees)
- Out-of-bag samples for validation
- Parallelizable

**324. What are out-of-bag (OOB) samples?**
- Samples not included in a bootstrap sample (~37% not selected)
- Can be used for validation without separate validation set
- Each sample is OOB for some trees
- OOB score: Validation performance estimate
- Computationally efficient alternative to cross-validation

**325. How does boosting work?**
- Sequentially train weak learners
- Each focuses on samples previous models struggled with
- Combine with weighted voting
- Reduces bias and variance
- Examples: AdaBoost, Gradient Boosting

**326. What is AdaBoost?**
- Adaptive Boosting
- Increases weights of misclassified samples
- Combines weak learners with weighted vote (based on accuracy)
- Exponential loss function
- Sensitive to noisy data and outliers

**327. What is the difference between gradient boosting and AdaBoost?**
- AdaBoost: Reweights samples; Gradient Boosting: Fits residuals
- AdaBoost: Exponential loss; GB: General loss functions
- GB: More flexible, handles regression, various objectives
- GB: More popular in practice

**328. What is stacking (stacked generalization)?**
- Train meta-model on predictions of base models
- Base models: Level-0; Meta-model: Level-1
- Meta-model learns how to combine base models
- Can use cross-validated predictions to avoid overfitting
- More complex than bagging/boosting

**329. What is blending?**
- Simpler version of stacking
- Holdout set for meta-model (vs cross-validation in stacking)
- Base models train on train set, predict on holdout
- Meta-model trains on holdout predictions
- Less computationally expensive than stacking

**330. What is voting ensemble?**
- Combines predictions by voting (classification) or averaging (regression)
- Hard voting: Majority vote on class labels
- Soft voting: Average predicted probabilities, then classify
- Soft voting usually better (uses confidence information)
- Simple, effective baseline ensemble

**331. What makes a good ensemble?**
- Diversity: Models make different errors
- Accuracy: Individual models better than random
- Methods to increase diversity: Different algorithms, different hyperparameters, different subsets of data/features
- Trade-off: More diversity vs individual accuracy
- Diminishing returns with too many similar models

**332. How many models should be in an ensemble?**
- Bagging: Diminishing returns after ~50-100 trees
- Boosting: More iterations can overfit (use early stopping)
- Stacking: 5-10 diverse base models typically
- More models: Better performance, more computation, more overfitting risk
- Monitor validation performance to decide

**333. What is snapshot ensembling?**
- Train single model with cyclic learning rate
- Save snapshots at local minima
- Ensemble of snapshots
- Computationally efficient (one training run)
- Works because different minima provide diversity

**334. What is fast geometric ensembling (FGE)?**
- Ensemble points along low-loss path in weight space
- Uses cyclic learning rate
- More efficient than training separate models
- Connects local minima with low-loss path
- Provides diversity with minimal overhead

---

## Unsupervised Learning

**335. What is the difference between supervised and unsupervised learning?**
- Supervised: Labeled data, learn input→output mapping, evaluation straightforward
- Unsupervised: Unlabeled data, find patterns/structure, evaluation subjective
- Semi-supervised: Mix of labeled and unlabeled
- Self-supervised: Labels derived from data structure

**336. What is dimensionality reduction?**
- Reducing number of features while preserving information
- Benefits: Visualization, noise reduction, computation, storage
- Methods: PCA, t-SNE, UMAP, autoencoders
- Trade-off: Information loss vs compression
- Linear (PCA) vs non-linear (t-SNE)

**337. Explain Principal Component Analysis (PCA) mathematically**
- Find directions of maximum variance
- Eigenvalue decomposition of covariance matrix: Σ = VΛV^T
- PCs are eigenvectors; variance along PC is eigenvalue
- Project data: Z = XV (where V has top-k eigenvectors)
- Minimizes reconstruction error for linear projection

**338. How do you choose number of components in PCA?**
- Scree plot: Look for elbow in eigenvalue curve
- Cumulative explained variance: Threshold (e.g., 95%)
- Cross-validation: Use downstream task performance
- Kaiser criterion: Keep components with eigenvalue > 1
- Domain knowledge, computational constraints

**339. What is kernel PCA?**
- Non-linear version using kernel trick
- Projects to high-dimensional space, then PCA
- Captures non-linear relationships
- Common kernels: RBF, polynomial
- No explicit mapping needed (kernel trick)

**340. What is sparse PCA?**
- PCA with sparsity constraint on loadings
- Most loadings are zero (interpretability)
- Trade-off: Explained variance vs sparsity
- Better interpretability than standard PCA
- Solved via optimization (e.g., elastic net)

**341. What is Linear Discriminant Analysis (LDA)?**
- Supervised dimensionality reduction
- Maximizes between-class variance, minimizes within-class variance
- Finds linear combinations that separate classes
- Maximum k-1 components for k classes
- Assumes Gaussian with equal covariance

**342. What is Independent Component Analysis (ICA)?**
- Finds independent sources from mixed signals
- Assumes sources are statistically independent and non-Gaussian
- Applications: Blind source separation (cocktail party problem)
- Non-Gaussian: Gaussian components unidentifiable
- Order and scale ambiguity

**343. What is Non-negative Matrix Factorization (NMF)?**
- Factorizes non-negative matrix: V ≈ WH (W, H ≥ 0)
- Parts-based representation (additive only)
- Applications: Topic modeling, image processing
- Sparsity and interpretability
- Various update algorithms (multiplicative update, alternating least squares)

**344. What is t-SNE (t-Distributed Stochastic Neighbor Embedding)?**
- Non-linear dimensionality reduction for visualization
- Preserves local structure (neighbors stay neighbors)
- t-distribution in low-dimensional space (heavy tails)
- Computationally expensive: O(n²)
- Hyperparameters: Perplexity (local neighborhood size)

**345. What is perplexity in t-SNE?**
- Balances local vs global aspects of data
- Rough measure of effective number of neighbors
- Typical range: 5-50
- Lower: Focus on local structure; Higher: Global structure
- Should try multiple values

**346. What is UMAP (Uniform Manifold Approximation and Projection)?**
- Alternative to t-SNE with theoretical foundation
- Preserves global structure better than t-SNE
- Faster than t-SNE
- Hyperparameters: n_neighbors, min_dist
- Based on Riemannian geometry and topology

**347. What is an autoencoder?**
- Neural network that learns compressed representation
- Encoder: Input → Latent; Decoder: Latent → Reconstruction
- Trained to minimize reconstruction error
- Unsupervised (uses input as target)
- Generalization of PCA (non-linear with layers)

**348. What is a variational autoencoder (VAE)?**
- Probabilistic autoencoder
- Encodes distribution (mean, variance) not point
- Latent space has prior (typically Gaussian)
- Loss: Reconstruction + KL divergence
- Can generate new samples

**349. What is the reparameterization trick in VAEs?**
- Enables backpropagation through sampling
- z = μ + σ ⊙ ε, where ε ~ N(0,1)
- Gradients flow through μ and σ
- Without: Sampling is non-differentiable
- Key to VAE training

**350. What is anomaly detection?**
- Identifying unusual/rare instances
- Applications: Fraud, network intrusion, manufacturing defects
- Methods: Statistical (Z-score), density-based (LOF), isolation forest, autoencoders
- Challenge: Imbalanced (few anomalies), unlabeled
- One-class vs supervised approaches

**351. What is Local Outlier Factor (LOF)?**
- Density-based anomaly detection
- Compares local density to neighbors' density
- LOF >> 1: Outlier; LOF ≈ 1: Normal
- Captures local anomalies (may be normal globally)
- Parameter: Number of neighbors k

**352. What is Isolation Forest?**
- Ensemble of trees for anomaly detection
- Anomalies are easier to isolate (fewer splits)
- Path length in tree measures normality
- Fast, scalable, no distance computation
- Works well in high dimensions

**353. What is one-class SVM?**
- Learns boundary around normal data
- Treats origin as sole outlier
- Kernel for non-linear boundaries
- Requires mostly normal data for training
- ν parameter controls outlier fraction

**354. What is association rule mining?**
- Finding frequent patterns/rules in transactional data
- Example: {milk, bread} → {butter}
- Measures: Support (frequency), confidence (conditional probability), lift (correlation)
- Applications: Market basket analysis, recommendation
- Algorithms: Apriori, FP-Growth

**355. What is the Apriori algorithm?**
- Finds frequent itemsets via breadth-first search
- Apriori principle: Subsets of frequent sets are frequent
- Prunes candidate sets early
- Generates association rules from frequent itemsets
- Can be slow with many items

---

## Reinforcement Learning

**356. What is reinforcement learning?**
- Learning by interaction with environment
- Agent takes actions, receives rewards, learns policy
- Goal: Maximize cumulative reward
- Differs from supervised (no correct action given) and unsupervised (reward signal)
- Applications: Games, robotics, recommendation

**357. What is the Markov Decision Process (MDP)?**
- Mathematical framework for RL
- Components: States, actions, transition probabilities, rewards, discount factor
- Markov property: Future depends only on current state
- Policy: Mapping state→action
- Goal: Find optimal policy π*

**358. What is the difference between value function and Q-function?**
- Value function V(s): Expected return from state s under policy π
- Q-function Q(s,a): Expected return from state s, taking action a, then following π
- V(s) = max_a Q(s,a) for optimal policy
- Q enables action selection without model
- Bellman equations relate current to future values

**359. What is the Bellman equation?**
- Recursive relationship for value function
- V(s) = max_a [R(s,a) + γΣP(s'|s,a)V(s')]
- Q(s,a) = R(s,a) + γΣP(s'|s,a)max_a' Q(s',a')
- Basis for many RL algorithms
- Optimality: V* satisfies Bellman optimality equation

**360. What is the discount factor γ?**
- Weights future rewards: Total return = Σγ^t r_t
- γ ∈ [0,1]; γ=0: Myopic; γ→1: Far-sighted
- Ensures convergence (infinite horizon)
- Reflects preference for immediate rewards
- Typical: 0.9-0.99

**361. What is the explore-exploit trade-off?**
- Exploration: Try new actions (gather information)
- Exploitation: Choose best known action (maximize reward)
- Pure exploitation: May miss better options
- Pure exploration: Never uses knowledge
- Strategies: ε-greedy, UCB, Thompson sampling

**362. What is ε-greedy exploration?**
- Exploit best action with probability 1-ε
- Explore random action with probability ε
- Simple, effective
- ε decays over time (more exploitation later)
- Doesn't account for uncertainty in estimates

**363. What is model-free vs model-based RL?**
- Model-free: Learn policy/value directly from experience (Q-learning, policy gradient)
- Model-based: Learn environment model (transition, reward), then plan (Dyna, AlphaZero)
- Model-free: Sample-inefficient, no model errors; Model-based: Sample-efficient, model bias
- Hybrid approaches combine benefits

**364. What is Q-learning?**
- Off-policy, model-free value-based RL
- Update: Q(s,a) ← Q(s,a) + α[r + γmax_a' Q(s',a') - Q(s,a)]
- Learns optimal Q-function Q*
- Doesn't require policy to generate optimal policy
- Can diverge with function approximation

**365. What is SARSA?**
- On-policy alternative to Q-learning
- Update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- Uses actual action a' taken (not max)
- More conservative than Q-learning
- Learns value of policy being followed

**366. What is Deep Q-Network (DQN)?**
- Q-learning with deep neural network as Q-function approximator
- Innovations: Experience replay, target network
- Stabilizes training (Q-learning + function approximation is unstable)
- Enabled RL on high-dimensional states (images)
- Breakthrough: Atari games

**367. What is experience replay?**
- Stores transitions in buffer, samples mini-batches for training
- Breaks correlation between consecutive samples
- Enables reuse of experience (sample efficiency)
- More stable training
- Uniform or prioritized sampling

**368. What is a target network in DQN?**
- Separate network for computing target values
- Periodically copied from main network
- Prevents moving target problem (chasing own tail)
- Stabilizes training
- Soft update (Polyak averaging) or hard update

**369. What is policy gradient?**
- Directly optimize policy parameters
- Gradient: ∇_θJ(θ) = E[∇_θ log π_θ(a|s) Q(s,a)]
- Advantage: Works for continuous actions, stochastic policies
- High variance (requires many samples)
- REINFORCE: Basic policy gradient algorithm

**370. What is the REINFORCE algorithm?**
- Monte Carlo policy gradient
- Update after full episode: θ ← θ + α∇_θ log π_θ(a|s)G_t
- G_t: Return from time t
- High variance, unbiased
- Baseline reduces variance without bias

**371. What is the actor-critic architecture?**
- Actor: Policy (selects actions)
- Critic: Value function (evaluates actions)
- Actor updated with policy gradient, using critic for advantage
- Lower variance than pure policy gradient
- Combines benefits of value-based and policy-based methods

**372. What is Advantage Actor-Critic (A2C/A3C)?**
- Uses advantage function: A(s,a) = Q(s,a) - V(s)
- Advantage reduces variance (baseline V(s))
- A3C: Asynchronous parallel actors (faster training)
- A2C: Synchronous version (more stable)
- State-of-art for continuous control

**373. What is Proximal Policy Optimization (PPO)?**
- Constrains policy update size for stability
- Clipped objective: Prevents too-large updates
- Simpler than TRPO, similar performance
- Most popular policy gradient method
- Default choice for many applications

**374. What is Trust Region Policy Optimization (TRPO)?**
- Constrains update in distribution space (KL divergence)
- Guarantees monotonic improvement
- More complex than PPO (second-order optimization)
- Inspired PPO (simpler approximation)
- Strong theoretical guarantees

**375. What is the multi-armed bandit problem?**
- Simplified RL: Single state, multiple actions
- Trade-off: Explore arms vs exploit best
- Applications: A/B testing, ad placement, clinical trials
- Algorithms: UCB, Thompson sampling, ε-greedy
- Foundation for exploration strategies

**376. What is Upper Confidence Bound (UCB)?**
- Select action with highest upper confidence bound
- Balances exploitation (mean reward) and exploration (uncertainty)
- a_t = argmax[Q̄(a) + c√(ln t / N(a))]
- Theoretical guarantees (regret bounds)
- No parameters to tune (vs ε-greedy)

**377. What is Thompson sampling?**
- Bayesian approach to exploration
- Maintains belief distribution over action values
- Samples from distributions, selects best sample
- Probability matching: Sample ∝ probability of being optimal
- Often outperforms UCB in practice

**378. What is reward shaping?**
- Providing additional intermediate rewards
- Guides agent toward goal
- Risk: Agent optimizes shaped reward, not true objective
- Potential-based shaping provably preserves optimal policy
- Alternative: Curriculum learning

**379. What is inverse reinforcement learning (IRL)?**
- Infer reward function from expert demonstrations
- Given: Policy (demonstrations); Find: Reward
- Applications: Imitation learning, understanding behavior
- Ill-posed: Many reward functions explain same policy
- MaxEnt IRL: Chooses reward with maximum entropy policy

**380. What is imitation learning?**
- Learn policy from expert demonstrations
- Behavioral cloning: Supervised learning (state→action)
- DAgger: Interactive imitation learning (collects more demos)
- Easier than RL from scratch (no reward engineering)
- Distribution shift: Train and test distributions differ

---

## MLOps & Production

**381. What is MLOps?**
- Practices for deploying and maintaining ML systems in production
- Combines ML, DevOps, and data engineering
- Includes: Versioning, monitoring, automation, CI/CD
- Goals: Reliability, reproducibility, scalability
- Critical for production ML

**382. What is model versioning?**
- Tracking different model versions (architecture, hyperparameters, weights)
- Enables rollback, A/B testing, reproducibility
- Tools: MLflow, DVC, Weights & Biases
- Version data, code, configs, and models
- Essential for collaboration and debugging

**383. What is data versioning?**
- Tracking versions of training/validation data
- Challenges: Large size, frequent updates
- Tools: DVC, LakeFS, Pachyderm
- Critical for reproducibility
- Links model performance to data version

**384. What is feature store?**
- Centralized repository for feature definitions and values
- Ensures consistency between training and serving
- Reduces duplicate computation
- Examples: Feast, Tecton, Hopsworks
- Manages feature transformations, versioning

**385. What is model monitoring?**
- Tracking model performance in production
- Metrics: Accuracy, latency, throughput, data drift, concept drift
- Alerts for degradation
- Enables quick response to issues
- Continuous evaluation

**386. What is data drift?**
- Distribution of input features changes over time
- P(X) in production ≠ P(X) in training
- Causes: Seasonality, new user segments, external events
- Detection: Statistical tests (KS test, KL divergence), monitoring distributions
- May require retraining

**387. What is concept drift?**
- Relationship between features and target changes
- P(Y|X) in production ≠ P(Y|X) in training
- More serious than data drift (model becomes wrong)
- Example: Fraud patterns evolve
- Requires model update

**388. What is A/B testing for ML models?**
- Compare two model versions on random user groups
- Measures real-world impact
- Statistical test for significant difference
- Considers: Sample size, duration, metrics
- Gold standard for production validation

**389. What is shadow mode deployment?**
- New model receives data but predictions aren't used
- Allows monitoring without user impact
- Comparison with production model
- Low-risk deployment strategy
- Identifies issues before full rollout

**390. What is canary deployment?**
- Gradual rollout to increasing user percentage
- Monitor for issues before full deployment
- Quick rollback if problems detected
- Lower risk than immediate full deployment
- Common: 5% → 25% → 100%

**391. What is model serving?**
- Making model available for prediction requests
- Considerations: Latency, throughput, scalability, cost
- Options: REST API, batch processing, edge deployment
- Tools: TensorFlow Serving, TorchServe, Triton
- SLA requirements vary by application

**392. What is batch vs real-time inference?**
- Batch: Process many inputs together, offline (higher throughput, higher latency)
- Real-time: Process single request, online (lower latency, interactive)
- Batch: Recommendation preprocessing, analytics
- Real-time: Search, fraud detection, assistants
- Hybrid: Pre-compute where possible

**393. What is model quantization for deployment?**
- Reduces precision (FP32 → INT8) for faster inference
- Post-training or quantization-aware training
- 4-8× speedup, 4× memory reduction
- Minimal accuracy loss with proper calibration
- Critical for edge deployment

**394. What is model distillation for deployment?**
- Compress large model to smaller student
- Maintains much of original performance
- Faster inference, lower memory
- Can distill ensemble to single model
- Trade accuracy for efficiency

**395. What is model caching?**
- Store predictions for common inputs
- Reduces latency and compute for repeated queries
- Invalidation strategy when model updates
- Hit rate depends on query distribution
- Suitable for deterministic models

**396. What is feature engineering in production?**
- Compute features at serving time
- Challenges: Latency, consistency with training
- Solutions: Feature store, pre-computation, low-latency pipelines
- Avoid training-serving skew
- Balance freshness and latency

**397. What is training-serving skew?**
- Discrepancy between training and serving environments
- Causes: Different feature computation, different data processing, bugs
- Leads to performance degradation
- Prevention: Shared feature pipelines, testing, monitoring
- Major cause of production issues

**398. What is model interpretability in production?**
- Understanding why model makes predictions
- Important for: Trust, debugging, regulation, fairness
- Methods: Feature importance, SHAP, LIME, attention visualization
- Trade-off with performance (complex models less interpretable)
- Varies by application criticality

**399. What is model fairness?**
- Ensuring equitable treatment across groups
- Metrics: Demographic parity, equal opportunity, equalized odds
- Sources of bias: Training data, feature choice, objective
- Mitigation: Re-sampling, fairness constraints, post-processing
- Legal and ethical imperative

**400. What is model compression?**
- Reducing model size and computation
- Techniques: Quantization, pruning, distillation, low-rank factorization
- Enables deployment on resource-constrained devices
- Trade accuracy for efficiency
- Different methods combinable

---

## Advanced Topics

**401. What is meta-learning?**
- Learning to learn: Acquire learning algorithm itself
- Few-shot learning: Learn from few examples using prior tasks
- Examples: MAML (Model-Agnostic Meta-Learning), Prototypical Networks
- Inner loop: Task-specific learning; Outer loop: Meta-learning
- Applications: Rapid adaptation, personalization

**402. What is few-shot learning?**
- Learn from very few labeled examples (1-shot, 5-shot)
- Leverages prior knowledge from related tasks
- Approaches: Metric learning, meta-learning, transfer learning
- K-shot N-way: K examples for each of N classes
- More practical than requiring large labeled datasets

**403. What is zero-shot learning?**
- Classify classes never seen during training
- Uses side information (attributes, word embeddings)
- Example: Recognize "zebra" using "horse" + "stripes"
- Semantic embedding space for transfer
- Extreme form of transfer learning

**404. What is neural architecture search (NAS)?**
- Automatically design neural network architectures
- Search space: Possible architectures
- Search strategy: RL, evolution, gradient-based
- Expensive but can find better architectures than human design
- Examples: EfficientNet, AmoebaNet

**405. What is AutoML?**
- Automating ML pipeline: Preprocessing, feature engineering, model selection, hyperparameter tuning
- Democratizes ML (less expertise needed)
- Tools: Auto-sklearn, H2O AutoML, Google AutoML
- Trade-offs: Compute cost vs human time
- Still requires domain knowledge for problem formulation

**406. What is hyperparameter optimization?**
- Finding best hyperparameter values
- Methods: Grid search, random search, Bayesian optimization, Hyperband
- Expensive: Many training runs
- Random often better than grid (high-dimensional)
- Bayesian optimization more sample-efficient

**407. What is Bayesian optimization?**
- Black-box optimization using probabilistic model
- Gaussian process models objective function
- Acquisition function balances exploration-exploitation
- Sample-efficient (fewer evaluations)
- Used for expensive objectives (hyperparameter tuning)

**408. What is multi-task learning?**
- Train single model on multiple related tasks simultaneously
- Shared representations benefit all tasks
- Improves generalization, especially for data-scarce tasks
- Challenges: Task weighting, negative transfer
- Examples: Joint training for NER + POS tagging

**409. What is transfer learning?**
- Use knowledge from source task for target task
- Common: Pre-train on large dataset, fine-tune on small dataset
- Effective when tasks are related
- More data/compute for source task benefits target
- Foundation of modern NLP and vision

**410. What is domain adaptation?**
- Transfer learning when domains differ (distribution shift)
- Source (training) and target (test) domains have different distributions
- Methods: Feature alignment, adversarial training, self-training
- Unsupervised: No target labels; supervised: Some target labels
- Critical for real-world deployment

**411. What is adversarial training?**
- Train on adversarially perturbed examples
- Improves robustness to small perturbations
- Min-max optimization: Minimize loss on worst-case perturbations
- Computationally expensive (inner maximization)
- Trade-off: Robustness vs clean accuracy

**412. What are adversarial examples?**
- Inputs with imperceptible perturbations causing misclassification
- Exposes model vulnerabilities
- Transferable across models
- Generated by: FGSM, PGD, C&W
- Security concern for deployed systems

**413. What is certified robustness?**
- Provable guarantees of model behavior under perturbations
- Certification: Prove no adversarial example exists in region
- Methods: Interval bound propagation, randomized smoothing
- Expensive but provides guarantees
- Alternative to empirical robustness

**414. What is federated learning?**
- Train model on decentralized data (without centralizing)
- Privacy-preserving: Data stays on devices
- Challenges: Non-IID data, communication cost, stragglers
- Aggregation: FedAvg (average model updates)
- Applications: Mobile keyboards, healthcare

**415. What is differential privacy?**
- Formal privacy guarantee: Output doesn't reveal individual data
- (ε,δ)-differential privacy: Bounded information leakage
- Mechanisms: Noise addition (Laplace, Gaussian)
- Trade-off: Privacy vs utility
- Critical for sensitive data

**416. What is graph neural network (GNN)?**
- Neural network on graph-structured data
- Message passing: Aggregate neighbor information
- Applications: Social networks, molecules, knowledge graphs
- Architectures: GCN, GAT, GraphSAGE
- Challenges: Over-smoothing, scalability

**417. What is attention mechanism in GNNs?**
- Weight neighbor contributions (not equal aggregation)
- Graph Attention Networks (GAT)
- Learn importance of neighbors
- More flexible than fixed aggregation
- Improves expressiveness

**418. What is continual/lifelong learning?**
- Learn from non-stationary data stream
- Avoid catastrophic forgetting
- Challenges: Stability-plasticity dilemma
- Methods: Regularization (EWC), rehearsal, architecture expansion
- Requires memory of previous tasks

**419. What is elastic weight consolidation (EWC)?**
- Regularizes important weights when learning new task
- Identifies important weights via Fisher information
- Penalty: λΣFᵢ(θᵢ - θᵢ*)²
- Slows forgetting while allowing plasticity
- One approach to continual learning

**420. What is active learning?**
- Iteratively query most informative samples for labeling
- Reduces labeling cost
- Query strategies: Uncertainty sampling, query-by-committee, expected model change
- Useful when labeling is expensive
- Human-in-the-loop ML

**421. What is semi-supervised learning?**
- Leverage unlabeled data along with labeled data
- Unlabeled data provides structure (manifold, clustering)
- Methods: Self-training, co-training, consistency regularization
- Effective when labeled data is scarce
- Between supervised and unsupervised

**422. What is self-training?**
- Use model's predictions on unlabeled data as pseudo-labels
- Iteratively retrain with increasing confidence threshold
- Risk: Error amplification if predictions are wrong
- Works when model is reasonably accurate
- Common semi-supervised approach

**423. What is consistency regularization?**
- Model should give similar predictions for perturbed versions of same input
- Loss: Difference between predictions on original and perturbed
- Examples: Π-model, Mean Teacher, UDA
- Encourages smooth predictions
- Effective semi-supervised technique

**424. What is knowledge distillation?**
- Transfer knowledge from teacher (large/ensemble) to student (small)
- Student trained on soft labels (teacher's output distribution)
- Soft labels contain inter-class relationships
- Temperature: Softens distribution (τ > 1 reveals more information)
- Compression and transfer learning

**425. What is Neural ODE?**
- Models hidden state as continuous-time dynamical system
- dh/dt = f(h,t,θ)
- Advantages: Adaptive computation, memory-efficient backprop (adjoint method)
- Applications: Time series, physics-informed ML
- Continuous depth networks

**426. What is Attention mechanism's computational complexity?**
- Standard: O(n²d) time, O(n²) space
- n: Sequence length, d: Dimension
- Bottleneck for long sequences
- Solutions: Sparse attention, linear attention, chunking
- Memory is often the limiting factor

**427. What is Mixture of Experts (MoE)?**
- Multiple expert networks, router selects which to use
- Only subset of parameters active per input
- Scales capacity without proportional compute
- Challenges: Load balancing, training stability
- Examples: Switch Transformer, GShard

**428. What is the winner-takes-all principle in MoE?**
- Only top-k experts activated per input (typically k=1,2)
- Sparsely activated network
- Router learns which expert for which input
- Load balancing loss ensures experts used equally
- Enables massive scale

**429. What is the difference between hard and soft attention?**
- Hard: Select single location (discrete, non-differentiable)
- Soft: Weighted combination of all locations (continuous, differentiable)
- Hard: More efficient, requires REINFORCE for gradients
- Soft: Standard in transformers
- Hard: Interpretable, sparse

**430. What is cross-lingual transfer?**
- Transfer learning across languages
- Multilingual models (mBERT, XLM-R) enable zero-shot transfer
- High-resource language helps low-resource language
- Shared representation space across languages
- Democratizes NLP for low-resource languages

**431. What is code-switching in NLP?**
- Mixing multiple languages in single utterance
- Common in multilingual communities
- Challenges: Limited data, language identification, parsing
- Models: Multilingual BERT handles some code-switching
- Important for realistic multilingual NLP

**432. What is temperature in softmax?**
- T in softmax: σ(x/T)
- T→0: Sharper distribution (argmax); T→∞: Uniform
- Knowledge distillation: High T reveals inter-class similarities
- Sampling: Controls randomness in generation
- Trade-off: Exploitation vs exploration

**433. What is the Gumbel-Softmax trick?**
- Differentiable approximation to sampling from categorical
- Enables gradient-based optimization with discrete variables
- Temperature parameter controls approximation quality
- Used in VAEs with discrete latents, NAS
- Continuous relaxation of discrete sampling

**434. What is causal inference and how does it relate to ML?**
- Determining cause-effect relationships (not just correlation)
- ML predicts; CI explains and intervenes
- Tools: Randomized experiments, instrumental variables, do-calculus
- Causal ML: Predicting treatment effects, counterfactuals
- Critical for decision-making, not just prediction

**435. What is uplift modeling?**
- Predicting causal effect of treatment on individual
- Goes beyond prediction: "Will treatment change outcome?"
- Applications: Marketing, medicine
- Challenges: Observational data, selection bias
- Methods: S-learner, T-learner, X-learner

**436. What is instrumental variables in causal inference?**
- Variable that affects treatment but not outcome directly
- Enables causal inference from observational data
- Requirements: Relevance, exclusion, exchangeability
- Two-stage least squares: Common estimation method
- Addresses confounding

**437. What is propensity score matching?**
- Match treated and control units with similar treatment probability
- Balances covariates between groups
- Enables causal inference from observational data
- Propensity score: P(treatment | covariates)
- Reduces confounding, but assumes no unmeasured confounders

**438. What is counterfactual reasoning?**
- Reasoning about "what would have happened" under different action
- Fundamental to causal inference
- Individual treatment effects: ITE(i) = Y₁(i) - Y₀(i)
- Only observe one outcome (fundamental problem of causal inference)
- ML estimates counterfactuals from data

**439. What is catastrophic forgetting?**
- Neural networks forget previous tasks when learning new ones
- Overwrites weights useful for old tasks
- Worse for dissimilar tasks
- Solutions: Regularization (EWC), rehearsal, progressive networks
- Key challenge in continual learning

**440. What is curriculum learning?**
- Training order from easy to hard
- Mimics human learning
- Can speed convergence, find better minima
- Defining difficulty is challenge
- Examples: Sentence length, noise level

**441. What is hard negative mining?**
- Focus training on difficult negative examples
- Common in: Metric learning, object detection
- Improves discrimination of hard cases
- Risk: Outlier focus, overfitting to noise
- Balance with random negatives

**442. What is the bias-variance-noise decomposition?**
- Error = Bias² + Variance + Irreducible Error
- Bias: Systematic error from incorrect assumptions
- Variance: Error from sensitivity to training data
- Noise: Inherent randomness in data
- Trade-off between bias and variance

**443. What is double descent?**
- Test error decreases, increases, then decreases again with model size
- Modern regime: Overparameterized models generalize well
- Challenges classical bias-variance trade-off
- Related to implicit regularization
- Active research area in ML theory

**444. What is neural tangent kernel (NTK)?**
- Describes infinite-width neural network dynamics
- Connects NNs to kernel methods
- Infinite width: Gradient descent behaves like kernel regression
- Provides theoretical understanding
- Explains some generalization properties

**445. What is sample efficiency?**
- Amount of data needed to learn task
- Model-based RL > model-free (learns environment model)
- Meta-learning improves sample efficiency
- Critical for expensive data (robotics, clinical trials)
- Trade-off with computational efficiency

**446. What is sim-to-real transfer?**
- Training in simulation, deploying in real world
- Common in robotics (simulation is safe, cheap)
- Challenge: Reality gap (simulation ≠ real world)
- Methods: Domain randomization, realistic physics, adaptation
- Enables data collection for RL

**447. What is domain randomization?**
- Train on varied simulation parameters
- Forces learning robust representations
- Helps sim-to-real transfer (reality is within training distribution)
- Example: Randomize object textures, lighting, physics
- Simple but effective for robustness

**448. What is the cold start problem?**
- Making predictions for new users/items without history
- Common in recommendation systems
- Solutions: Content-based methods, demographic data, popularity baseline
- Trade-off: Exploration vs exploitation
- Active learning can help

**449. What is the long-tail problem?**
- Majority of data in few popular items/classes
- Rare items/classes have little data
- Challenges: Class imbalance, generalization to tail
- Solutions: Re-sampling, class weights, two-stage models
- Critical in recommendation, classification

**450. What is label noise and how to handle it?**
- Incorrect labels in training data
- Causes: Annotator error, ambiguity, outdated labels
- Effects: Overfitting to noise, worse generalization
- Solutions: Confident learning, robust loss functions, noise modeling, co-teaching
- Active research area

**451. What is confident learning?**
- Identifies label errors in dataset
- Uses predicted probabilities to find inconsistencies
- Can clean dataset by removing/relabeling errors
- Improves data quality systematically
- Open-source: cleanlab

**452. What is co-teaching?**
- Two networks teach each other
- Each selects small-loss samples for the other
- Resistant to label noise (disagreement indicates noise)
- Both networks converge to correct labeling
- Simple, effective for noisy labels

**453. What is the Matthew effect in ML?**
- Rich get richer: Popular items recommended more, become more popular
- Feedback loop in recommendation systems
- Reduces diversity, disadvantages new items
- Mitigation: Exploration, fairness constraints, debiasing
- Ethical and business concern

**454. What is representation learning?**
- Learning useful features automatically from data
- Manual feature engineering vs learned representations
- Deep learning excels at this
- Unsupervised (autoencoders) or supervised (task-driven)
- Good representations transfer across tasks

**455. What is disentangled representation?**
- Separate factors of variation in independent dimensions
- Each dimension corresponds to interpretable concept
- Enables controlled generation, better generalization
- Hard to define and measure formally
- VAEs, β-VAE attempt to learn disentangled representations

**456. What is multi-modal learning?**
- Learning from multiple data types (vision + language, audio + video)
- Richer understanding than single modality
- Challenges: Alignment, fusion, missing modalities
- Examples: CLIP (vision-language), audio-visual speech recognition
- Increasingly important with real-world data

**457. What is continual pre-training?**
- Further pre-training on domain data
- Adapts general model to specialized domain
- Cheaper than training from scratch
- Example: BioBERT (BERT + PubMed), FinBERT (BERT + financial news)
- Risk: Catastrophic forgetting of general knowledge

**458. What is the chinchilla paper's main finding?**
- Models are typically undertrained (too large, too little data)
- Optimal compute allocation: Equal scaling of model size and data
- Chinchilla (70B, 1.4T tokens) > Gopher (280B, 300B tokens)
- Previous scaling laws emphasized model size over data
- Shifted focus to data quality and quantity

**459. What is low-rank adaptation (LoRA) and why is it effective?**
- Freezes pre-trained weights, adds trainable low-rank matrices
- ΔW = BA where B: d×r, A: r×k, r << d,k
- Far fewer parameters than full fine-tuning
- No inference latency (merge ΔW into W)
- Hypothesis: Update is low-rank (task adaptation is lower-dimensional)

**460. What is scaling law for LLMs?**
- Loss scales predictably with compute, model size, dataset size
- Power law: L(C) ∝ C^α
- Enables forecasting performance before training
- Chinchilla findings: Data and model size should scale equally
- Guides resource allocation decisions

**461. What is prompt engineering?**
- Crafting input to elicit desired behavior from LLM
- No weight updates, just better inputs
- Techniques: Few-shot examples, chain-of-thought, instruction formatting
- Emergent skill with large models
- Complements fine-tuning

**462. What is in-context learning?**
- Model learns from examples in prompt without weight updates
- Emerged in GPT-3
- Improves with model scale
- Mechanism not fully understood (induction heads, mesa-optimization?)
- Distinguishes LLMs from traditional ML

**463. What is mesa-optimization?**
- Inner optimization learned within outer optimization
- Hypothesis: LLMs learn in-context via gradient descent in forward pass
- Speculative but explains in-context learning
- Raises alignment concerns (inner objectives may differ)
- Active research area

**464. What is Constitutional AI (CAI)?**
- AI feedback instead of human feedback (scalable)
- Model critiques own outputs based on principles
- Revises responses to align with principles
- More scalable than RLHF (less human labor)
- Developed by Anthropic

**465. What is red teaming for AI?**
- Adversarial testing to find failure modes
- Human or AI red teamers try to elicit harmful outputs
- Identifies vulnerabilities before deployment
- Iterative: Fix issues, red team again
- Critical for safe deployment

**466. What is the alignment problem?**
- Ensuring AI systems pursue intended goals
- Challenges: Specifying goals, robust to distribution shift, scalable oversight
- Outer: Define right objective; Inner: Optimize that objective
- Increasingly important as AI capabilities grow
- Active safety research area

**467. What is scalable oversight?**
- Supervising AI systems more capable than humans
- Challenge: How to evaluate outputs we couldn't produce?
- Approaches: Debate, recursive reward modeling, amplification
- Critical for advanced AI
- Fundamental alignment research

**468. What is reward hacking?**
- Agent finds unintended ways to maximize reward
- Misspecified reward function exploited
- Example: Boat racing agent spinning to collect rewards without finishing
- Prevention: Careful reward design, adversarial testing, robustness
- Major RL and alignment concern

**469. What is specification gaming?**
- Exploiting literal specification rather than intended behavior
- Similar to reward hacking but broader
- Goodhart's law: When measure becomes target, it ceases to be good measure
- Examples throughout ML: Overfitting, adversarial examples
- Requires careful problem formulation

**470. What is mode collapse in GANs?**
- Generator produces limited variety (few modes of data distribution)
- Failure to capture full data diversity
- Causes: Optimization dynamics, objective mismatch
- Solutions: Minibatch discrimination, unrolled GAN, Wasserstein GAN
- Major GAN training challenge

**471. What are Wasserstein GANs (WGAN)?**
- Uses Wasserstein distance instead of Jensen-Shannon divergence
- More stable training (no mode collapse, vanishing gradients)
- Meaningful loss curve (correlates with sample quality)
- Lipschitz constraint: Weight clipping or gradient penalty
- State-of-art GAN variant

**472. What is the spectral normalization in GANs?**
- Normalizes discriminator weights by spectral norm
- Enforces Lipschitz constraint (stabilizes training)
- Simpler than gradient penalty
- Improves training stability and sample quality
- Common in modern GAN architectures

**473. What is contrastive learning?**
- Self-supervised learning via instance discrimination
- Positive pairs: Augmentations of same image; Negative: Different images
- Objective: Bring positives together, push negatives apart (InfoNCE)
- Examples: SimCLR, MoCo
- State-of-art for self-supervised vision

**474. What is the InfoNCE loss?**
- Contrastive loss based on noise-contrastive estimation
- L = -log[exp(sim(x,x⁺)/τ) / Σexp(sim(x,xⁱ)/τ)]
- Temperature τ controls distribution sharpness
- Connects to mutual information maximization
- Foundation of modern contrastive learning

**475. What is momentum contrast (MoCo)?**
- Contrastive learning with momentum encoder and queue
- Queue: Large bank of negative samples (consistency)
- Momentum encoder: Slowly updated copy of encoder (stability)
- Decouples batch size from negative sample count
- Efficient, state-of-art self-supervised learning

**476. What is masked image modeling (MIM)?**
- Self-supervised pre-training for vision (like MLM for NLP)
- Mask image patches, predict missing content
- Examples: MAE (Masked Autoencoder), BEiT, SimMIM
- Captures visual structure without labels
- Emerging as strong alternative to contrastive learning

**477. What is the Masked Autoencoder (MAE)?**
- Mask large portion of image (75%), reconstruct from visible
- Asymmetric encoder-decoder (small decoder)
- Very efficient pre-training (skips masked patches in encoder)
- Strong results with simple approach
- Democratizes vision pre-training

**478. What is vision-language pre-training?**
- Learning joint representations of images and text
- Pre-training on large image-caption datasets (internet scale)
- Examples: CLIP, ALIGN, BLIP
- Enables zero-shot vision tasks (text as classifier)
- Foundation for multi-modal understanding

**479. What is zero-shot transfer in CLIP?**
- Classify images without training on target dataset
- Text prompts as classifier: "a photo of a {class}"
- Matches image to prompt in embedding space
- Leverages internet-scale pre-training
- Democratizes computer vision

**480. What is prompt tuning for vision-language models?**
- Learn continuous prompts for CLIP-style models
- Only prompts trained, model frozen
- Task-specific adaptation with few parameters
- Example: CoOp (Context Optimization)
- Extends prompt tuning from NLP to vision

---

## Attention Mechanisms Deep Dive

### Fundamentals of Attention

**481. What is the core intuition behind attention mechanisms?**
- Selectively focus on relevant parts of input
- Weighted combination based on relevance/importance
- Query-Key-Value paradigm: Query asks "what am I looking for?", Keys answer "what do I contain?", Values provide "what I actually output"
- Replaces fixed-weight averaging with learned, context-dependent weighting
- Enables handling variable-length sequences and long-range dependencies

**482. Explain the mathematical formulation of basic attention**
- Attention(Q, K, V) = Σᵢ αᵢVᵢ where αᵢ = softmax(score(Q, Kᵢ))
- Score functions: Dot product (Q·K), Scaled dot product (Q·K/√d), Additive (tanh(W[Q;K]))
- Softmax ensures weights sum to 1
- Output is weighted sum of values
- Context vector captures relevant information

**483. What are the different attention score functions?**
- **Dot product**: score(Q,K) = Q^T K (simple, efficient, assumes normalized)
- **Scaled dot product**: score(Q,K) = Q^T K / √d_k (prevents saturation for large d)
- **Additive (Bahdanau)**: score(Q,K) = v^T tanh(W₁Q + W₂K) (more parameters, flexible)
- **Multiplicative (Luong)**: score(Q,K) = Q^T W K (learned transformation)
- **Bilinear**: score(Q,K) = Q^T W K (general form)

### Multi-Head Attention (MHA)

**484. What is Multi-Head Attention and why is it important?**
- Runs multiple attention mechanisms in parallel (typically 8-16 heads)
- Each head learns different attention patterns
- Formula: MHA(Q,K,V) = Concat(head₁,...,headₕ)W^O
- Where headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)
- Allows model to attend to information from different representation subspaces

**485. How do multiple heads provide different perspectives?**
- Each head has separate learned projection matrices (W^Q, W^K, W^V)
- Different heads can capture: syntactic relations, semantic relations, positional patterns
- Example: One head for subject-verb agreement, another for coreference
- Empirical observation: Heads specialize in different linguistic phenomena
- Concatenation combines all perspectives

**486. What is the computational complexity of Multi-Head Attention?**
- Per head: O(n²d/h) where n=sequence length, d=model dimension, h=num heads
- Total: O(n²d) across all heads (same as single-head with full dimension)
- Each head operates on d/h dimensions
- Parallelizable across heads
- Memory: O(h·n²) for attention matrices (can be significant for long sequences)

**487. How are projection dimensions chosen in MHA?**
- Typically d_k = d_v = d_model/h (equal split across heads)
- Example: d_model=512, h=8 → d_k=d_v=64 per head
- Output dimension after concat: h·d_v = d_model
- Final projection W^O: d_model → d_model
- Maintains dimension throughout

**488. What are the learned parameters in Multi-Head Attention?**
- Per head: W^Q_i (d×d_k), W^K_i (d×d_k), W^V_i (d×d_v)
- Output projection: W^O (h·d_v × d)
- Total: h·(3d·d_k + d·d_v) parameters
- For BERT-base (d=768, h=12, d_k=64): ~2.4M parameters per MHA layer
- No bias terms in standard formulation (though some variants add them)

**489. Can you visualize what different attention heads learn?**
- Visualization: Plot attention weights as heatmaps
- Common patterns: Local attention (neighboring tokens), global attention (position-independent), syntactic patterns (attending to syntactic parents)
- Some heads appear interpretable (e.g., delimiter heads, positional heads)
- Many heads are less interpretable (distributed representations)
- Tools: BertViz, exBERT for visualization

**490. What is the difference between self-attention and cross-attention in MHA?**
- **Self-attention**: Q, K, V all from same sequence (intra-sequence dependencies)
- **Cross-attention**: Q from one sequence, K,V from another (inter-sequence dependencies)
- Self-attention: Encoder and decoder self-attention layers
- Cross-attention: Decoder attending to encoder outputs
- Both use same MHA mechanism, different input sources

### Multi-Query Attention (MQA)

**491. What is Multi-Query Attention?**
- Variant of MHA where K and V are shared across all heads
- Only Q has multiple heads
- Formula: MQA = Concat(head₁,...,headₕ)W^O where headᵢ = Attention(QW^Q_i, K, V)
- Dramatically reduces parameters and memory
- Developed for faster inference in autoregressive models

**492. Why does Multi-Query Attention improve inference speed?**
- Autoregressive decoding: Caches K,V for all previous tokens
- MHA: Cache size = h×n×d_v (grows with heads)
- MQA: Cache size = n×d_v (independent of heads)
- Reduces memory bandwidth requirements (major bottleneck)
- Can achieve 2-3× speedup in decoder inference

**493. What is the trade-off between MHA and MQA?**
- MQA: Fewer parameters (no K,V projections per head)
- MQA: Less expressive (shared K,V across heads)
- MQA: Faster inference, lower memory
- MQA: Slight quality degradation (0.5-1% typically)
- Can be mitigated with training techniques (uptraining)

**494. When should you use Multi-Query Attention?**
- Decoder models with autoregressive generation (GPT-style)
- Memory-constrained environments (mobile, edge devices)
- Latency-critical applications (real-time systems)
- Large batch inference
- Not as beneficial for encoder-only models (BERT) or non-autoregressive tasks

**495. What is uptraining for Multi-Query Attention?**
- Converting pre-trained MHA model to MQA
- Method: Continue training with MQA on small fraction of data
- Projects separate K,V heads to single shared K,V (average or learned projection)
- Recovers most performance quickly (5-10% of original training)
- Enables retrofitting existing models

### Grouped-Query Attention (GQA)

**496. What is Grouped-Query Attention?**
- Interpolation between MHA and MQA
- Divides heads into groups, shares K,V within each group
- G groups: Each group has h/G query heads sharing K,V
- Example: 32 heads, 4 groups → 8 Q heads per group, 4 total K,V pairs
- Balances quality and efficiency

**497. How does GQA compare to MHA and MQA?**
- **MHA**: G=h (each head has own K,V) - highest quality, slowest
- **GQA**: 1<G<h (groups share K,V) - balanced
- **MQA**: G=1 (all heads share K,V) - fastest, lowest quality
- GQA with G=h/2: ~1.5-2× speedup vs MHA, minimal quality loss
- Provides tunable quality-speed trade-off

**498. What are the benefits of Grouped-Query Attention?**
- Maintains most of MHA's expressiveness (separate K,V per group)
- Significantly reduces KV cache (factor of h/G)
- Modest speedup at inference (not as dramatic as MQA)
- Better quality than MQA (more K,V diversity)
- Recommended for production LLMs (Llama 2, Mistral use GQA)

**499. How do you choose the number of groups in GQA?**
- Trade-off: More groups = better quality but more memory/compute
- Common choices: 2, 4, 8 groups for 32-64 head models
- Rule of thumb: G=h/4 to h/8 provides good balance
- Depends on: Model size, hardware constraints, quality requirements
- Empirical tuning recommended for specific use case

**500. What is the implementation difference between MHA, MQA, and GQA?**
```python
# MHA: Separate K,V per head
K = [K_1, K_2, ..., K_h]  # h different K matrices
V = [V_1, V_2, ..., V_h]  # h different V matrices

# MQA: Single shared K,V
K = K_shared  # 1 K matrix for all heads
V = V_shared  # 1 V matrix for all heads

# GQA: K,V shared within groups
K = [K_1, K_1, ..., K_1, K_2, K_2, ..., K_G]  # Repeated per group
V = [V_1, V_1, ..., V_1, V_2, V_2, ..., V_G]  # G total K,V pairs
```

### Flash Attention

**501. What is Flash Attention?**
- IO-aware attention algorithm that reduces memory access
- Computes exact attention (not an approximation)
- Fuses operations (softmax, dropout, masking) into single CUDA kernel
- Reduces HBM (high-bandwidth memory) reads/writes
- Key insight: Memory bandwidth, not compute, is bottleneck

**502. Why is standard attention slow?**
- Standard attention materializes full n×n attention matrix in HBM
- Multiple memory reads/writes: Q,K→S (scores), S→P (softmax), P,V→O (output)
- Memory bandwidth limited (~1.5 TB/s for A100)
- Compute underutilized (A100 has 312 TFLOPS but limited by memory)
- Quadratic memory usage O(n²) in sequence length

**503. How does Flash Attention work?**
- **Tiling**: Divide Q,K,V into blocks that fit in SRAM (on-chip fast memory)
- **Recomputation**: Recompute attention scores instead of storing
- **Fused kernels**: Combine softmax, masking, dropout in single pass
- **Online softmax**: Compute softmax incrementally without full matrix
- Reduces HBM accesses from O(n²) to O(n²/B) where B is block size

**504. What is the online softmax trick in Flash Attention?**
- Standard softmax requires full row (memory bottleneck)
- Online softmax: Update running max and sum incrementally
- For blocks: m_new = max(m_old, m_block), correction factor = exp(m_old - m_new)
- Maintains numerically stable softmax computation
- Enables block-wise processing without materializing full matrix

**505. What are the performance improvements of Flash Attention?**
- **Speed**: 2-4× faster than standard attention on GPUs
- **Memory**: Reduces memory usage from O(n²) to O(n)
- **Longer sequences**: Enables 64k+ tokens (limited by memory before)
- **Training**: Faster iteration, larger batches
- **Quality**: Exact attention (not approximate), identical outputs

**506. What is Flash Attention 2?**
- Improved version with additional optimizations
- **Better parallelism**: Parallelizes over sequence length (not just batch/heads)
- **Reduced non-matmul FLOPs**: Fewer ops on attention matrix
- **Work partitioning**: Better GPU utilization
- **Performance**: 2× faster than Flash Attention 1, up to 9× vs baseline
- Standard in modern LLM training (GPT-4, Llama, etc.)

**507. What are the limitations of Flash Attention?**
- Requires CUDA (GPU-specific, limited CPU support)
- Backward pass more complex (recompute forward during backward)
- Debugging harder (fused kernels less transparent)
- Requires modern GPUs (Ampere/Ada architecture for best performance)
- Not beneficial for very short sequences (<512 tokens)

**508. How does Flash Attention handle causal masking?**
- Integrated into algorithm (no explicit mask matrix)
- During block computation, skips future positions
- Only computes lower triangular part of attention matrix
- Saves both compute and memory
- More efficient than applying mask post-hoc

**509. Can Flash Attention be used with sparse attention patterns?**
- Yes, with modifications (Flash Attention with block-sparse patterns)
- Implementations: BlockSparse FlashAttention
- Applies tiling to sparse block patterns
- Further reduces computation for sparse patterns
- Used in models with fixed sparse patterns (Longformer-style)

**510. What is the impact of Flash Attention on training?**
- Enables longer context training (4k→32k+ tokens)
- Faster training: 2-3× wall-clock speedup
- Larger batch sizes (memory savings allow more samples)
- No accuracy degradation (exact attention)
- Now standard in state-of-art LLM training

### Paged Attention

**511. What is Paged Attention?**
- Memory management technique for KV cache during inference
- Stores KV cache in non-contiguous memory pages (like OS virtual memory)
- Developed for vLLM (high-throughput LLM serving system)
- Reduces memory waste from fragmentation and overprovisioning
- Enables dynamic allocation as sequence grows

**512. What problem does Paged Attention solve?**
- **Traditional**: Pre-allocate max_length×d memory per sequence (wasteful)
- **Problem**: Most sequences shorter than max, wasted memory
- **Fragmentation**: Variable lengths cause memory fragmentation
- **Batching**: Limited by worst-case memory usage
- **Paged Attention**: Allocates on-demand in fixed-size pages

**513. How does Paged Attention work?**
- KV cache divided into fixed-size blocks (pages), e.g., 16-64 tokens per page
- Page table maps logical KV positions to physical memory blocks
- Allocate pages as sequence grows (like OS demand paging)
- Pages can be non-contiguous in physical memory
- Attention kernel modified to follow page table indirection

**514. What are the benefits of Paged Attention?**
- **Memory efficiency**: Near-zero waste from fragmentation
- **Higher throughput**: 2-4× more sequences in same memory
- **Larger batches**: Better GPU utilization
- **Dynamic batching**: Mix sequences of different lengths efficiently
- **Memory sharing**: Share KV cache across requests (prefix caching)

**515. How does Paged Attention enable KV cache sharing?**
- Common prefixes (system prompts) stored once
- Multiple sequences point to same physical pages for shared prefix
- Copy-on-write: Share until sequence diverges, then allocate new page
- Example: 100 requests with same system prompt → store once
- Massive memory savings for batch inference

**516. What is the overhead of Paged Attention?**
- Page table lookups add minimal overhead (<5%)
- More complex kernel implementation
- Memory fragmentation within pages (internal fragmentation)
- Trade-off: Small overhead for large memory savings
- Net benefit: 2-4× throughput increase

**517. How does page size affect Paged Attention performance?**
- **Smaller pages** (16 tokens): Less internal fragmentation, more overhead
- **Larger pages** (64 tokens): Less overhead, more waste per page
- Typical: 16-32 tokens per page
- Depends on: Sequence length distribution, hardware (cache sizes)
- Empirically tuned per workload

**518. How does Paged Attention compare to continuous batching?**
- **Continuous batching**: Iterative batching, finish sequences removed and replaced
- **Paged Attention**: Enables efficient continuous batching (reduces memory waste)
- Complementary techniques: Paged Attention improves memory, continuous batching improves throughput
- Together: State-of-art inference serving (vLLM, TensorRT-LLM)

**519. What is prefix caching in Paged Attention?**
- Caching KV states for common prompt prefixes
- Example: Same system instruction across requests
- Paged attention enables sharing: Multiple logical sequences → same physical pages
- Hit rate depends on prompt similarity
- Crucial for: RAG systems, agents with fixed prompts, chatbots

**520. What are the implementation challenges of Paged Attention?**
- Custom CUDA kernels (modify attention to follow page table)
- Memory management system (allocate, free, defragment pages)
- Scheduling: Decide which requests to batch
- Fault handling: What if sequence exceeds expected length?
- Currently implemented in: vLLM, TensorRT-LLM, Ray Serve

### Advanced Attention Variants

**521. What is Linear Attention?**
- Reduces attention complexity from O(n²) to O(n)
- Key insight: Avoid explicit softmax computation
- Kernel trick: Attention(Q,K,V) = φ(Q)(φ(K)^T V) / φ(Q)φ(K)^T
- Reformulation allows associative computation: (Q·K^T)V = Q·(K^T V)
- Trade-off: Approximation of softmax attention (quality loss)

**522. What is the Performer (FAVOR+ algorithm)?**
- Uses random Fourier features to approximate softmax kernel
- φ(x) = exp(x²/2) approximated by random projections
- Unbiased estimation of softmax attention
- Linear complexity: O(nd²) vs O(n²d)
- Quality closer to standard attention than other linear methods

**523. What is Sparse Attention?**
- Only computes attention for subset of pairs (not all n² pairs)
- Patterns: Local (sliding window), strided, global tokens, random
- Reduces complexity to O(n√n) or O(n log n)
- Examples: Longformer, BigBird, Sparse Transformer
- Trade-off: Reduced expressiveness for efficiency

**524. What is Sliding Window Attention?**
- Each token attends to w neighbors on each side (window size 2w+1)
- Complexity: O(n·w) - linear if w fixed
- Receptive field grows with depth: Layer L sees w·L tokens
- Used in: Longformer, Mistral
- Good for local dependencies (most language is local)

**525. What is Global Attention in sparse patterns?**
- Special tokens attend to all positions (and all attend to them)
- Example: [CLS] token in BERT, delimiter tokens
- Enables long-range information propagation
- Longformer: Few global tokens + sliding window for rest
- Balances efficiency and expressiveness

**526. What is Cross-Attention in encoder-decoder models?**
- Decoder queries, encoder keys/values
- Allows decoder to attend to relevant encoder positions
- Different from self-attention (different sequences)
- Multiple cross-attention layers in transformer decoder
- Critical for seq2seq tasks (translation, summarization)

**527. What is Relative Positional Attention?**
- Attention scores depend on relative position (not absolute)
- Modifies score: score(Q,K) + learnable bias(i-j)
- Better extrapolation to longer sequences
- Used in: T5, DeBERTa, Transformer-XL
- More parameter-efficient than absolute positions

**528. What is ALiBi (Attention with Linear Biases)?**
- Adds linear bias to attention based on distance: -m×|i-j|
- Different slopes m for different heads (learned or geometric)
- No positional embeddings needed
- Excellent extrapolation: Train on 512, test on 2048+
- Very simple, highly effective

**529. What is Local-Global Attention?**
- Combines local sliding window + global sparse connections
- Local: Captures nearby dependencies efficiently
- Global: Long-range connections via sparse pattern
- BigBird: Local + global + random attention
- Balances expressiveness and efficiency

**530. What is Axial Attention for 2D data (images)?**
- Factorizes 2D attention into row and column attention
- Attend along each axis separately, then combine
- Reduces n² to 2n for n×n image (flattened to n² sequence)
- Used in: AxialTransformer for images
- Generalizes to higher dimensions

### Attention Optimization Techniques

**531. What is Attention Dropout?**
- Applies dropout to attention weights (after softmax)
- Regularization: Prevents over-reliance on specific positions
- Typical rate: 0.1 (lower than hidden layer dropout)
- Applied during training only
- Can improve generalization

**532. What is Query-Key Normalization?**
- L2 normalize Q and K before dot product
- Prevents attention saturation (more stable training)
- Makes score independent of vector magnitude
- Used in some vision transformers (CLIP)
- Alternative to temperature scaling

**533. What is Attention Temperature?**
- Scale attention logits before softmax: softmax(QK^T/τ)
- τ>1: Softer attention (more distributed)
- τ<1: Sharper attention (more peaked)
- Can be learned or fixed
- Affects attention entropy

**534. What is Attention Sink phenomenon?**
- Model assigns high attention to initial tokens (often [BOS])
- Acts as "sink" for irrelevant attention mass
- Common in long-context models
- Can be exploited for compression (preserve sink tokens)
- Understanding attention patterns improves efficiency

**535. What is Attention Head Pruning?**
- Remove redundant or low-importance attention heads
- Identify via: Attention entropy, downstream performance, gradient-based importance
- Can remove 20-50% of heads with minimal performance loss
- Reduces parameters and computation
- Improves inference efficiency

**536. What is Attention Weight Quantization?**
- Reduce precision of attention computations (FP16→INT8)
- Significant memory bandwidth savings
- Flash Attention supports FP16 natively
- INT8 attention: More challenging (softmax precision sensitive)
- Active research area

**537. How does Batch Size affect Attention computation?**
- Attention parallelizes over batch and heads
- Memory: O(b·h·n²) for batch size b, h heads, n length
- Large batches: Better GPU utilization but more memory
- Gradient accumulation: Simulate large batch with less memory
- Batch size matters more for training than inference

**538. What is the role of Attention in Model Interpretability?**
- Attention weights show "what model looks at"
- Visualization: Heatmaps, attention flow
- Caution: Attention ≠ explanation (ongoing debate)
- Not always aligned with gradient-based importance
- Useful but should be combined with other interpretability methods

**539. What is Monotonic Attention?**
- Constrains attention to be monotonic (left-to-right)
- Used in: Speech recognition, online translation
- Enforces alignment property
- Enables streaming/online inference
- More constrained than standard attention

**540. What is the future of Attention mechanisms?**
- **Efficiency**: Sub-quadratic attention (linear, O(n log n))
- **Long context**: Million-token contexts (Paged Attention, sparse patterns)
- **Hardware co-design**: Custom accelerators for attention
- **Learned sparsity**: Adaptively sparse attention patterns
- **Alternatives**: State-space models (Mamba), hybrid architectures
- Active research: Balancing expressiveness, efficiency, and scalability

### Attention vs Alternatives

**541. How do State Space Models (SSMs) compare to Attention?**
- SSMs (S4, Mamba): Linear complexity O(n)
- Recurrent formulation (vs parallel attention)
- Efficient long sequences, constant memory during inference
- Trade-off: May lose some long-range expressiveness
- Emerging alternative/complement to Transformers

**542. What is Mamba and how does it differ from Attention?**
- Selective state-space model with gating
- Linear time complexity for training and inference
- Selective copying: Learns what to remember
- No attention mechanism (recurrent processing)
- Competitive with Transformers on many tasks, more efficient

**543. Can you combine Attention with SSMs?**
- Hybrid architectures: Some layers attention, some SSM
- Example: Alternating attention and SSM layers
- Attention for global context, SSM for efficient local processing
- Potential best-of-both-worlds
- Active research direction (2024-2025)

**544. What is the Attention-Free Transformer (AFT)?**
- Replaces softmax attention with learned position-based weights
- Reduces quadratic complexity
- Less expressive than full attention
- Research direction: Can we avoid attention entirely?
- Limited adoption so far

**545. Why has Attention become dominant despite alternatives?**
- **Expressiveness**: Captures arbitrary dependencies
- **Parallelizability**: Trains efficiently on GPUs
- **Empirical success**: State-of-art across domains
- **Ecosystem**: Tools, optimizations, understanding
- **Scaling**: Performance improves predictably with scale
- Alternatives may complement, not replace, attention

---

## Closing

This comprehensive collection covers 480 questions across the full breadth of ML, DL, and LLMs. Topics include:

- **Fundamentals**: Probability, statistics, linear algebra, optimization theory
- **Classical ML**: Linear models, SVMs, trees, boosting, clustering
- **Deep Learning**: Neural networks, CNNs, RNNs, optimization, regularization
- **Computer Vision**: Object detection, segmentation, architectures, transformers
- **NLP**: Tokenization, embeddings, sequence models, language understanding
- **LLMs**: Transformers, pre-training, fine-tuning, prompting, RLHF, alignment
- **Advanced**: Meta-learning, GNNs, federated learning, causal inference, MLOps

Each question includes detailed answers covering theory, intuition, applications, and trade-offs. This resource serves as:
- Interview preparation guide
- Comprehensive ML/DL/LLM reference
- Self-study curriculum
- Quick refresher on key concepts

For deeper understanding, each topic should be supplemented with:
- Hands-on implementation
- Paper reading (foundational and recent)
- Real-world projects
- Continuous learning as field evolves

The ML field is rapidly evolving, especially in LLMs and foundation models. Stay updated with:
- ArXiv papers (cs.LG, cs.CL, cs.CV)
- Conference proceedings (NeurIPS, ICML, ICLR, ACL, CVPR)
- Industry blogs (OpenAI, Anthropic, DeepMind, etc.)
- Open-source implementations

Good luck with your interviews and ML journey!
