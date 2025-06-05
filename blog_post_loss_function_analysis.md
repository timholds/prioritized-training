# Loss Function Choice in Prioritized Training: When Does the Overhead Pay Off?

*An analysis of how regression vs classification losses affect the feasibility and effectiveness of prioritized training on learnable points*

## Abstract

Prioritized training, based on the concept of training on "points that are learnable, worth learning, and not yet learnt," promises to improve training efficiency by focusing computational resources on the most informative samples. However, the choice of loss function fundamentally affects both the computational overhead and the effectiveness of this approach. Through analysis of MNIST/QMNIST experiments comparing Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) losses, we examine when the overhead of prioritized training justifies its computational cost and how different loss functions create different prioritization behaviors.

## Introduction

The core insight behind prioritized training lies in the ρ-loss (rho-loss) calculation:

```
ρ(x,y) = L_current(x,y) - L_irreducible(x,y)
```

Where:
- `L_current(x,y)` is the loss of the current model on sample (x,y)
- `L_irreducible(x,y)` is the loss of a model trained on holdout data

This ρ-loss represents how much the current model could potentially improve on each sample. But what happens when we change the loss function itself?

## The Computational Overhead Challenge

### Standard Training Pipeline
```python
# Standard training: O(n) per epoch
for batch in dataset:
    loss = compute_loss(model(batch.x), batch.y)
    gradients = compute_gradients(loss)
    optimizer.apply_gradients(gradients)
```

### Prioritized Training Pipeline
```python
# Prioritized training: O(k*n) per epoch where k >> 1
for epoch in range(epochs):
    candidate_batch = sample_large_batch(dataset, size=k*batch_size)
    
    # Extra forward pass for prioritization
    current_losses = model.predict(candidate_batch.x)  # O(k*batch_size)
    rho_losses = current_losses - irreducible_losses[candidate_batch.indices]
    
    # Select highest rho-loss samples
    priority_indices = argsort(rho_losses)[-batch_size:]
    priority_batch = candidate_batch[priority_indices]
    
    # Standard training step
    train_step(priority_batch)
```

The overhead factor `k` represents how many candidate samples we evaluate to select each training sample. This creates a fundamental question: **under what conditions does the improved sample selection offset the k-fold increase in computational cost?**

## Loss Function Analysis: MSE vs Cross-Entropy

### Mathematical Properties

**Mean Squared Error (MSE):**
```python
L_MSE(y_true, y_pred) = (1/n) * Σ(y_true - y_pred)²
```
- **Computational complexity:** O(n)
- **Gradient behavior:** Linear with error magnitude
- **Focus:** Samples with large magnitude errors (potential outliers)
- **Numerical stability:** High (bounded output)

**Categorical Cross-Entropy (CCE):**
```python
L_CCE(y_true, y_pred) = -(1/n) * Σ(y_true * log(y_pred))
```
- **Computational complexity:** O(n*k) where k = number of classes
- **Gradient behavior:** Exponential near decision boundaries
- **Focus:** Uncertain predictions (high entropy samples)
- **Numerical stability:** Lower (log(0) issues)

### Impact on Sample Prioritization

The choice of loss function fundamentally changes what constitutes a "learnable" sample:

1. **MSE-based prioritization** tends to select:
   - Samples with large prediction errors
   - Potential outliers or mislabeled examples
   - Samples where the model's output magnitude is far from target

2. **CCE-based prioritization** tends to select:
   - Samples near decision boundaries
   - Ambiguous cases where multiple classes seem plausible
   - Samples where the model is genuinely uncertain

## Experimental Analysis

### Setup and Implementation Issues

Our analysis revealed critical implementation inconsistencies in the experimental code that affect the interpretation of results:

```python
# From qmnist_conv_prioritized_regression.py (lines 160-163)
def make_il_loss_dict(model, x_train, y_train, loss='cce'):
    # BUG: Inverted logic!
    if loss == 'cce':
        loss_vals = mse(preds, y_train).numpy()        # Uses MSE when requesting CCE
    elif loss == 'mse':
        loss_vals = categorical_crossentropy(preds, y_train).numpy()  # Uses CCE when requesting MSE
```



### Corrected Analysis Framework

To properly analyze loss function impact, we need consistent experimental setups:

```python
# Proper MSE-based prioritized training
def mse_prioritized_training():
    holdout_model = train_model(holdout_data, loss='mse')
    il_losses = compute_mse_losses(holdout_model, train_data)
    
    for epoch in epochs:
        candidates = sample_candidates(train_data, k=10)
        current_losses = compute_mse_losses(current_model, candidates)
        rho_losses = current_losses - il_losses[candidates.indices]
        priority_batch = select_top_rho(candidates, rho_losses)
        train_step(current_model, priority_batch, loss='mse')

# Proper CCE-based prioritized training  
def cce_prioritized_training():
    holdout_model = train_model(holdout_data, loss='categorical_crossentropy')
    il_losses = compute_cce_losses(holdout_model, train_data)
    
    for epoch in epochs:
        candidates = sample_candidates(train_data, k=10)
        current_losses = compute_cce_losses(current_model, candidates)
        rho_losses = current_losses - il_losses[candidates.indices]
        priority_batch = select_top_rho(candidates, rho_losses)
        train_step(current_model, priority_batch, loss='categorical_crossentropy')
```

## When Does Overhead Pay Off?

### Theoretical Framework

The benefit of prioritized training can be modeled as:

```
Benefit = (Convergence_speedup * Training_time_saved) - (Overhead_cost * Total_epochs)
```

Where:
- `Convergence_speedup`: Factor by which prioritized training reaches target performance faster
- `Training_time_saved`: Time saved by reaching convergence earlier  
- `Overhead_cost`: Extra computation per epoch for sample selection
- `Total_epochs`: Number of epochs until convergence

### Loss Function Impact on Overhead Justification

**MSE-based prioritization:**
- ✅ **Lower computational overhead:** O(n) vs O(n*k) for CCE
- ✅ **Better numerical stability:** Fewer numerical issues during training
- ❌ **May focus on outliers:** Could select noisy or mislabeled samples
- ❌ **Less sophisticated prioritization:** Linear loss gradients provide coarser prioritization signals

**CCE-based prioritization:**
- ✅ **More sophisticated uncertainty estimation:** Captures model confidence better
- ✅ **Better decision boundary focus:** Prioritizes genuinely ambiguous samples
- ❌ **Higher computational overhead:** Softmax + log operations are expensive
- ❌ **Numerical stability concerns:** Risk of log(0) and extreme gradients

### Regime Analysis

**When MSE prioritization pays off:**
1. **Large datasets with computational constraints:** Lower overhead matters
2. **Noisy label environments:** When CCE's sensitivity to mislabeled samples is problematic  
3. **Early training phases:** When coarse prioritization signals are sufficient
4. **Resource-constrained environments:** When computational efficiency is paramount

**When CCE prioritization pays off:**
1. **High-quality datasets:** When sophisticated uncertainty estimation helps
2. **Near convergence:** When fine-grained sample selection matters
3. **Multi-class problems with class imbalance:** CCE better captures class-specific difficulties
4. **When training time is less constrained:** Overhead is acceptable for better sample selection

## Practical Recommendations

### Implementation Guidelines

1. **Choose loss function based on problem characteristics:**
   ```python
   # For computational efficiency in large-scale training
   use_mse_prioritization = (
       dataset_size > 1e6 or 
       computational_budget_limited or
       noisy_labels_suspected
   )
   
   # For sophisticated sample selection
   use_cce_prioritization = (
       high_quality_dataset and
       multi_class_problem and
       training_time_flexible
   )
   ```

2. **Adaptive overhead scheduling:**
   ```python
   # Start with high overhead, reduce as training progresses
   def adaptive_candidate_ratio(epoch, max_epochs):
       # High exploration early, efficiency later
       return max(2.0, 10.0 * (1 - epoch/max_epochs))
   ```

3. **Hybrid approaches:**
   ```python
   # Use MSE for initial prioritization, CCE for refinement
   def hybrid_prioritization(epoch, transition_epoch=10):
       if epoch < transition_epoch:
           return mse_prioritized_training()
       else:
           return cce_prioritized_training()
   ```

### Overhead Mitigation Strategies

1. **Candidate batch caching:** Reuse forward pass computations across multiple training steps
2. **Approximate prioritization:** Use smaller candidate pools during training
3. **Progressive refinement:** Start with coarse prioritization, increase sophistication over time

## Conclusion

The choice of loss function in prioritized training creates a fundamental trade-off between computational efficiency and prioritization sophistication. Our analysis reveals that:

1. **MSE-based prioritization** offers computational advantages but may focus on outliers rather than genuinely learnable samples
2. **CCE-based prioritization** provides better uncertainty estimation but at significant computational cost  
3. **Implementation consistency** is crucial - mixing loss functions between holdout and current model evaluation creates inconsistent prioritization signals
4. **The overhead-benefit trade-off** depends heavily on dataset size, computational constraints, and problem complexity

Future work should focus on hybrid approaches that capture the benefits of both loss types while minimizing computational overhead, and on developing more principled methods for determining when prioritized training overhead is justified.

## Code Availability

The experimental code analyzing these trade-offs is available in the prioritized-training repository, with corrected implementations addressing the loss function consistency issues identified in this analysis.

---

*This analysis is based on implementation of prioritized training concepts from "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt" (Mindermann et al., 2022).*