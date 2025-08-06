# Methodology Assessment Template

Use this template to systematically assess any new methodology or approach before and during development.

## 1. BASELINE ESTABLISHMENT

### Current Best Performance
- **Method**: [Current best approach]
- **Performance Metric**: [Accuracy/F1/etc.] = [X.XX%]
- **Test Data**: [Dataset/sample size]
- **Computational Cost**: [Training time/prediction time]

### Simple Baseline
- **Method**: [Simple comparison method - random, statistical, etc.]
- **Performance**: [Baseline metric] = [X.XX%]
- **Purpose**: Minimum acceptable performance threshold

## 2. NEW METHODOLOGY PROPOSAL

### Approach Description
- **Method Name**: [New methodology name]
- **Core Technique**: [Brief technical description]
- **Expected Advantage**: [Why this should work better]
- **Theoretical Justification**: [Mathematical/domain reasoning]

### Success Criteria
- **Minimum Viable Performance**: [Must exceed baseline by X%]
- **Target Performance**: [Ambitious but realistic goal]
- **Secondary Metrics**: [Other important measures]
- **Computational Constraints**: [Acceptable cost limits]

## 3. IMPLEMENTATION CHECKPOINTS

### Checkpoint 1: Basic Implementation (Week 1)
- [ ] Method implemented and runs without errors
- [ ] Produces reasonable outputs (no NaN, inf, etc.)
- [ ] Performance test against baseline completed
- **Result**: [Pass/Fail] - Performance: [X.XX%] vs Baseline: [Y.YY%]

### Checkpoint 2: Optimization Attempt 1 (Week 2)
- [ ] First round of improvements implemented
- [ ] Hyperparameter tuning completed
- [ ] Performance comparison with baseline
- **Result**: [Pass/Fail] - Performance: [X.XX%] vs Previous: [Y.YY%]

### Checkpoint 3: Optimization Attempt 2 (Week 3)
- [ ] Second round of improvements implemented
- [ ] Different approach/parameters tested
- [ ] Performance comparison with baseline
- **Result**: [Pass/Fail] - Performance: [X.XX%] vs Previous: [Y.YY%]

## 4. PERFORMANCE ASSESSMENT

### Quantitative Results
| Method | Accuracy | Precision | Recall | F1 | Training Time | Inference Time |
|--------|----------|-----------|--------|----|--------------| --------------|
| Baseline | X.XX% | X.XX% | X.XX% | X.XX | X.Xs | X.Xs |
| New Method | X.XX% | X.XX% | X.XX% | X.XX | X.Xs | X.Xs |
| **Difference** | **±X.XX%** | **±X.XX%** | **±X.XX%** | **±X.XX** | **±X.Xs** | **±X.Xs** |

### Performance Classification
- [ ] **SUCCESS**: >10% improvement over baseline
- [ ] **MINOR SUCCESS**: 5-10% improvement over baseline  
- [ ] **MARGINAL**: 0-5% improvement over baseline
- [ ] **WARNING**: 0-5% degradation from baseline
- [ ] **FAILURE**: 5-10% degradation from baseline
- [ ] **CRITICAL FAILURE**: >10% degradation from baseline

## 5. FAILURE ANALYSIS (If Applicable)

### Root Cause Analysis
- **Primary Issue**: [Main reason for failure]
- **Contributing Factors**: [Secondary issues]
- **Unexpected Results**: [Surprising findings]
- **Data Issues**: [Problems with dataset/features]

### Lessons Learned
- **What We Learned**: [Technical insights]
- **Assumptions Proven Wrong**: [Incorrect hypotheses]
- **Transferable Knowledge**: [Useful for other approaches]

## 6. GO/NO-GO DECISION

### Decision Criteria Met
- [ ] Method shows clear improvement (>5%)
- [ ] Computational cost is acceptable
- [ ] Implementation is robust and reliable
- [ ] Theoretical foundation remains sound

### Decision: [GO / NO-GO / PIVOT]

**Rationale**: [Detailed explanation of decision with supporting data]

### If GO: Next Steps
1. [Specific action 1]
2. [Specific action 2]
3. [Timeline and milestones]

### If NO-GO: Alternative Actions
1. **Pivot Option 1**: [Alternative methodology]
2. **Pivot Option 2**: [Different approach]
3. **Resource Reallocation**: [Where to focus effort instead]

## 7. PROJECT IMPACT ASSESSMENT

### Timeline Impact
- **Delay Caused**: [Days/weeks lost]
- **Lessons Value**: [How this helps future work]
- **Overall Project Risk**: [Low/Medium/High]

### Resource Impact
- **Development Time**: [Hours invested]
- **Computational Resources**: [Cost of experiments]
- **Opportunity Cost**: [What else could have been done]

### Strategic Impact
- **Technical Understanding**: [How this advances knowledge]
- **Market Position**: [Effect on competitive advantage]
- **Future Options**: [Doors opened or closed]

---

## APPROVAL SIGNATURES

**Technical Assessment**: [Pass/Fail] - [Name/Date]
**Performance Validation**: [Pass/Fail] - [Name/Date]  
**Strategic Alignment**: [Pass/Fail] - [Name/Date]

**FINAL DECISION**: [GO/NO-GO/PIVOT] - [Date]

---

*Use this template for every new methodology to ensure systematic, honest assessment and avoid wasted effort on failing approaches.*