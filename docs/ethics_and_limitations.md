# Ethics and Limitations

## Purpose Statement

This system is a **research prototype and academic demonstration** of how AI can assist
in early screening for learning disabilities. It is **not a medical device** and must
not be used as the sole basis for any educational or clinical decision.

## Key Ethical Considerations

### 1. No Clinical Validation
The ML models are trained entirely on **synthetic data** generated from published
statistical ranges.  No real patient data has been used, and no clinical trial has been
conducted. Accuracy metrics reported on synthetic test sets do **not** predict
real-world clinical sensitivity or specificity.

### 2. Risk of Misclassification
- **False positives** may cause unnecessary anxiety for parents and teachers.
- **False negatives** may provide false reassurance, delaying needed intervention.
- The system should always be framed as a *first-pass screening*, not a diagnosis.

### 3. Demographic & Cultural Bias
- Handwriting norms vary across scripts, languages, and educational contexts.
- Speech features (reading speed, pause patterns) are language-dependent and may not
  generalise across languages or dialects.
- The system has not been tested for fairness across gender, ethnicity, or
  socio-economic groups.

### 4. Privacy
- Uploaded images and audio are processed in-memory and deleted after analysis.
- Speech transcription uses the Google Speech-to-Text API, which transmits audio to
  Google's servers. Users should be informed of this third-party processing.
- No personally identifiable information is stored persistently on the server.

### 5. Informed Consent
- The system includes a mandatory consent screen explaining its limitations.
- Users must acknowledge that results are not a diagnosis before proceeding.

### 6. Professional Oversight
- Results are accompanied by a recommendation to consult a qualified professional.
- The recommendation engine provides evidence-based interventions but cannot
  replace individualised professional assessment.

## Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Synthetic training data | Model may not generalise | Clearly labelled as prototype |
| Single-language STT | Only English supported | Document in UI |
| No longitudinal tracking | Cannot monitor progress over time | Future work |
| Image quality dependency | Poor lighting / camera → poor features | Quality checks + warnings |
| Age range restriction | Designed for 6–12; norms differ outside | Validate age input |

## Responsible Use Guidelines

1. **Always** present the disclaimer to end users.
2. **Never** use automated results to make placement, labelling, or diagnostic decisions
   without professional review.
3. **Inform** parents/guardians about what data is collected and how it is processed.
4. **Document** any deployment context and seek ethics board approval if used in a
   research study involving minors.
5. **Monitor** for adverse impacts and discontinue use if harm is observed.

## Future Work to Address Limitations

- Collect and validate on real clinical data (with IRB approval).
- Conduct fairness audits across demographic groups.
- Add multi-language support for speech analysis.
- Implement differential privacy for any stored data.
- Pursue CE marking or FDA consideration if clinical deployment is planned.
