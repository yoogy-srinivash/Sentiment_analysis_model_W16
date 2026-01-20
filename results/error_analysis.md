# Error Analysis and Bias Discussion

## Error Analysis

The model performs very well on clearly polarized movie reviews, producing high-confidence predictions for
strongly positive and strongly negative inputs. For example, reviews containing explicit sentiment words such
as “fantastic” or “worst” are classified correctly with probabilities above 0.99.

However, several systematic limitations were observed. Mixed-sentiment inputs such as *“Good acting but
terrible story”* are classified with high confidence toward a single label, even though the sentiment is
genuinely balanced. This suggests the model tends to overweight dominant sentiment cues instead of fully
balancing contrasting clauses.

Sarcastic inputs (e.g., *“Yeah great… I slept through it.”*) produce conflicting signals. While keyword-based
explanations detect positive words, the model’s probability distribution is split, resulting in an


**UNCERTAIN** prediction. This behavior highlights the effectiveness of the confidence threshold in preventing
overconfident misclassification for ambiguous or ironic language.

Another observed issue is overconfidence in mixed-keyword cases such as *“I loved the cinematography but hated
the pacing.”* Despite acknowledging mixed sentiment in the explanation, the model outputs extremely high
confidence for a negative label, indicating softmax probability miscalibration rather than a clear semantic
understanding.

Longer or more complex sentences generally produce lower confidence scores, which aligns with increased
uncertainty across multiple contextual signals.


## Bias and Limitations

The model was trained on movie reviews, introducing domain bias. As a result, it may perform poorly on
non-movie domains such as product reviews, social media posts, or news articles, where language structure and
sentiment expression differ.

The system is biased toward English-only, well-formed text and may misclassify informal language, slang,
Hinglish, or non-native grammar patterns.

Confidence scores are derived from softmax probabilities, which are known to be poorly calibrated in
transformer models. High confidence should not be interpreted as guaranteed correctness, particularly for
short, mixed, or sarcastic inputs.