Findings:  
The model "learned" almost everything it was going to learn within the first two epochs.  

Early Saturation: Between Epoch 2 and Epoch 3, the Accuracy (0.9144) and Recall@1 (0.0954) remained perfectly flat.  

The "Dip" in Quality: Interestingly, almost all top-tier retrieval metrics (NDCG, MRR, and MAP) actually decreased slightly in the final epoch.  

MRR: 0.1874 → 0.1845  
NDCG@10: 0.2498 → 0.2446  

Success vs. Baseline: Even with that slight end-of-training dip, the custom model is still vastly superior to the baseline in retrieval, effectively doubling the Mean Reciprocal Rank (MRR).  


Model Behavior:  
The model is exhibiting a behavior often seen in contrastive learning where it reaches a "local optimum" very quickly and then begins to struggle with fine-grained differentiation.  

Coherence over Separation: The earlier results showed Neighbor Coherence increased while Separation Ratio decreased. The model is getting very good at grouping things (Accuracy is high), but it is failing to rank the absolute best result at the very top.  

Training Efficiency: I spent roughly 12 minutes training, but the 3rd epoch contributed essentially zero value.  


Challenges

1. The Recall@1 Ceiling  
While Recall@10 is decent (44%), Recall@1 is under 10%. This means that in 9 out of 10 searches, the model knows the general "neighborhood" of the answer but cannot identify the specific correct document as the #1 choice.  

2. The Clustering Paradox  
The Silhouette Score drop suggests the model is pulling the entire dataset into a tighter ball rather than creating distinct, well-separated clusters. This makes retrieval easier but makes classification or categorical separation much harder.  


Summary of Final State  

Metric | Epoch 1 | Epoch 2 | Epoch 3 | Verdict  
--- | --- | --- | --- | ---  
Cosine Accuracy | 0.891 | 0.914 | 0.914 | Peaked at E2  
Recall@10 | 0.430 | 0.453 | 0.440 | Regressed at E3  
MRR@10 | 0.167 | 0.187 | 0.184 | Regressed at E3  


Verdict  
My model is significantly improved over the baseline for general search (retrieval), but it is currently over-optimized for the training set or hitting a bottleneck in architectural capacity. The 3rd epoch was unnecessary, and the model is currently "smearing" its embeddings together, which explains why clustering metrics (Silhouette/Separation) are trending downward while retrieval is up.
