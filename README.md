# tensorflow-DTN
Most prior domain adaptation methods only focus on reducing the difference in
the marginal distribution between domains. An emerging question, however, is
that the MMD value is quite small, but the two distributions are indistinguishable
due to the small variance and inter-class distance. We instead use optimized
MMD to ensure that the full feature distributions discrepancy is reduced, and
discriminative property for classification tasks is reserved. This helps avoid the
common failure of domain adaptation models, where the whole domain data
collapses to outputting the indistinguishable situation of different categories.

# experiment result

