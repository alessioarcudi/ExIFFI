Let's draw here a possible structure of the new version of the paper with a detailed list. The idea is also to include the kind of plots we want to insert in each different section so that we have a clear idea of the experiments to carry on and have a clear idea on where exactly to put each different thing. 

We will start from the structure of the first version of the paper and re adapt it depending on the new things to add according to the suggestions added in the review. 

# *ExIFFI and EIF+: Interpretability and Enhanced Generalizability to Extend the Extended Isolation Forest*

0.  **Abstract**
1. **Introduction**: Introduce the problem of Anomaly Detection and the need for interpretability and at the end report the structure of the paper saying what will appear in each different section 

> [!missing] 
> In the first version of the paper Section 2 is used to present the Isolation Based Approaches for Anomaly Detection: IF, EIF and DIFFI but we did not insert a Related Work section presenting other anomaly detection and interpretability methods for unsupervised learning. See [[ExIFFI PAPER REVIEW#Papers for Related Work|here]] for some papers that may cite in the Related Work section.  

2. **Related Work** → Insert here citations to AD methods (e.g. DIF, PID, INNE, AutoEncoder) and interpretability methods used in Anomaly Detection. These methods are mainly used on DL based models (e.g. AutoEncoders) so here we may justify the importance of the introduction of an interpretability method on a very efficient for Anomaly Detection as IF/EIF/EIF+ are. Successively we can talk about IF/EIF and DIFFI but maybe removing all the detailed formulas used to describe them (to save some space since we have added the citations to other papers in the literature). 
		2.1 **Introduction of EIF+**: Introduce EIF+ with a more detailed description on how it is better than the other approaches 
3. **Interpretability for Isolation-based Anomaly Detection Approaches**: Description of DIFFI. Here we can also cite the paper by Mattia Carletti on the application of DIFFI on an industrial setting (***Interpretable Anomaly Detection for Knowledge Discovery in Semiconductor Manufacturing***) to justify the goodness of this model and why it is important ExIFFI (that is essentially a generalization of DIFFI). 
		3.2. **ExIFFI** → Explanation of ExIFFI with Global and Local Feature Importance 
		 3.2.3 **Visualizing Explanations** → Explanation of Bar Plot, Score Plot and Importance Map 
4. **Experimental Results** 