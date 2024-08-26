Reviews to the `ExIFFI` paper received on the second submission to the `EAAI` (Engineering Application of Artificial Intelligence) journal. 

- Manuscript Number: EAAI-24-5135 
- Title: Interpretability and Generalizability in Extended Isolation Forest Models with Extended Isolation Forest Feature Importance and Enhanced Extended Isolation Forest  
# Reviewer 1

This paper focuses on enhancing interpretability in Anomaly Detection (AD) systems, particularly in engineering contexts where understanding the root cause of anomalies is essential. The paper introduces two key contributions:  
1. Extended Isolation Forest Feature Importance (ExIFFI): A novel approach designed to explain the predictions made by the Extended Isolation Forest (EIF) model. ExIFFI provides feature importance insights at both global and local levels, helping users understand why certain anomalies are flagged. This is crucial for fostering trust in the model and facilitating root cause analysis.  
2. Enhanced Extended Isolation Forest (EIF+): An improved version of EIF that aims to enhance generalization capabilities through a refined splitting hyperplane design strategy. This new variant is designed to perform better across different datasets.

## Comment 1

Some sentences are complex and could be simplified for better readability. For example, “Moreover, the end users of these systems are becoming more diverse, including people from various backgrounds who may not have a knowledge in data-driven methods” could be rephrased as “Moreover, the end users of these systems are increasingly diverse, often lacking expertise in data-driven methods.”  

> [!note] 
> Use simpler sentences following the example above.  
## Comment 2

The sentence structure could be streamlined to improve the flow. For example, “Despite the remarkable recent advancements in eXplainable Artificial Intelligence (XAI), most approaches are designed for supervised tasks, leaving unsupervised tasks, like Anomaly Detection (AD), rarely discussed in the literature” could be split into two sentences for clarity.  

> [!note] 
> Avoid long sentences with a lot of commas and prefer to split them into multiple sentences.  
## Comment 3

Before introducing `ECOD`, a sentence that links the discussion from `XAD` models to `ECOD` could help readers follow the narrative more easily.  

> [!note] 
> Create a connection between `XAD` (Explainable `AD`) and `ECOD` saying for example that there are some `AD` methods that are inherently interpretable like `ECOD` (i.e. they do not need to have an interpretation algorithm designed ad-hoc for them).  
## Comment 4

The citation “[16]” should match the format of others like “[17]” or “[18].”  

> [!note] 
> This is just a technical thing → check the formats of the citations to make them all the same.  
## Comment 5

Figure `1a` and `1b` should be well-labeled and referenced clearly in the text. Consider adding captions that briefly describe what each figure illustrates.  

> [!question] 
> This comment was done also by  [[ExIFFI PAPER REVIEW EAAI#Reviewer 4|Reviewer 4]]. Actually in this case there is a long caption for Figure 1 that links to Figure `1a` and `1b` inside it. So they want us to move the explanations inside the caption `Figure 1` to the captions behind figure `1a` and figure `1b`? 
## Comment 6

“Induced Imbalance Coefficients,” “Cumulative feature importance” briefly define these terms when first introduced or provide references for readers who may need additional context.  

> [!note] 
> Maybe here is easier to add the citation to the `DIFFI` paper when we introduce this terms (to avoid adding more sentences that may make the paper longer since some reviewers said that the paper is too long).  
## Comment 7

In Section 4 authors are suggested to write benefit from a brief mention of why interpretability is crucial in the context of anomaly detection models. This would set the stage for the discussion that follows.  

> [!note] 
> Probably we discussed this thing in another section so we can move it here or we can repeat it.  
## Comment 8

In Section 4 consider summarizing the main takeaway of DIFFI in simpler terms at the end of the section to reinforce understanding.  

## Comment 9

In Section 5 the organization of content could be improved. Separating the description of datasets from the performance and interpretability evaluations could enhance clarity.  

## Comment 10

Justify why Scenario II (0 contamination of outliers) is particularly relevant in certain AD model applications could provide more context for the reader.  

## Comment 11

Consider adding a brief explanation of why each dataset was chosen and how its characteristics align with the goals of the evaluation.  

## Comment 12

> [!warning] Focus on this 
>  In the **Time Scaling Experiments** (Section 6.3) add the Importance Time computations also for `KernelSHAP` (use the `KernelSHAP` code used for the `ExIFFI` Industrial Test paper). 

Providing specific examples or data points that illustrate the computational burden of SHAP would strengthen the argument.  

## Comment 13

The comparison with other models is thorough, the discussion could benefit from deeper insights into why certain models like DIF and AutoEncoder perform poorly in specific scenarios. Providing a more detailed explanation of the underlying reasons for these performance discrepancies could add value to the analysis.  

## Comment 14

The significance of feature 0 in different scenarios is noted, but discussing how this impacts the practical application of EIF+ and ExIFFI in real-world scenarios could provide more context.  

## Comment 15

The AI finds its applications in various domains. Authors are suggested to cite recent articles such as Green Fault Tolerant AIoT-enabled Mobile Sink Data Collection Scheme in Sensor Networks, Fault Diagnosis for AIT based Smart Farming Applications, AI Revolutionizing Healthcare: Innovations, Challenges, and Ethical Considerations, Unlocking the power of industrial artificial intelligence towards Industry 5.0:Insights, pathways, and challenges.
# Reviewer 2

The paper focuses on unsupervised anomaly detection, proposing ExIFFI for interpretability and EIF+ for improved generalization of the EIF model. It compares their performance using synthetic and real-world datasets, demonstrating the effectiveness of ExIFFI and the superiority of EIF+. A metric is introduced for quantitative evaluation, and future research directions are suggested, including exploring information in splitting nodes and deploying the methods in industrial contexts.The introduction of ExIFFI and EIF+ is a noteworthy contribution, enhancing interpretability and generalization in unsupervised anomaly detection. Comprehensive comparisons using diverse datasets offer robust evidence of the models’ performance and effectiveness. The AUCFS metric provides a quantitative means to assess the effectiveness of interpretation algorithms, adding objectivity to the evaluation.However, the article can be further strengthened, so I propose the following comments:  

1. When describing the ExIFFI algorithm, please use example data and intermediate calculation results to help readers better understand how the algorithm calculates feature importance. Additionally, consider providing pseudocode to make the algorithm steps clearer.  
2. When explaining the improvements of EIF+ compared to EIF, please quantify and compare the performance differences between the two on several representative datasets, and conduct an in-depth analysis of the fundamental reasons for this difference (i.e., the difference in the hyperplane selection strategy).  
3. For the generation process of the synthetic datasets, it is best to provide specific data generation functions or algorithm pseudocode, rather than just a simple description. If the dataset involves noise or other complex situations, they also need to be clearly explained.  
4. During the experimental evaluation, consider adding some of the latest anomaly detection baseline models for comparison, rather than just limiting it to classic models such as IF, to demonstrate the superiority of the proposed method.  

> [!note] 
> We have already done this → we have inserted `DIF` and `AutoEncoders`

5. In the interpretability evaluation, in addition to using feature selection as a proxy task, artificial datasets can also be designed, and based on the known ground-truth, a quantitative score can be given to the explanations provided by ExIFFI to more directly evaluate the quality of its explanations.  
6. For the correlation analysis between the explanations of ExIFFI and the outliers, it can be specifically examined whether the samples with high feature importance scores also correspond to the higher anomaly scores given by the algorithm, and whether there is a clear positive correlation between the two.  
7. In addition to analyzing the differences of the model under different data distributions and anomaly types, the differences in the explanation quality of ExIFFI can also be analyzed, such as in which cases the model’s explanation is more reliable and more in line with human cognition. 

> [!warning] Focus on this
> Amplify the description (in particular on the `glass` and `moodify` datasets (Sections 6.2.4 and Section 6.2.7)) saying that we chose these datasets because they have a sort of ground truth on what are the most important features and the fact that they align with the human cognition and then we can highlight this fact in explaining the results. 
> 

# Reviewer 3

This is a relevant article. However, it is still being prepared to be considered as a scientific paper. Below are some key observations:  
1. The title and content could be simpler for anyone to read and follow.  
2. The objectives and the focus of the study need to be clarified.  
3. The authors spend most of their time describing what they have done without highlighting why it is unique and how it will benefit others. Unless these two questions are answered upfront, I am not sure why someone would read this work.  
4. Too lengthy  
5. There are too many details (Figures and Tables) that need to highlight their purpose. It reads more like a part of a thesis or a report.  
6. The topic is about relevant and if work is presented well may have an impact.  
7. The authors should expand their review of relevant work and present it succinctly. Many researchers have widely used advanced data-driven techniques, such as the Bayesian Network-based methods; researchers from C-RISE, Memorial University, Canada, have proposed scores of fault diagnosis and isolation models and methods that are extended to complex engineering system safety and risk analysis. The authors have also ignored these studies.  
8. The authors should attempt to benchmark their study with other published work.

# Reviewer 4

- Include all important numerical results in the abstract.  
- Better highlight new contributions and novelty of your work in the introduction section.  
- Enhance linkage to recent applied literature that demonstrates the great potential and usefulness of different machine learning models for forecasting purposes (doi: 10.1108/AJEB-01-2024-0007; 10.1016/j.meaene.2024.100001; 10.1007/s00521-024-09531-2; 10.1108/JM2-12-2023-0315; 10.1177/03019233241254891) to better motivate your present investigation.  
- Provide a summary of previous studies in terms of what has been done, what is still missing, and correspondingly your contributions.  
- Improve the quality of the figures to make them clearly visualized by using high-resolution formats. 
- Improve the organization of the tables to make them clearly laid out. 
- Add more explanations to the figures and tables to make them self-explanatory.  
- Offer more detailed discussions of the results by linking to the figures and tables.  
- Use more mathematical formulae and equations to demonstrate your proposed approach.  
- Conduct further benchmark analysis to demonstrate the advantage or disadvantage of your method.  
- Elaborate on the potential limitations of your work, and correspondingly, the path to future studies in the conclusion.  
- More carefully proofread your manuscript to avoid writing issues.