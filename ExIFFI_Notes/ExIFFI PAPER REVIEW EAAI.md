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

> [!info] In progress
>  Connected to [[ExIFFI PAPER REVIEW EAAI#^7eb8d8|this other comment]]. Together with the amplification of the description of `glass` and `moodify` we can also add an introductory description to the `wine` dataset → in this case we can say that we decided to use it because of its high dimensionality and we wanted to see how the model behaves when we have an high dimensionality. 

> [!todo] 
> See Overleaf 
## Comment 12

Providing specific examples or data points that illustrate the computational burden of SHAP would strengthen the argument.  

> [!warning] Focus on this 
>  In the **Time Scaling Experiments** (Section 6.3) add the Importance Time computations also for `KernelSHAP` (use the `KernelSHAP` code used for the `ExIFFI` Industrial Test paper). 

> [!info] In progress
>  I am running the Time Scaling experiments with `KernelSHAP`. It is taking a lot of time. At this point I do not even know weather I would be able to finish all the experiment or it is worth to just produce the plot with less data points for `SHAP` that will in any case show how its computational cost is much higher than the one of all the other models considered. 
## Comment 13

The comparison with other models is thorough, the discussion could benefit from deeper insights into why certain models like DIF and AutoEncoder perform poorly in specific scenarios. Providing a more detailed explanation of the underlying reasons for these performance discrepancies could add value to the analysis.  

> [!note] 
> Perchè sono meno robusti quando ci sono isolated anomalies che sono quelle per cui sono stati creati i vari modelli isolation-based (i.e. IF,EIF,EIF+) 

## Comment 14

The significance of feature 0 in different scenarios is noted, but discussing how this impacts the practical application of EIF+ and ExIFFI in real-world scenarios could provide more context.  

## Comment 15

The AI finds its applications in various domains. Authors are suggested to cite recent articles such as Green Fault Tolerant AIoT-enabled Mobile Sink Data Collection Scheme in Sensor Networks, Fault Diagnosis for AIT based Smart Farming Applications, AI Revolutionizing Healthcare: Innovations, Challenges, and Ethical Considerations, Unlocking the power of industrial artificial intelligence towards Industry 5.0:Insights, pathways, and challenges.

- *Green Fault Tolerant AIoT-enabled Mobile Sink Data Collection Scheme in Sensor Networks* → This paper focuses on the Wireless Sensor Networks (WSN) that are used in `IoT` infrastructures and proposed an AI based system to provide green and fault-tolerant data collection in these networks to avoid problems like energy holes (due to limited battery life) and network partitioning issues. It is an intelligent Mobile Sink-based data collection scheme. From what I read from the abstract is something related to telecommunications but it may be cited in the Introduction since it's an example on how AI finds applications in various domains. In any case if we then want in the future to deploy `ExIFFI` in such a system it may be relevant to use this approach. 
- *Fault Diagnosis for AIT based Smart Farming Applications* → Interestingly this paper is connected to the Smart Agricolture word. In particular here there is an `IoT` system used for Smart Farm Monitoring composed of multiple sensors used to measure physical quantities (e.g. humidity,soil moisture,temperature,etc.). All these sensors can be seen as nodes in a network and they can fail any time because of natural calamities (in fact these kind of systems are exposed to natural phenomena since they work outside). Moreover we have also to consider the fact that all these sensors have a battery with limited working time. To solve these issues this paper proposed a Fault Detection algorithm: AI-based hyperparameter-tuned least square support vector machine (HT-LS-SVM) → a very fancy way of saying that they tuned the hyperparameters of an `SVM` using `MSE` as the loss function. Then they also propose mobile sink that applies Reinforcement Learning (`RL`) to perform obstacle-aware path planning. 
- *Unlocking the power of industrial artificial intelligence towards Industry 5.0:Insights, pathways, and challenges* → This is probably a very broad and general paper (like a Survey or Report) talking about the application of AI solutions in the industrial world. 

> [!note] How to cite these articles
>  The articles the reviewers suggested us to cite have a very slight connection with our work: they are all coming from the AI sector. So I think that the only way we can cite them is in the Introduction (or in the Related Work section) talking about the fact that AI is being used in very various and different domains, or in general it has been applied in Industry (so we can connect to the Industry 5.0 topic) as `ExIFFI` may do. 

> [!done] 
> I inserted the three papers above in the very first paragraph of the paper (in the Introduction section) where we talk in very general terms about ML and AI and how they found applications in various domains. 
# Reviewer 2

The paper focuses on unsupervised anomaly detection, proposing ExIFFI for interpretability and EIF+ for improved generalization of the EIF model. It compares their performance using synthetic and real-world datasets, demonstrating the effectiveness of ExIFFI and the superiority of EIF+. A metric is introduced for quantitative evaluation, and future research directions are suggested, including exploring information in splitting nodes and deploying the methods in industrial contexts.The introduction of ExIFFI and EIF+ is a noteworthy contribution, enhancing interpretability and generalization in unsupervised anomaly detection. Comprehensive comparisons using diverse datasets offer robust evidence of the models’ performance and effectiveness. The AUCFS metric provides a quantitative means to assess the effectiveness of interpretation algorithms, adding objectivity to the evaluation.However, the article can be further strengthened, so I propose the following comments:  

1. When describing the ExIFFI algorithm, please use example data and intermediate calculation results to help readers better understand how the algorithm calculates feature importance. Additionally, consider providing pseudocode to make the algorithm steps clearer.  

> [!note] 
> See [[ExIFFI PAPER REVIEW EAAI#Pseudocode in Latex|the pseudocode tests section]].  

> [!info] In progress
> - Inserted the `ExIFFI` algorithm pseudocode.  
> - Inserted the Feature Selection Proxy Task algorithm pseudocode.
> - Inserted the Synthetic Dataset generation pseudocode. 
> Only thing left to do is solve the problem that puts all zeros as the line numbers. 

2. When explaining the improvements of EIF+ compared to EIF, please quantify and compare the performance differences between the two on several representative datasets, and conduct an in-depth analysis of the fundamental reasons for this difference (i.e., the difference in the hyperplane selection strategy).  

> [!question] 
>  Bring back the violin plots?  

3. For the generation process of the synthetic datasets, it is best to provide specific data generation functions or algorithm pseudocode, rather than just a simple description. If the dataset involves noise or other complex situations, they also need to be clearly explained.  

> [!done] 
>  Inserted the pseudocode for the algorithm used to generate the synthetic datasets 

> [!todo] See Overleaf 
> Modify the names $\mathcal{D}_O$ and $\mathcal{D}_I$ otherwise it seems like they are two different datasets 

4. During the experimental evaluation, consider adding some of the latest anomaly detection baseline models for comparison, rather than just limiting it to classic models such as IF, to demonstrate the superiority of the proposed method.  

> [!done] 
> We have already done this → we have inserted `DIF` and `AutoEncoders`

> [!todo] Add `ECOD`
>  See [PyOD Documentation](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.ecod)

5. In the interpretability evaluation, in addition to using feature selection as a proxy task, artificial datasets can also be designed, and based on the known ground-truth, a quantitative score can be given to the explanations provided by ExIFFI to more directly evaluate the quality of its explanations.  

> [!note] 
> We have already created artificial datasets with known ground-truth. Maybe we can introduce a sort of quantitative metric of the interpretability methods on the synthetic datasets (e.g. the famous Discounted Cumulative Gain?) 

6. For the correlation analysis between the explanations of ExIFFI and the outliers, it can be specifically examined whether the samples with high feature importance scores also correspond to the higher anomaly scores given by the algorithm, and whether there is a clear positive correlation between the two.  

> [!note] 
> This may be an interesting analysis to do. The only thing to understand is what he/she intends with samples with high importance scores → the ones with the highest value in their most important feature? the ones with the highest sum of importance values?  Obviously since here we are talking about single samples we have to use the `LFI` scores. 

> [!todo] 
> Do this procedure for each interpretation method we compared inside the paper. 
> - Do a table with the correlation values → add a column to the tables of the Average Precision,Time, 
> - For each dataset
> 	- Compute the `LFI` of all the samples 
> 	- Compute the sum of the `LFI` scores over all the features
> 	- Compute the Anomaly Score of the AD method 
> 	- Compute the correlation 

7. In addition to analyzing the differences of the model under different data distributions and anomaly types, the differences in the explanation quality of ExIFFI can also be analyzed, such as in which cases the model’s explanation is more reliable and more in line with human cognition. 

> [!warning] Focus on this
> Amplify the description (in particular on the `glass` and `moodify` datasets (Sections 6.2.4 and Section 6.2.7)) saying that we chose these datasets because they have a sort of ground truth on what are the most important features and the fact that they align with the human cognition and then we can highlight this fact in explaining the results. 

^7eb8d8

> [!warning] Review
>  I added some sentences on Sections 6.2.4 and Section 6.2.7 on the relevance of these datasets to evaluate the alignment of `ExIFFI` with human intuition/cognition but I will probably need to review a little bit what I wrote.

# Reviewer 3

This is a relevant article. However, it is still being prepared to be considered as a scientific paper. Below are some key observations:  

1. The title and content could be simpler for anyone to read and follow.  

> [!note] 
> Too general, how should we change this title? 

2. The objectives and the focus of the study need to be clarified.  

> [!note] 
> We have already clarified in the Introduction how we want to add interpretability in AD. Maybe we can expand a little bit the explanation saying how in the Industry 5.0 era (where interaction between humans and machines are crucial) it is important to have explanations on what are the most important features so that the domain experts can react accordingly.  

^af90c7

3. The authors spend most of their time describing what they have done without highlighting why it is unique and how it will benefit others. Unless these two questions are answered upfront, I am not sure why someone would read this work.  

> [!info] 
> See [[ExIFFI PAPER REVIEW EAAI#^af90c7|above]] 

^71192e

4. Too lengthy  

> [!note] 
> Simplifying the text and the sentences, as suggested also by other reviewers, we may be able to reduce also the length of the paper

5. There are too many details (Figures and Tables) that need to highlight their purpose. It reads more like a part of a thesis or a report.  
6. The topic is about relevant and if work is presented well may have an impact.  
7. The authors should expand their review of relevant work and present it succinctly. Many researchers have widely used advanced data-driven techniques, such as the Bayesian Network-based methods; researchers from C-RISE, Memorial University, Canada, have proposed scores of fault diagnosis and isolation models and methods that are extended to complex engineering system safety and risk analysis. The authors have also ignored these studies.  

> [!note] 
> There is not a clear link pointing out to the paper the reviewer is referring to and moreover the topic of the paper does not seem very well connected to the main topics of our paper (i.e. Anomaly Detection and Explainability) 

> [!done] 
> Already done by Alessio who inserted some sentences on the Related Work section.  

8. The authors should attempt to benchmark their study with other published work.

> [!done] 
> We already done it comparing to `DIF` and `AutoEncoder`. Maybe we can add the `ECOD` but this will require a lot of new experiments.  
# Reviewer 4

1. Include all important numerical results in the abstract.  

> [!note] 
> Here we have to carefully select which result to include because there are a lot of numerical results from all the experiments we did.  

> [!todo] 
> Bring the fact that ExIFFI is better in 10 datasets over 15 (as we wrote in the Section 6 describing the big tables) 

2. Better highlight new contributions and novelty of your work in the introduction section.  

> [!note] 
>  See [[ExIFFI PAPER REVIEW EAAI#^71192e]]

3. Enhance linkage to recent applied literature that demonstrates the great potential and usefulness of different machine learning models for forecasting purposes (doi: 10.1108/AJEB-01-2024-0007; 10.1016/j.meaene.2024.100001; 10.1007/s00521-024-09531-2; 10.1108/JM2-12-2023-0315; 10.1177/03019233241254891) to better motivate your present investigation.  

> [!note] 
> From an initial look at the papers linked above they seem to be very far from the topics covered in this paper. These papers in fact are all abour price forecasting with NN based models while here we are talking about Anomaly Detection and Interpretability.    

> [!done] 
> I inserted one of the papers (the one about yellow corn) in the very first paragraph of the paper (in the Introduction section) where we talk in very general terms about ML and AI and how they found applications in various domains.  

4. Provide a summary of previous studies in terms of what has been done, what is still missing, and correspondingly your contributions.  

> [!note] 
> I think we have already done something similar in the Related Work section 

5. Improve the quality of the figures to make them clearly visualized by using high-resolution formats. 
6. Improve the organization of the tables to make them clearly laid out. 

> [!note] 
> For these two points we need to work a little bit with Latex  

7. Add more explanations to the figures and tables to make them self-explanatory.  
8. Offer more detailed discussions of the results by linking to the figures and tables.  

> [!note] 
> Add some captions to the images and cite them in the explanations of the results  

9. Use more mathematical formulae and equations to demonstrate your proposed approach.  

> [!note] 
> I think that there are already a lot of mathematical formulas in the explanations of the proposed approaches. In any case adding the pseudocode we are also adding formulas. 

10. Conduct further benchmark analysis to demonstrate the advantage or disadvantage of your method.  

> [!note] 
> Also this comment is very general. We have already performed a lot of experiments. What should we do next? 

11. Elaborate on the potential limitations of your work, and correspondingly, the path to future studies in the conclusion.  

> [!note] 
> One of the possible limitations is that our method is model specific (works only with Isolation based methods) and maybe the fact that the Scenario II and the Feature Selection proxy task require to have labels (that are usually not available). 
> Possible research direction → apply to real-world industrial data and inside an IoT system.  

- More carefully proofread your manuscript to avoid writing issues.

> [!note] 
> Check for some keystroke mistakes I guess.  

## Pseudocode in Latex 

[[ExIFFI PAPER REVIEW EAAI#Reviewer 2|Reviewer 2]] in comment 1 suggests to provide the pseudocode of the algorithms presented (i.e. `DIFFI`, `EXIFFI`, Feature Selection Proxy task) in order to provide a clearer explanation of how they work → in this way we may reduce a little bit the algorithm explanation in plain text. As I saw in other papers it should be possible to produce algorithms pseudocode with Latex so let's try it out. 

### Feature Selection Pseudocode 

```
\begin{algorithm}
\caption{Feature Selection Proxy Task} \label{alg:fs}
\begin{algorithmic}[1]
\State \textbf{Input:} Isolation based AD model (i.e. IF,EIF,\EIFplus) $F$, $n$ different interpretation algorithms (i.e. \approach,DIFFI,Random Forest) $\{\mathcal{A}_1,\mathcal{A}_2,\dots,\mathcal{A}_n\}$, $GFI$ feature rankings from the interpretation algorithms (i.e. \approach,DIFFI,Random Forest) $\mathcal{R} = \{\mathcal{R}_1,\mathcal{R}_2,\dots,\mathcal{R}_n\} \in \mathcal{N}^p$ in decreasing order of importance, Input dataset $\mathcal{X} \in \mathbb{R}^{n \times p}$ 
\State \textbf{Output:} Average Precision vectors for each one of the interpretation algorithms under comparison: $\{AUC_{FS}_1,AUC_{FS}_2,AUC_{FS}_n\dots\} \in \mathbb{R}$, $\{AP_1,AP_2,\dots,AP_n\} \in [0,1]^p$
\State Initialize the Average Precision vectors at zero:
\For {$i \gets 1$ to $n$} 
    $AP_i \gets [0,\dots,0]$
\EndFor
\For {$i \gets 1$ to $p$}
\State Compute the Average Precision using the current set of features for all the interpretation algorithms and update the Average Precision vector:
\For {$a \in \mathcal{A}$}
    $AP_a \gets \text{average_precision}(F,\mathcal{X_a})$
\EndFor
\State Update the input dataset for each interpretation algorithm removing the least/most important feature according to the $GFI$ ranking
\For {$a \in \mathcal{A}$}
\State Inverse Approach: Remove the least important feature
    $\mathcal{X}_a \gets \mathcal{X}_a \setminus \mathcal{X}[:,\mathcal{R}_a[-1]]$
    $\mathcal{R}_a \gets \mathcal{R}_a[:-1]$
\State Direct Approach: Remove the  most important feature
    $\mathcal{X}_a \gets \mathcal{X}_a \setminus \mathcal{X}[:,\mathcal{R}_a[0]]$
    $\mathcal{R}_a \gets \mathcal{R}_a[1:]$
\EndFor
\EndFor
\For {$a \in \mathcal{A}$}
\State \textbf{Return} $AP_a$
\EndFor

\end{algorithmic}
\end{algorithm}
```
