# Synthetic-promoters-via-GAN
***
## 1. Introduction

### 1.1 Problem Definition 

During the early stages of the coronavirus, the world struggled to provide a vaccine to all humans. Due to the high manufacturing costs, vaccines have distribution and availability issues that prevent them from becoming available to everyone. Without a doubt, if there was a way to make this process less expensive, it would have a huge impact on our world.

All creatures need to produce proteins inside their bodies to sustain their lives, proteins are like soldiers at the service of the body's cells, and they play an essential role in all the vital reactions in our bodies. Within the human body, there are millions of proteins, each of which serves a specific purpose and has a distinct shape. The cell nucleus, specifically the DNA, contains all of the orders required for protein synthesis. As illustrated in figure.1, promoter sequences are DNA sequences that define where the transcription of a gene begins.

Designing synthetic promoters in labs would be a way to generate stronger diverse proteins, yet the length of a promoter varies from 100 -1000 base pairs, and since there exist 4 base pairs, A, G, C, and T, this expands our search space to 4^1000. So, we need In-silico (performed on a computer or via computer simulation) means in general and deep learning techniques, in particular. Thanks to AI technologies especially Generative adversarial networks (GAN) which offer to generate a new and highly diverse protein sequence with physical properties that resemble those of natural protein sequences. As we know in eukaryotes, the transcriptional complex can bend DNA, allowing regulatory sequences to be placed far from the transcription site. The distal promoter is upstream of the gene and may contain additional regulatory elements with a weaker influence. RNA polymerase II (RNAP II) bound to the transcription start site promoter can start mRNA synthesis.

So, our main objective is to study the structure of promoters and develop a method for promoter synthesis, which will be cheaper and faster than lab-based synthesis.

![Figure 1. Gene Expression Process (Central Dogma)](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Gene%20Expression%20Process%20(Central%20Dogma).PNG)

### 1.2 Background

A promoter is a sequence of DNA to which proteins bind to initiate transcription of a single RNA transcript from the DNA downstream of the promoter. Another definition of the promoter is a portion of DNA where RNA polymerase starts to transcribe a gene. Since promoters are the main regulators of gene transcription and accordingly gene expression (transformation from genotype to phenotype), promoters enable the control over gene expression mechanism and accordingly allow the induction of unnaturally occurring but desired features through genetic engineering.  

![Figure 2. Transcription Process](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Transcription%20Process.PNG)

The length of a promoter varies from 100-1000 base pairs and since there exist 4 base pairs, A, G, C, and T, this expands our search space to 41000. In-silico means in general and deep learning techniques, in particular, are proven to optimize the development of synthetic promoters conquering the limitation of the massive number of possible DNA combinations needing testing and verification by mimicking the most valid and strong naturally occurring promoters and existing synthetic ones. 

In the paper “Synthetic design of strong promoters”, they showed that the strongest eukaryotic enhancers are identical to sequences found in nature and generally have only minimal modification.  A promoter that evolved in the context of an organism will possess levels of transcriptional activation tuned to maximize the fitness of their complete host organism rather than seeking the strongest possible transcription level. Overly strong transcription levels could easily be detrimental to the overall fitness of a virus. The small size, spatial effects, and complexity resulting from the combinatorial mechanism of action of transcription factor binding sites, and the discovery of the exact elements responsible for transcriptional activation have proven difficult. They used in their work oligonucleotide microarrays enables the parallel generation of very large numbers of designed DNA sequences, which allows the systematic investigation of a large portion of transcription factor binding site sequence space [1].

In the paper “Synthetic promoter design in Escherichia coli based on a deep generative network”, they worked on generating a prokaryotic promoter resembling Escherichia coli. the experimental evidence showed that up to 70.8% of the AI-designed promoters were functional, and some of them displayed equivalent or even higher activity than the majority of active natural promoters and their strongest mutants. they used a Position-specific scoring matrix (PSSM) to find the –10 and –35 motifs, then used Discriminative Regular Expression Motif Elicitation (DERME) to the top 1% highest expression promoters and scanned using FIMO [2].

In the paper “Expanding functional protein sequence spaces using generative adversarial networks” they used in their work a protein sequence using chose malate dehydrogenase (MDH) to be their dataset. They build a Generative adversarial network (GAN) and compared it with Hidden Markov Model (HMM), they found that the GAN outperformed the HMM for all comparison points. In their GAN architecture, for Generator, they used two transposed convolutions for up-sampling and convolution layer with LeakyRelu an as activation function. For the discriminator, they used three convolution layers with LeakyRelu as an activation function except for the last layer which used the sigmoid function. For both networks, they used one self-attention layer. They used R1 regularization to generalize the model. For evaluation, they used ClustalW as multiple sequence alignment to remove the columns that consist of 75% gaps from further analysis, then used Amino-acid pair association matrices to investigate the local relationship between amino acids. After that, they used the pairwise Amino-acid frequency distribution to investigate global amino-acid relationships, and also used these frequency distributions were then used to calculate correlations between the training and generated sequences. The result that they get was that the positional variability in generated sequences was highly similar to that of the natural sequences, with peaks (high entropy) and valleys (low entropy) appearing at nearly identical positions in the sequence alignment, demonstrating an overall high correlation between the entropy values of generated and natural sequences. The average similarity of the positional order in the natural and generated sequences is 82% showing that ProteinGAN captures the local amino-acid relationships existing in natural sequences. we found strong correlations between the natural and generated sequences. thus, the HMMs do not capture global amino-acid relationships [3].

In the paper “Unsupervised representation learning with deep convolutional generative adversarial networks”, they built GAN with convolution layers. They replaced any pooling layers with stride convolutions (discriminator) and fractional-strided convolutions (generator), and they used the Batch norm in both the generator and discriminator. For the activation function, they used the Relu function except the last layer used Tanh in the generator, then for the discriminator, they used the LeakyReLU function except for the last layer they used the Sigmoid function [4].
•	In the paper “Wasserstein GAN”, they introduced the WGAN algorithm as an alternative to the traditional GAN. There are many differences between their model (WGAN) and the original GANs. To begin, instead of the discriminator network, they used critic, a neural network that does not use a Sigmoid as its last layer's activation function. Second, for each generator's training time, they train the critic five times. They also used Wasserstein loss as their loss function and introduced a concept known as weight clipping to enforce a Lipschitz constraint. Hopefully, it has been demonstrated that the new algorithm can improve learning stability and overcome the mode collapse problem [5].

Having generated our GAN-based hyperactive synthetic promoters, we can then use them for recombinant protein production relying on insects, in particular, and eukaryotes, in general, to produce the desired protein through its own system supporting the expression of the desired gene. This work unleashes the limits of GANs in designing hyperactive synthetic promoters for eukaryotes. 

## 2. Design Overview 

As illustrated in figure.3, our system could be broken down into six major stages, which are, understanding of business domain, data selection and preprocessing, deep learning models, promoters’ generation, evaluation, and finally deployment. To begin with business understanding we needed to identify what is a promoter and what makes a strong promoter. For the biological data we have used the dataset of the Drosophila melanogaster from the Eukaryotic Promoter Database (EPD). This selected dataset consists of almost 16000 promoter sequences, each sequence is of length 600 base pairs (A, T, C, G). Regarding the preprocessing phase, we have preformed one hot encoding. And for the modeling stage, we have built two generative models, which are, DCGAN and WGAN-GP, those models are used then to generate our desired synthetic promoters. After generating, we moved to evaluate our generated promoters, we used a tool named Berkely Drosophila Genome Project (BDGP) [6] to evaluate our generated promoters based on their activity score. Finally comes the deployment stage, we have built a GUI to help displaying the results. 

![Figure 3. Project’s main stages](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Project%E2%80%99s%20main%20stages.PNG)

### 2.1	Requirements

In Proteinea, they have patented a technology for using insect larvae as soft bioreactors to produce different proteins of interest, at mass-scale, with highly reduced costs. Proteinea inject insect larvae with their own internally developed DNA constructs that integrate into insect genomes using transposition. A suitable host fulfills the requirements of fast and cost-efficient reproduction to correspond to efficient protein production such as the Black Soldier Fly (BSF) due to its feeding on biowaste and high reproduction rate. Having our GAN-based hyperactive synthetic promoters, we can then use them for recombinant protein production relying on insects, in particular, and eukaryotes, in general, to produce the desired protein through its own system supporting the expression of the desired gene. This work unleashes the limits of GANs in designing hyperactive synthetic promoters for eukaryotes. To sum up, our main objective regarding the expectations of the stakeholders is to generate strong Drosophila promoters which will be used by Proteinea to generate their desired proteins.

### 2.2	Detailed Design 

#### 2.2.1 Business Understanding 

##### 2.2.1.1 What is a promoter?

A promoter is a region of DNA upstream of a gene where relevant proteins (such as RNA polymerase and transcription factors) bind to initiate transcription of that gene. The resulting transcription produces an RNA molecule (such as mRNA) [7]. Promoters are key elements that regulate gene expression at the level of transcription [8].

![Figure 4. Promoter](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Promoter.PNG)

##### 2.2.1.2 What makes a strong promoter?

A critical step in gene expression is the initiation of transcription at the core promoter. The core promoter is typically about 80 nucleotides long and encompasses the RNA start site. Core promoters consist of functional subregions, termed core promoter elements, or motifs, which confer specific properties to the core promoter. These core promoter elements include the TATA box, TFIIB recognition elements (BREu and BREd), downstream core element (DCE), Inr, MTE and DPE. Notably, the best-known core promoter motif, the TATA box, is estimated to be present in about 10–20% of human core promoters. The properties of each core promoter are specified by the presence or absence of the core promoter motifs [9].

![Figure 5. Promoter Motifs](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Promoter%20Motifs.PNG)

##### 2.2.1.3 Differences between eukaryotic and prokaryotic promoters.

Eukaryotic promoters consist of three main portions; core promoter, proximal promoter, and distal promoter. Examples for some eukaryotic promoters’ motifs are Pribnow box (TATA box), GC box, CAAT box. In the context of TATA box, it is a sequence of 5’-TATAA-3’ that is present in the core promoter region. To the TATA box, transcription factor proteins and histone proteins are bound. The binding of transcription factor proteins to the TATA box assists in the binding of RNA polymerase, which then results in the formation of transcription complex. In simple terms, the binding of these proteins will drive the process of transcription.  This process will be inhibited when histone proteins get bound to the TATA box. Therefore, the TATA box is an important promoter element that involves in regulation of the rate of eukaryotic transcription.

![Figure 6. Eukaryotic Promoters’ Motifs](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Eukaryotic%20Promoters%E2%80%99%20Motifs.PNG)

The prokaryotic promoter contains only three types of promoter elements. The less involvement of promoter elements in prokaryotes is the main reason why their transcription process is less sophisticated when compared with eukaryotic transcription that involves a higher number of promoter sequences. Out of the three promoter elements of prokaryotes, there are two main important two short DNA sequences. These sequences are classified according to their location. They are, -10 promoters or element (that is present 10bp upstream of transcription start site), -35 promoters or element (that is present 35bp upstream of transcription start site). The -10 promoter is equivalent to the eukaryotic TATA box or Pribnow box and is an essential component for the initiation of transcription in prokaryotes. The -35 promoter consists of a sequence that is TTGACA that actively involves in the regulation of the rate of prokaryotic transcription.

![Figure 7. Prokaryotic Promoter’s Motifs](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Prokaryotic%20Promoter%E2%80%99s%20Motifs.PNG)

Table 1: Differences between eukaryotic and prokaryotic promoters [11]

| **Eukaryotic Promoters**                                                                                 | **Prokaryotic Promoters**                                |
|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
|Eukaryotic promoters are the regulatory sequences that initiate the transcription of eukaryotic organisms.|Prokaryotic promoters are the regulatory sequences that initiates the transcription of prokaryotic genes.|
|**Elements**| **Elements** |                                                                                            
| Eukaryotic promoter consists of Pribnow box (TATA box), CAAT box, GC box and initiator elements.| Prokaryotic promoter consists of upstream elements, -10 element and -35 elements.|

##### 2.2.1.4 Database

We have been searching for a while to understand the biological databases and select a relevant one. Our selected database was the Eukaryotic Promoter Database (EPD). EPD is an annotated non-redundant collection of eukaryotic POL II promoters, for which the transcription start site has been determined experimentally. Access to promoter sequences is provided by pointers to positions in nucleotide sequence entries. EPD is structured in a way that facilitates dynamic extraction of biologically meaningful promoter subsets for comparative sequence analysis. EPD is a collection of databases of experimentally validated promoters for selected model organisms. Evidence comes from TSS-mapping from high-throughput experiments such as CAGE and Oligocapping. The resulting databases are the following:

•	**Animals** : 

    o	Homo sapiens: 29598 promoters.
    o	Macaca mulatta: 9575 promoters.
    o	Mus musculus: 25111 promoters.
    o	Rattus norvegicus: 12601 promoters.
    o	Gallus gallus: 6127 promoters.
    o	Canis familiaris: 7545 promoters.
    o	Drosophila melanogaster: 16972 promoters.
    o	Apis mellifera: 6493 promoters.
    o	Danio rerio: 10728 promoters.
    o	Caenorhabditis elegans: 7120 promoters.

•	**Plants** :

    o	Arabidopsis thaliana: 22703 promoters.
    o	Zea mays: 17081 promoters.
  
•	**Fungi** :

    o	Saccharomyces cerevisiae: 5117 promoters.
    o	Schizosaccharomyces pombe: 4802 promoters.
  
•	**Invertebrates** :

    o	Plasmodium falciparum: 5597 promoters [12].

![Figure 8. EPD Species Counts](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/46b8434ce660506c81470a38eb19761407a505a7/images/EPD%20Species%20Counts.PNG)

##### 2.2.1.5 Sequence Alignment

Sequence alignment is a way of arranging sequences of DNA, RNA, or protein to identify regions of similarity made to align the entire sequence. the similarity may indicate the functional, structural, and evolutionary significance of the sequence. The sequence alignment is made between a known sequence and an unknown sequence (pairwise alignment) or between two unknown sequences (multiple alignment). The known sequence is called the reference sequence. The unknown sequence is called a query sequence. We have been searching for a while to select a proper alignment tool as displayed in table 3, until we settled down on MAFFT. We have picked MAFFT for its high efficiency. The CPU time of MAFFT is drastically reduced as compared with CLUSTALW with comparable accuracy. Also, MAFFT is over 100 times faster than T-COFFEE, when the number of input sequences exceeds 60, without sacrificing the accuracy [13].

Table 2: Alignment Tools

| Tool | Input Format | Output Format | Web | Max # seq | File Size | Sequence Type | Method | Server |
|------|--------------|---------------|-----|-----------|-----------|---------------|--------|--------|
| CLUSTAL-OMEGA | FASTA, EMB, GenBank	| ClustalW/Pearson/FASTA/MSF | Yes | Max 4000 sequences | Max file size of 4 MB. | Protein, DNA, RNA	| global/ Progressive |	[Here](http://www.clustal.org/omega/)|
| MUSCL |	FASTA, EMB, GenBank |	Fasta, Clustalw, MSF/html	| Yes	| Up to 500 sequences |	Max file size of 1 MB.	| Protein	| Progressive Step1 and Step2 iterative Step 3	| [Here](http://www.drive5.com/muscle/)|
| MAFFT |	FASTA, EMB, GenBank |	ClustalW/Pearson/FASTA | Yes |	Up to 500 sequences |	Max file size of 1 MB.	| Protein, DNA, RNA	| global/ Iterative	| [Here]( https://www.ebi.ac.uk/T) |
| KALIGN | FASTA, EMB, GenBank | MACSIM/ ClustalW/ Pearson/FASTA	| Yes| 	Up to 2000 sequences	|Max file size of 2 MB.| 	Protein, DNA, RNA	| Progressive |	[Here]( http://msa.sbc.su.se/cgi-bin/msa.cgi) |
|RETALIGN |	FASTA |	ClustalW |	No version 0.22 |	Max 1000 sequences	| Not limited	|Protein |	Progressive Corner cutting Multiple Sequence Alignment |	[Here]( http://phylogenycafe.elte.hu/RetAlign/) |
| PROBCONS|	MFA	| MFA/ ClustalW	| Yes version 1.12	| Max 1000 sequences	| Not limited	| Protein |	Probabilistic Consistency-based Multiple Alignment of Amino Acid Sequences |	[Here](http://probcons.stanford.edu/)|

#### 2.2.2 Data Selection and Preprocessing

##### 2.2.2.1 Selected dataset

After selecting the Eukaryotic Promoter Database (EPD) to be our database, we have then picked the Drosophila melanogaster promoters to be our dataset. The fruit fly Drosophila melanogaster is a versatile model organism that has been used in biomedical research for over a century to study a broad range of phenomena. There are many technical advantages of using Drosophila over vertebrate models; they are easy and inexpensive to culture in laboratory conditions, have a much shorter life cycle, they produce large numbers of externally laid embryos, and they can be genetically modified in numerous ways [14].

##### 2.2.2.2 Preprocessing
Data preprocessing is a critical preliminary step which include any processing technique performed on the raw data to transform it to a more reasonable and usable format, to be then used properly by machine and deep learning models.

###### 2.2.2.2.1 Preprocessing: Remove Duplicates 
Our first preprocessing technique was performing multiple sequence alignment between all the Drosophila promoters’ sequences to remove any duplicated sequence.



###### 2.2.2.2.2 Preprocessing: Length and Character Checking
Following the removal of duplicates, we needed to ensure that all promoters’ sequences have the same length and consist of the only valid base pairs (A, C, G, T).

###### 2.2.2.2.3 Preprocessing: Encoding
The most crucial preprocessing technique regarding our scope is encoding, which means transform all the DNA sequences (strings of ACGTs) into a matrix of numbers. One common strategy is to one-hot encode the DNA: treat each nucleotide as a vector of length 4, where 3 positions are 0 and one position is a 1, depending on the nucleotide, as seen in Figure 9.

![Figure 9. One-hot Encoding](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/One-hot%20Encoding.PNG)

#### 2.2.3 Deep Learning Models
Generative Adversarial Networks (GANs) are an approach to generative modeling using deep learning methods. GANs contain two ‘adversarial’ networks: the generator G and the discriminator D. The generator tries to capture the data distribution and produces artificial samples to fool the discriminator, whereas the discriminator tries to distinguish generated samples from training data. 

##### 2.2.3.1 DCGAN
Deep Convolutional Generative Adversarial Network (DCGAN) is a type of GAN which uses convolution layers as its backbone.
###### 2.2.3.1.1 DCGAN: Generator
As illustrated in Figure 10, our generator neural network consists of seven layers of transposed convolution followed by two convolution layers. Each transposed convolution layer is followed by a spectral norm layer for the purpose of normalization. Also, two self attention layers are used after each block of three transposed convolution layers.

![Figure 10. DCGAN Generator](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/DCGAN%20Generator.PNG)

###### 2.2.3.1.2 DCGAN: Discriminator
As illustrated in Figure 11, the discriminator neural network consists of seven convolution layers. Each convolution layer is followed by a spectral norm layer for the purpose of normalization. Also, two self attention layers are used after each block of three convolution layers. The last convolution layer is using a Sigmoid function as an activation function to scale the output between 0 and 1 for the purpose of classifying the natural and the synthetic promoters.

![Figure 11. DCGAN Discriminator](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/DCGAN%20Discriminator.PNG)

###### 2.2.3.1.3 DCGAN: Loss Function
The original loss function for GANs is called Jensen-Shannon (JS) divergence. It is a minmax game between the two networks which could be described in the following function [15]:

![Figure 12. Jensen-Shannon Divergence equation](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Jensen-Shannon%20Divergence%20equation.PNG)

##### 2.2.3.2 WGAN-GP
Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) is a type of GAN which uses Wasserstein loss as its loss function and consists of two neural networks: generator and critic.

###### 2.2.3.2.1 WGAN-GP: Generator
We have used the same generator architecture of DCGAN as illustrated in Figure 10.

###### 2.2.3.2.2 WGAN-GP: Critic (Discriminator)
We have used almost the same discriminator architecture of DCGAN as illustrated in Figure 11. Yet there are two main differences, firstly, instead of the spectral normalization layers, instance normalization layers are used. Secondly, we didn’t use Sigmoid as the last activation function.

###### 2.2.3.2.3 WGAN-GP: Loss Function
The original loss function for WGAN is called Wasserstein loss. The Wasserstein distance is the minimum cost of transporting mass in converting the data distribution q to the data distribution p. As written in the formula below, a gradient penalty term is used to enforce the Lipschitz constraint [16].

![Figure 13. Wasserstein equation](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Wasserstein%20equation.PNG)

#### 2.2.4 Promoters’ Generation

##### 2.2.4.1 Promoters’ Generation via DCGAN
After building and training the two neural networks of DCGAN, now it is time to generate new synthetic promoters giving the natural ones. One thousand new promoters’ sequences have been generated by our model. The highest promoter activity score achieved was 94%, which is still below the reference score.

##### 2.2.4.2 Promoters’ Generation via WGAN-GP
After building and training the two neural networks of WGAN-GP, now it is time to generate new synthetic promoters giving the natural ones. One thousand new promoters’ sequences have been generated by our model. The highest activity score achieved was 99% which outperformed the natural promoters.

#### 2.2.5 Evaluation and Filtration

##### 2.2.5.1 Sequence Alignment
There are two kinds of Sequence Alignment (MSA) have been applied. Firstly, Multiple Sequence Alignment (MSA) between all the generated promoters to ensure that there are no duplicated sequences then eliminate any redundant sequences. Secondly, Pairwise Sequence Alignment between the generated promoters and the Drosophila ones to ensure that the generated promoters resemble the natural Drosophila ones.

##### 2.2.5.2 Promoter Activity
Regarding the evaluation of our synthetic promoters, we needed to measure the promoter’s activity, so we selected a tool named Berkeley Drosophila Genome Project (BDGP). The BDGP evaluates the giving promoters and gives them an activity score based on the presence of some motifs such as the TATA box. As a filtration procedure, we have set the threshold to 85%, so any promoter that gets a lower activity score will be removed from the generated dataset

#### 2.2.6 Deployment

##### 2.2.6.1 GUI
We have built a website to display our work. Firstly, the users may enter their desirable model. Then, they need to determine which species to be used, regarding our project we are just concerned with Drosophila, but we may add extra species in the future. Finally, the users shall enter the number of generated promoters and press generate. After the promoters are generated, users can download their generated sequences as well as their evaluation files. The evaluation file contains the activity score that each promoter gets. Evaluation also gives us the start and the end points of the promoter sequence.

## 3	Results and Comparison with State of Art Models
The dataset had been trained and generated by these two models which were DCGAN and WGAN-GP. The generator and discriminator of these two models were trained for 500 epochs.
First, in the DCGAN model, we used the loss function which was JS Divergence. After applying the loss function, we trained the model and updated the weight of both the generator and discriminator and optimized them by Adam optimizer with a learning rate of 0.000001 and momentum of 0.5 at each epoch. The result in DCGAN was that we faced a collapse problem so we used the WGAN-GP to solve it. Second, we applied the WGAN-GP model in our data with the loss function Wasserstein loss. The generator was the same as the generator of DCGAN. For the discriminator, we used also the same architecture of DCGAN with some modifications which replaced spectral norm with instance norm and removed the sigmoid layer to make the network critic. In the loss function, we used Wasserstein loss to train the critic and the generator models that promote large differences between scores for real and generated sequences. During training, the weights of the critic were updated and optimized using the Adam optimizer with a learning rate of 0.0001 and momentum of 0 at each epoch. for generator, the weights were updated and optimized with Adam optimizer with a learning rate and momentum the same as the critic at every five epochs. After training the two models, we generated a thousand promoters from each model and saved the generated promoters into Fasta files for evaluation.

![Figure 14. DCGAN Generated Promoters](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/DCGAN%20Generated%20Promoters.PNG)

![Figure 15. WGAN-GP Generated Promoters](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/WGAN-GP%20Generated%20Promoters.PNG)

## 4	Evaluation 
Evaluation is done in three stages. The first stage, using the Maffit algorithm to check if there is any duplication or not, Fortunately, there is no duplication on generated promoters. In the second stage, using pairwise alignment to check the similarity of functionality between Drosophila melanogaster promoters and generated promoters, the result was the best similarity score is 63.5% to 53%.  In the final stage, select only the top five scores of similarity to calculate promoter activity using the Berkley drosophila genome product (BDGP). As we know that is a hard step, but we reached these high results. Our result of promoter activity was the promoter id DCGAN 414, DCGAN 282, and WGAN-GP 525 which were 1,1, and 0.99 receptively  is better than its reference Drosophila FP004187_Hsp70Ba_1 which was 0.96.

![Figure 16. Part of Maffit Result of DCGAN](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Part%20of%20Maffit%20Result%20of%20DCGAN.PNG)

![Figure 17. Part of Maffit Result of WGAN-GP](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Part%20of%20Maffitt%20Result%20of%20WGAN-GP.PNG)

![Figure 18. Part of Similarity Scores of DCGAN](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Part%20of%20Similarity%20Scores%20of%20DCGAN.PNG) ![Figure 19. Part of Similarity Scores of WGAN-GP](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/1cf537cdaf3a64f66bcddc217765d442b1091c57/images/Part%20of%20Similarity%20Scores%20of%20WGAN-GP.PNG)

![Figure 20. Promoter Activity of top similarity promoters ](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/46b8434ce660506c81470a38eb19761407a505a7/images/Promoter%20Activity%20of%20top%20similarity%20promoters.PNG)

## 5 Deployment Plan  

### 5.1	Design
The design depended on smooth and dark colors which make the users comfortable when using our UI. The opacity of the background was 47 to be smooth when looking at it. The double strands of DNA background, icons, font, and colors were suitable to our project idea. Our pages contain the logo of the university (Ottawa), and logo of our company sponsor (Proteinea), and the name of our project. It is a simple, and comfortable design to enable users to use it easily as we see in the Figures 21:23.

![Figure 21. Design of the first page](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/46b8434ce660506c81470a38eb19761407a505a7/images/Design%20of%20the%20first%20page.PNG)
![Figure 22. Design of the second page](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/46b8434ce660506c81470a38eb19761407a505a7/images/Design%20of%20the%20second%20page.PNG)
![Figure 23. Design of the third page](https://github.com/TokaMamdoh/Synthetic-promoters-via-GAN/blob/46b8434ce660506c81470a38eb19761407a505a7/images/Design%20of%20the%20third%20page.PNG)

### 5.2	Front-End & Back-End Phases
First, in the front end, we used HTML, CSS, Bootstrap5, and JQuery to can implement our design. Second, in the back end, we used the Node.js template engine to integrate FE with BE server using EJS. Third, the AI server uses a flask and loads the saved model, and communicates between the AI server and BE server throw REST API using Axios request.

### 5.3	Testing
We have built a website to display our work. Firstly, the users may enter their desirable model. Then, they need to determine which species to be used, regarding our project we are just concerned with Drosophila, but we may add extra species in the future. Finally, the users shall enter the number of generated promoters and press generate. After the promoters are generated, users can download their generated sequences as well as their evaluation files. The evaluation file contains the activity score that each promoter gets. The evaluation also gives the start and the end points of the promoter sequence.

## 6 Conclusions and Future Works

In this project, we described the strong promoters are the promoters that maximize the binding between it and RNA polymerase II. The goal of our project is to synthesize promoters that maximize the binding to RNA polymerase II. For the dataset, we selected the Drosophila Melanogaster dataset for several reasons, first their relationship between their genes and human genes is very close, simple cultivating environment, it has a simple and short reproductive cycle, and inexpensive. For preprocessing, we checked if the data had the same length or not, if they had any strange characters and finally convert the sequence into numeric using one hot encoder. Then we implemented two models which were DCGAN and WGAN-GP and generate from each model one thousand promoters. Finally evaluated the promoters used bioinformatics tools, first using MSA to check if there was any duplication or not, then using pairwise alignment to check if their functionality resembles the natural one or not and the last evaluation calculates the activity of the promoter to figure out that the generated promoters are outperformed the natural one or not. The result of our project is that the generated promoter whose id is WGAN 525 outperformed the natural one which was FP004187_Hsp70Ba_1. The promoter activity of FP004187_Hsp70Ba_1 was 0.96 and the promoter activity of WGAN 525 was 0.99. 

In future works, we will use transfer learning to utilize the parameters of DNA-based pre-trained transformer on our GANs, will utilize the performance scores of the previous attempts in deriving possible model improvements as well as computational optimization means, and will experimentally validate that our synthetically generated sequences perform as expected.

The path that we followed consumed time and costs, and the length of promoters is within 100 -1000 base pairs, and since there exist 4 base pairs, A, G, C, and T, this expands our search space to 4^1000. This imposes a significant load on experimental methods designed to find novel sequences with improved properties. Instead of relying on a semi-random local search procedure, machine learning approaches directly infer promoter features and functions from the nucleotide sequence since they are not constrained by the number of sequence variations that may be processed. To meet the challenges and demands for novel promoter diversity in the biomedical and biotechnology fields, computational approaches that generate novel functional sequence variants without resorting to an experimental screening of the enormous promoter sequence space are becoming increasingly crucial.

## 7 References 

[1]	R. Schlabach, M. et al. (no date) Synthetic design of strong promoters | PNAS. Available at: https://www.pnas.org/doi/10.1073/pnas.0914803107 (Accessed: January 15, 2023).

[2]	Ye Wang et al. (2020) Synthetic promoter design in Escherichia coli based on a deep generative network, Academic.oup.com. Available at: https://academic.oup.com/nar/article/48/12/6403/5837049 (Accessed: January 15, 2023).

[3]	Repecka, D. et al. (2021) Expanding functional protein sequence spaces using generative adversarial networks, Nature News. Nature Publishing Group. Available at: https://www.nature.com/articles/s42256-021-00310-5 (Accessed: January 15, 2023).

[4]	Radford, A., Metz, L. and Chintala, S. (2016) Unsupervised representation learning with deep convolutional generative Adversarial Networks, arXiv.org. Available at: https://arxiv.org/abs/1511.06434 (Accessed: January 15, 2023).

[5]	Arjovsky, M., Chintala, S. and Bottou, L. (2017) Wasserstein Gan, arXiv.org. Available at: https://arxiv.org/abs/1701.07875 (Accessed: January 15, 2023).

[6]	BDGP: Home. [Online]. Available: https://www.fruitfly.org/. [Accessed: 15-Jan-2023].

[7]	“Promoter,” Genome.gov. https://www.genome.gov/genetics-glossary/Promoter

[8]	 Ye Wang, Haochen Wang, Lei Wei, Shuailin Li, Liyang Liu, Xiaowo Wang, Synthetic promoter design in Escherichia coli based on a deep generative network, Nucleic Acids Research, Volume 48, Issue 12, 09 July 2020, Pages 6403–6412, https://doi.org/10.1093/nar/gkaa325

[9]	Juven-Gershon, T., Cheng, S. & Kadonaga, J. Rational design of a super core promoter that enhances gene expression. Nat Methods 3, 917–922 (2006). https://doi.org/10.1038/nmeth937

[10] V. Brázda, M. Bartas, and R. P. Bowater, “Evolution of Diverse Strategies for 
  Promoter Regulation,” Trends in Genetics, vol. 37, no. 8, pp. 730–744, Aug. 2021,
  doi: 10.1016/j.tig.2021.04.003.

[11] “Difference Between Eukaryotic and Prokaryotic Promoters,” Compare the 
  Difference Between Similar Terms. https://www.differencebetween.com/difference- 
  between-eukaryotic-and-vs-prokaryotic-promoters/

[12] “EPD - Eukaryotic Promoter Database,” Epfl.ch, 2012. https://epd.epfl.ch//index.php

[13] Katoh K, Misawa K, Kuma K, Miyata T. MAFFT: a novel method for rapid multiple 
        sequence alignment based on fast Fourier transform. Nucleic Acids Res. 2002 Jul 
        15;30(14):3059-66. doi: 10.1093/nar/gkf436. PMID: 12136088; PMCID: PMC135756.

[14] Barbara H. Jennings, Drosophila – a versatile model in biology & medicine, Materials 
       Today, Volume 14, Issue 5, 2011, Pages 190-195, ISSN 1369-7021, 
       https://doi.org/10.1016/S1369-7021(11)70113-4.

[15] Goodfellow, Ian & Pouget-Abadie, Jean & Mirza, Mehdi & Xu, Bing & Warde-Farley, 
        David & Ozair, Sherjil & Courville, Aaron & Bengio, Y.. (2014). Generative Adversarial 
        Networks. Advances in Neural Information Processing Systems. 3. 10.1145/3422622.

[16] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein GAN,” arXiv.org, 2017.
        https://arxiv.org/abs/1701.07875

[17] “DCGAN Tutorial — PyTorch Tutorials 1.6.0 documentation,” pytorch.org. 
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

[18] “Papers with Code - WGAN GP Explained,” paperswithcode.com.
  https://paperswithcode.com/method/wgan-
  gp#:~:text=Wasserstein%20GAN%20%2B%20Gradient%20Penalty%2C%20or 
  (accessed Jan. 15, 2023).
