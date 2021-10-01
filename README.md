# SSGVQAP
# A Smart System to Generate and Validate Question Answer Pairs for COVID-19 Literature 

ABSTRACT: Automatically generating question answer (QA) pairs from the rapidly growing coronavirus-related literature is of great valueto the medical community. Creating high quality QA pairs would allow researchers to build models to address scientific queries for answers which are not readily available in support of the ongoing fight against the pandemic. QA pair generation is, however, a very tedious and time consuming task requiring domain expertise for annotation and evaluation. In this paper we present our contribution in addressing some of the challenges of building a QA system without gold data. We first present a method to create QA pairs from a large semi-structured dataset through the use of transformer and rule-based models.
Next, we propose a means of engaging subject matter experts (SMEs) for annotating the QA pairs through the usage of a web application.
Finally, we demonstrate some experiments showcasing the effectiveness of leveraging active learning in designing a high performing model with a substantially lower annotation effort from the domain experts.

 ![Screen Shot 2021-09-30 at 10 10 23 AM](https://user-images.githubusercontent.com/48797451/135471548-315ef288-7c34-4573-a4c1-b87a2b719973.png)

FIGURE 1: Generation of the Silver QA Data and the Process of Obtaining Gold Data using Active Learning

# Quick Links
DATA

SETUP

TRAINING

# License

# Citation

Please consider citing our work if you found it useful in your research:

```
@inproceedings{bhambhoria-etal-2020-smart,
    title = "A Smart System to Generate and Validate Question Answer Pairs for {COVID}-19 Literature",
    author = "Bhambhoria, Rohan  and
      Feng, Luna  and
      Sepehr, Dawn  and
      Chen, John  and
      Cowling, Conner  and
      Kocak, Sedef  and
      Dolatabadi, Elham",
    booktitle = "Proceedings of the First Workshop on Scholarly Document Processing",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.sdp-1.4",
    doi = "10.18653/v1/2020.sdp-1.4",
    pages = "20--30",
    abstract = "Automatically generating question answer (QA) pairs from the rapidly growing coronavirus-related literature is of great value to the medical community. Creating high quality QA pairs would allow researchers to build models to address scientific queries for answers which are not readily available in support of the ongoing fight against the pandemic. QA pair generation is, however, a very tedious and time consuming task requiring domain expertise for annotation and evaluation. In this paper we present our contribution in addressing some of the challenges of building a QA system without gold data. We first present a method to create QA pairs from a large semi-structured dataset through the use of transformer and rule-based models. Next, we propose a means of engaging subject matter experts (SMEs) for annotating the QA pairs through the usage of a web application. Finally, we demonstrate some experiments showcasing the effectiveness of leveraging active learning in designing a high performing model with a substantially lower annotation effort from the domain experts.",
}
```

# Acknowledgements
The authors would like to thank all the organizers of the COVID-19 Open Research Dataset Kaggle Challenge. We would like to thank Vector Institute
for making this collaboration possible and providing academic infrastructure and computing support during all phases of this work. We would also
like to thank Richard Pito from Thomson Reuters for his invaluable feedback and support throughout this project. Last but not least, special thanks to
Dr. Frank Rudzicz and Dr. Xiaodan Zhu for their
academic supervision and insights.
