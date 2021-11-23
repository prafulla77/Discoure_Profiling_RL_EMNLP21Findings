# Discoure_Profiling_RL_EMNLP21Findings

Run:
```
python3.6 main.py --drop 0.5 --learn_rate 5e-5 --seed 0
```

Cite:
```
@inproceedings{choubey-huang-2021-profiling-news,
    title = "{P}rofiling News Discourse Structure Using Explicit Subtopic Structures Guided Critics",
    author = "Choubey, Prafulla Kumar  and
      Huang, Ruihong",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.137",
    pages = "1594--1605",
    abstract = "We present an actor-critic framework to induce subtopical structures in a news article for news discourse profiling. The model uses multiple critics that act according to known subtopic structures while the actor aims to outperform them. The content structures constitute sentences that represent latent subtopic boundaries. Then, we introduce a hierarchical neural network that uses the identified subtopic boundary sentences to model multi-level interaction between sentences, subtopics, and the document. Experimental results and analyses on the NewsDiscourse corpus show that the actor model learns to effectively segment a document into subtopics and improves the performance of the hierarchical model on the news discourse profiling task.",
}
```
