# RCL: Relation Contrastive Learning for Zero-Shot Relation Extraction
Official implementation of "[RCL: Relation Contrastive Learning for Zero-Shot Relation Extraction](https://aclanthology.org/2022.findings-naacl.188/)", NAACL2022 findings.
## Prepare data
First, you can download the datasets employed in our work from the following link:
- [SemEval2010 Task8](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)
- [FewRel](https://thunlp.github.io/1/fewrel1.html)

Then, preprocess datas to same format as example file in the ./data folder.

## Dependencies
We use anaconda to create python environment:
```
conda create --name python=3.6
```
Install all required libraries:
```
pip install -r requirements.txt
```

## Train
```
python train.py --bert_model bert-base-uncased --data_path ./data/semeval_data.csv  --save_result output/ 
```
- --bert_model: The name or path of a transformers-based pre-trained checkpoint, e.g., "bert-base-uncased".
- --data_path: The path of preprocessed data.
- --save_result: The directory contains model file and eval results.


## Citation
If you use the code, we appreciate it if you cite the following paper:
```
@inproceedings{wang-etal-2022-rcl,
    title = "{RCL}: Relation Contrastive Learning for Zero-Shot Relation Extraction",
    author = "Wang, Shusen  and
      Zhang, Bosen  and
      Xu, Yajing  and
      Wu, Yanan  and
      Xiao, Bo",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.188",
    doi = "10.18653/v1/2022.findings-naacl.188",
    pages = "2456--2468",
    abstract = "Zero-shot relation extraction aims to identify novel relations which cannot be observed at the training stage. However, it still faces some challenges since the unseen relations of instances are similar or the input sentences have similar entities, the unseen relation representations from different categories tend to overlap and lead to errors. In this paper, we propose a novel Relation Contrastive Learning framework (RCL) to mitigate above two types of similar problems: Similar Relations and Similar Entities. By jointly optimizing a contrastive instance loss with a relation classification loss on seen relations, RCL can learn subtle difference between instances and achieve better separation between different relation categories in the representation space simultaneously. Especially in contrastive instance learning, the dropout noise as data augmentation is adopted to amplify the semantic difference between similar instances without breaking relation representation, so as to promote model to learn more effective representations. Experiments conducted on two well-known datasets show that RCL can significantly outperform previous state-of-the-art methods. Moreover, if the seen relations are insufficient, RCL can also obtain comparable results with the model trained on the full training set, showing the robustness of our approach.",
}
```
