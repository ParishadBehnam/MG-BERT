# MG-BERT

This is the source code of our paper, "Parishad BehnamGhader, Hossein Zakerinia, Mahdieh Soleymani Baghshah. (2021). MG-BERT: Multi Graph Augmented BERT in Masked Language Modeling."   

---

### Datasets
You may use the CoLA and SST datasets from the [GLUE repository](https://github.com/nyu-mll/GLUE-baselines) and the Brown dataset from the [Brown Corpus Manual](http://icame.uib.no/brown/bcm.html) to train and assess your models. The WN18 Knowledge Graph can also get accessed through [this repository](https://github.com/Mrlyk423/KG2E).

---

### Running the code

In order to run this code, first use the *prepare_data.py* to create the multi-graphs based on the corpus. 

    python prepare_data.py --dataset cola --kg WN11  

Then, you can train your MG-BERT model using *train.py*. Here, we train a model using a multi-graph consisting of tf-idf, pmi, and KG graphs.

    python train.py --dataset cola --kg WN11 --dyn 0.1 --graph-mode 123 --epoch 100  

Finally, evaluate your final model (via Hits@k metrics) via *evaluate.py*.   

    python evaluate.py --dataset cola --kg WN11 --dyn 0.1 --graph-mode 123 --epoch 100
    
Some parts of this project were originally implemented in the [VGCN-BERT repository](https://github.com/Louis-udm/VGCN-BERT) and [Huggingface transformers](https://github.com/huggingface/transformers/releases/tag/v0.6.2).
