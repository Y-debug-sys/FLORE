# Anonymous Implementation of the paper "On the (*Generative*) Linear Sketching Problem"

 > This repository contains the anonymous implementation of **FLORE**, the *first* deep generative framework for solving linear sketching problems in data streaming scenarios, submitted to ICDE ’27.

<p align="center"><img src="Figures/icon.png" width="100%"></p>

---

## ⚡ Getting started

### 📦 Dependencies

- Run `pip install -r requirements.txt` to install all Python dependencies.  
- 📌 [Miniconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) is required.  

### 📊 Datasets

For reproducibility, **all real-world data streams used in our experiments are included in the supplementary material** of our submission. After downloading, unzip and place them into the following folder:

```
├── Streams   # 🌐 Network traces or real-life streams (.dat)
```

---

## 📂 Code structure

```
.
├── 📁 FLORE                   # source code for FLORE (training / inference)
├── 📁 Baselines               # submodule for baselines
│   ├── 📁 classic_sketch               # Count-Min and Count
│   ├── 📁 compressed_sensing_sketch    # PR-sketch and NZE-sketch
│   ├── 📁 optimized_sketch             # Augmented and Conservative Update
│   └── 📁 ...                          # Others
├── 📁 Flows                   # submodule for Flow-based generative model
├── 📄 main.py                 # main file for simulation
├── 📄 helper.py               # config file for simulation
├── 📁 Streams                 # datasets (.dat)
├── 📁 Utils                   # useful modules (metircs, dataloader, ...)
├── 📁 Structure               # data structure
│   ├── 📁 cuda                         # Cuda implementation
│   ├── 📄 augmented_filter.py          # stream filtering mechanism
│   ├── 📄 bloom_filter.py              # key tracking mechanism
│   └── 📄 sketch.py                    # data-plane implementation
└── 📁 Scripts                 # scripts for running Pram
```

---

## 📈 Evaluating FLORE

Run the provided script to evaluate **FLORE** in comparison with some baselines on the **{dataset_name}**:

```bash
cd Scripts
bash {dataset_name}.sh
```

The script will automatically load or/and produce the corresponding data streams, and then execute **FLORE** with default settings.

---

## 📜 License

This repository will be released under a License once upon the acceptance.

---

## ✨ Citation

The citation information will be provided once the paper is accepted.
