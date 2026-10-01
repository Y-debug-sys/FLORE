# FLORE: On the (*Generative*) Linear Sketching Problem

**Xinyu Yuan, Yan Qiao, Zonghui Wang, Wenzhi Chen**

> This repository contains the official implementation of **FLORE**, the *first* deep generative framework for solving linear sketching problems in data streaming scenarios, accepted to ICDE ’27 (first round).

<p align="center"><img src="Figures/icon.png" width="100%"></p>

---

## 📚 Abstract

sketch techniques have been extensively studied in recent years and are especially well-suited to data streaming scenarios, where the sketch summary is updated quickly and compactly. However, it is challenging to recover the current state from these summaries in a way that is accurate, fast, and real. In this paper, we seek a new solution that reconciles this tension, aiming for near-perfect recovery with lightweight computational procedures. Focusing on linear sketching problems of the form Φf → f, our study proceeds in three phases. First, we revisit existing techniques and show the root cause of the dilemma: an orthogonal information loss during the summarization process. Second, we investigate how generative priors can be leveraged to bridge the gap and thereby allow theoretically superior recovery. Third, we present FLORE, the first generative sketching paradigm to embrace these analyses. At its core, FLORE integrates invertible flows with efficient sketch instances to achieve the best of two worlds. More importantly, FLORE can be trained online without access to ground-truth data, which is readily deployable in practice. 

## ⚡ Getting started

### 📦 Dependencies

- Run `pip install -r requirements.txt` to install all Python dependencies.  
- 📌 [Miniconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) is required.  

### 📊 Datasets

For reproducibility, **all real-world data streams used in our experiments are publicly available**. Download them from [Google Drive](https://drive.google.com/file/d/1lEYtnUl3bDaJXiaE02mgV_8CUGpvXuz9/), then unzip and place them into the following folder:

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
│   ├── 📁 optimized_sketch             # Augmented, Elastic and Conservative Update
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
├── 📁 Scripts                 # scripts for running Pram
└── 📁 PDFs                    # technical report (PDF)
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

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ✨ Citation

If you find this work useful, please cite:

```bibtex
@article{yuan2026generative,
  title={On the (Generative) Linear Sketching Problem},
  author={Yuan, Xinyu and Qiao, Yan and Wang, Zonghui and Chen, Wenzhi},
  journal={arXiv preprint arXiv:2603.14474},
  year={2026}
}
```
