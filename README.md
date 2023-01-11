# DWave model
for comparing whether two songs are same song or not

![FlowChart](DWave_Model.png "Flow Chart")

## Setup
```bash
# Install python dependencies
pip install -r requirements.txt
```

## Prediction
```bash
python3 model_run.py
```

## Description
The model contains two parts:
1.  DSSM model: comparing contributors of songs
2.  PERT model: comparing titles of songs

```
DSSM
│── test_bmat_contributors_match.py (To train DSSM)
│
├── Data (The necessary lexicon and the example corpus)
│   │
│   ├── contributors_dict.json (The dictionary of contributors)
│   │
│   ├── QA_DSP2_2020S2_2 (dw)_checked.xlsx (The training data)
│   │
│   └── QA_DSP1_20221h_Suspense - DW.xlsx (The testing data)
│
└── Output 
    │
    ├── 2022_secondhalf_negative_threshold_070.xlsx # All the data which is predicted False when threshold = 0.70
    │
    └── 2022_secondhalf_negative_threshold_085.xlsx # All the data which is predicted False when threshold = 0.85

```
```
PERT
│── dssm_process.py (To process the output of DSSM for the input of PERT)
│── PinyinCharDataProcesser.py (To provide the dataset)
│── py2wordPert.py (To do the Pinyin-to-character conversion task by PERT)
│
├── NEZHA (The NEZHA language model)
│
├── Configs (The configurations to train PERT at various scals)
│
├── Corpus (The necessary lexicon and the example corpus)
│   ├── CharListFrmC4P.txt (The list of Chinese characters)
│   ├── pinyinList.txt (The list of pinyin tokens)
│   ├── ModernChineseLexicon4PinyinMapping.txt (The word items and the corresponding pinyin tokens in Modern Chinese Lexicon)
│   ├── PERT_title_Chinese_test.txt (The corpus of Chinese character)
│   └── PERT_title_pinyin_test.txt (The corpus of pinyin)
│
└── Models 
    ├── Bigram (The Bigram model trained on some news corpus)
    └── pert_tiny_py_lr5e4_10Bs_1e (The PERT model trained on some news corpus under the conditions of learning rate: 5e-4, batch size: 10, and epoch number: 1)
```

## Result

```
Result Folder
├── PERT_result.xlsx
└── merge_result.xlsx (The data which need to be checked manually.)
```

## Reference
```
@inproceedings{huang2013learning,
  title={Learning deep structured semantic models for web search using clickthrough data},
  author={Huang, Po-Sen and He, Xiaodong and Gao, Jianfeng and Deng, Li and Acero, Alex and Heck, Larry},
  booktitle={Proceedings of the 22nd ACM international conference on Information \& Knowledge Management},
  pages={2333--2338},
  year={2013}
}
```
```
@article{DBLP:journals/corr/abs-2205-11737,
  author    = {Jinghui Xiao and
               Qun Liu and
               Xin Jiang and
               Yuanfeng Xiong and
               Haiteng Wu and
               Zhe Zhang},
  title     = {{PERT:} {A} New Solution to Pinyin to Character Conversion Task},
  journal   = {CoRR},
  volume    = {abs/2205.11737},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.11737},
  doi       = {10.48550/arXiv.2205.11737},
  eprinttype = {arXiv},
  eprint    = {2205.11737},
  timestamp = {Mon, 30 May 2022 15:47:29 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-11737.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
