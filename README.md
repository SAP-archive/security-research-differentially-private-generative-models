<!--
SPDX-FileCopyrightText: 2020 SAP SE

SPDX-License-Identifier: Apache-2.0
-->

# Differentially Private Generative Models

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-differentially-private-generative-models)](https://api.reuse.software/info/github.com/SAP-samples/security-research-differentially-private-generative-models)

## Description:
This repository explains how generative models can be used in combination with differential privacy to synthetize feature-rich realistic categorical datasets in a privacy preserving manner. It brings two jupyter notebooks for dp-GANs (differentially-private Generative Adversarial Networks) and dp-VAE (Variational Autoencoder) to generate new data in a differetnial private mode. The code allows to quickly generate new dataset (incl. numerical features) in private or public mode. dp_SGD and dp_Adam optimizers from tensowflow/ privacy library (https://github.com/tensorflow/privacy) are used these models. 

## Requirements
- [Python](https://www.python.org/)
- [Jupyter](https://jupyter.org/)
- [Tensorflow](https://github.com/tensorflow)
- Pandas, keras, and more see the notebooks import sections
- [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- Check further dependencies in the Jupyter notebooks Tutorial_dp-GAN.ipynb and Tutorial_dp-VAE.ipynb

## Download the tensorflow privacy project
1. Clone Tensorflow privacy into this project repository :
```
git clone https://github.com/tensorflow/privacy

cd privacy
pip install -e .
```


2. Open the notebooks in Jupyter and run them


## Authors / Contributors

 - Lyudmylla Dymytrova
 - Lorenzo Frigerio
 - Anderson Santana de Oliveira
 
## Known Issues
No issues known


## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.


## Citations
If you use this code in your research,
please cite:

```
@article{DBLP:journals/corr/abs-1901-02477,
  author    = {Lorenzo Frigerio and
               Anderson Santana de Oliveira and
               Laurent Gomez and
               Patrick Duverger},
  title     = {Differentially Private Generative Adversarial Networks for Time Series,
               Continuous, and Discrete Open Data},
  journal   = {CoRR},
  volume    = {abs/1901.02477},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.02477},
  archivePrefix = {arXiv},
  eprint    = {1901.02477},
  timestamp = {Fri, 01 Feb 2019 13:39:59 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-02477},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## References
- [1] Lorenzo Frigerio, Anderson Santana de Oliveira, Laurent Gomez, Patrick Duverger:
Differentially Private Generative Adversarial Networks for Time Series, Continuous, and Discrete Open Data. CoRR abs/1901.02477 (2019). https://arxiv.org/abs/1901.02477


## License
Copyright (c) 2020 SAP SE or an SAP affiliate company. All rights reserved.
This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
