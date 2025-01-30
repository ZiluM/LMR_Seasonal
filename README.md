#  Last Millennium Reanalysis Seasonal: "Coupled Seasonal Data Assimilation of Sea Ice, Ocean, and Atmospheric Dynamics over the Last Millennium"

Paper: [**<mark>Arxiv</mark>**](https://arxiv.org/abs/2501.14130)

## Authors

[Zilu Meng](https://zilum.github.io); Gregory J. Hakim; Eric J. Steig


## Introduction

Paleo data assimilation is a powerful tool to reconstruct past climate fields. Before the instrumental era, the climate system was not well observed. And the instrumental data is not long enough and strongly forced by human activities. This makes it difficult to study the earth climate variability. However, there are many paleoclimate proxies that can represent the past climate variability, like tree rings, ice cores, and corals. By combining these proxies with climate models, we can reconstruct the past climate fields, like temperature, precipitation, and wind fields and study the past climate variability like ENSO, PDO and AMO. 

This repo is the code for the first seasonal reanalysis dataset `LMR Seasonal` over the last millennium using "cycling" data assimilation. The reanalysis dataset will provide a gridded climate field for the last millennium, which can be used to study the past climate variability and the climate change. 

<!-- For example, the lower figure shows the Nino3.4 Index (a measure of El Niño and Southern Oscillation) from the our reanalysis comparing with the HadISST dataset. The reanalysis dataset can provide a accurate ENSO variability for the last millennium. In this case, we can study the ENSO variability in the past and compare it with the present. Additionally, the reanalysis dataset can also be used to study the climate change in the past and compare it with the present. -->
 
<table>
  <tr>
    <td align="center">
      <img src="./figures/dacycle.png" alt="intro" width="50%">
    </td>
  </tr>
</table>



<!-- ![intro](./figures/Nino34_compare_HadISST.png) -->

## Code Structure

[./DA]: The main code for the data assimilation. Following `sub.sh`  and `*.yml` to run the data assimilation. 

[./LIM]: The code for the linear inverse model. Following `main_lim.py` and `*.yml` to train the linear inverse model.

[./OBS]: The code for the observation operator (Proxy System Model). Following `~.ipynb` to calibrate the Proxy System Model.

[./utils]: The code for utilites.

[./slim]: The code for utilites.

[./OBS]: The code for the observation operator (Proxy System Model).


## Quick Start

### 0. clone the repo

```bash
git clone git@github.com:ZiluM/LMR_Seasonal.git 
```

### 1. Install the required packages

Following the file `requirements.txt` to install the required packages. Attention that the `slim` package is not available in the PyPI, you need to install it manually from the `./slim` folder.

### 2. Calibrate the Proxy System Model

Following the `./OBS/~.ipynb` to calibrate the Proxy System Model.

### 3. Train the Linear Inverse Model

Use the `./LIM/main_lim.py` to train the linear inverse model.

### 4. Data Assimilation

Use the `./DA/da.py` to run the data assimilation.


## Citation

If you use the code in your research, please cite the paper:

```
@article{meng2025coupled,
  title={Coupled Seasonal Data Assimilation of Sea Ice, Ocean, and Atmospheric Dynamics over the Last Millennium},
  author={Meng, Zilu and Hakim, Gregory J. and Steig, Eric J.},
  journal={arXiv preprint arXiv:2501.14130},
  year={2025}
}

@article{meng2024reconstructing,
  title={Reconstructing the tropical Pacific upper ocean using online data assimilation with a deep learning model},
  author={Meng, Zilu and Hakim, Gregory J},
  journal={Journal of Advances in Modeling Earth Systems},
  volume={16},
  number={11},
  pages={e2024MS004422},
  year={2024},
  publisher={Wiley Online Library}
}
```








