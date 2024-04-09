# E2USD Artifact - WWW 2024
Welcome to the artifact documentation for our paper, **E2Usd: Efficient-yet-effective Unsupervised State Detection for Multivariate Time Series**, accepted by the research tracks of The Web Conference 2024. This documentation outlines the steps required to reproduce our work.

## Library Dependencies
To run our code, we developed it using Python 3.9.18. You can install the remaining dependencies by executing the following command:
```bash
pip install -r requirements.txt
```
## Dataset Preparation
We evaluated E2USD on five real-world Unsupervised State Detection (USD) datasets and one synthetic dataset for comprehensive assessments. You can download these datasets from the following links:


| Dataset   | Type               | Download Link |
|----------|----------|--------------------|
| MoCap   | Real-world | [download](https://drive.google.com/file/d/1Z3HRSxUUfjiPRMzGrOcGie63S1HXA8nf/view?usp=sharing) |
| ActRecTut| Real-world | [download](https://drive.google.com/file/d/1tU5EmxRUk37TzgvpkcgTMQSVG8DBGCUt/view?usp=sharing) |
| PAMAP2| Real-world | [download](https://drive.google.com/file/d/11zwi7PwJiRujncT7kt0NOGOo_GavSSo2/view?usp=sharing) |
| UscHad| Real-world | [download](https://drive.google.com/file/d/1kBHPZZCCN1zrZd7CoSGzG3_W0Jdsm9kF/view?usp=sharing) |
| UcrSeg| Real-world | [download](https://drive.google.com/file/d/1nGH-l3tkp18SauzUUR6P0FhlhEQDLTu2/view?usp=sharing) |
| Synthetic | Synthetic | [download](https://drive.google.com/file/d/1C6Pl58O-un4DUPdzqC9PKs09wQi8knYx/view?usp=sharing) |

After downloading the datasets, move them to the '\data' directory, ensuring the following directory structure:


```
.
├── data
│   ├── synthetic
│   │   ├── test0.csv
│   │   ├── test1.csv
│   │   ├── ...
│   ├── ActRecTut
│   │   ├── subject1_walk
│   │   │   ├── S111.dat
│   │   │   ├── ...
│   │   ├── subject2_walk
│   │   │   ├── S111.dat
│   │   │   ├── ...
│   ├── MoCap
│   │   ├── 4d
│   │   │   ├── amc_86_01.4d
│   │   │   ├── ...
│   │   ├── raw
│   │   │   ├── amc_86_01.txt
│   │   │   ├── ...
│   ├── PAMAP2
│   │   ├── Protocol
│   │   │   ├── subject101.dat
│   │   │   ├── ...
│   ├── USC-HAD
│   │   ├── Subject1
│   │   ├── Subject2
│   │   ├── ...
│   ├── UCRSEG
│   │   ├── Cane_100_2345.txt
│   │   ├── DutchFactory_24_2184.txt
│   │   ├── ...

```
## Training command

Execute the train.py script located in the ./experiments directory. Make sure to specify the dataset you want to use for the experiment within the main function.

## Supplementary Study Between E2Usd and FLOSS

We provide a supplementary study between E2Usd and FLOSS, please refer to the [Supplementary Study Between E2Usd and FLOSS](Supplementary_Study_Between_E2Usd_and_FLOSS.pdf). We recommend downloading the PDF to a local viewer for better readability.


## Acknowledgements

This work leverages the [Time2State](https://github.com/Lab-ANT/Time2State) implementation as its foundation.

Our gratitude extends to the authors of the following studies for making their datasets publicly available:
- [UCRSEG](https://doi.org/10.1109/ICDM.2017.21)
- [Synthetic](https://dl.acm.org/doi/abs/10.1145/3588697)
- [MoCap](https://dl.acm.org/doi/abs/10.1145/2588555.2588556)
- [ActRecTut](https://dl.acm.org/doi/abs/10.1145/2499621)
- [PAMAP2](https://doi.org/10.1109/ISWC.2012.13)
- [USCHAD](https://doi.org/10.1145/2370216.2370438)

Special thanks are also due to [Prof. Eamonn Keogh](https://www.cs.ucr.edu/~eamonn/) for his valuable feedback on our work.
