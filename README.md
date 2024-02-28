# E2USD Artifact - WWW 2024
Welcome to the artifact documentation for our paper, **E2Usd: Efficient-yet-effective Unsupervised State Detection for Multivariate Time Series**, accepted by the research tracks of The Web Conference 2024. This documentation outlines the steps required to reproduce our work.


## Appendix
For a more comprehensive understanding of our work, including detailed dataset descriptions, implementation specifics, an impact assessment of the Energy-based Frequency Compressor, additional NMI results for component study, and parameter sensitivity study, please refer to the [Appendix](Appendix.pdf). We recommend downloading the PDF to a local viewer for better readability.

## Hardware information
Our experiments were conducted on a server equipped with an NVIDIA Quadro RTX 8000 GPU and an Intel Xeon Gold 5215 CPU (2.50GHz). For MCU deployment, we employed an STM32H747 device with a 480 MHz Arm Cortex-M7 core, 2 MB Flash memory, and 1 MB RAM.

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

## Acknowledgement

This code is based on the implementation of [Time2State](https://github.com/Lab-ANT/Time2State)
