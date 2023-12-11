# E2USD Artifact Submission - WWW 2024
Welcome to the artifact documentation for our paper, **E2Usd: Efficient-yet-effective Unsupervised State Detection for Multivariate Time Series**, submitted to the research tracks of The Web Conference 2024. This documentation outlines the steps required to reproduce our work.


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

## MCU deployment models

### Prerequisites

Ensure you have the necessary prerequisites in place:

- Download STM32CubeIDE from the official [STMicroelectronics website](https://www.st.com/zh/development-tools/stm32cubeide.html) based on your operating system.

### Installation and Configuration

1. **STM32CubeIDE Installation:** Install STM32CubeIDE according to your operating system.

2. **Project Initialization:** Launch STM32CubeIDE and create a new STM32 Project named "STM32H747_E2USD."

3. **Board Selection:** Select the development board model as "STM32H747I-DISCO" and proceed.

4. **Software Package Setup:** Choose the "cue-ai" package to facilitate AI development. Under software packs, select the required components.

5. **AI Application Configuration:** Under the "X-CUBE-AI" section, specify the M7 series for application validation. Click OK to confirm the setup.

6. **Pin Configuration:** Navigate to the pin configuration page. Select "RCC" for default configuration and set the power regulator voltage to "0" in the configuration column.

7. **Software Packs Configuration:** In the pin configuration page, locate the "cube-ai" package configuration under software packs. Configure the serial port settings in the platform configuration.

8. **Network Setup:** Add the AI model by clicking "Add Network" and selecting the E2USD model file.

### Model Validation and Deployment

9. **Model Validation Initiation:** Commence the model validation process on your desktop.

10. **Clock Configuration:** After successful model validation, proceed with model deployment. Click on the clock configuration column to configure system clock settings automatically.

11. **Code Generation:** Within the IDE compiler, select the project and choose "Generate Code."

12. **Compilation:** Post code generation, right-click on "STM32H747_E2USD_CM7" and select "Build Project."

13. **Compilation Verification:** Allow time for the compilation process to complete. If there are no errors, move on. In case of a successful compilation, right-click on the project name and select "Run."

14. **CubeMX Configuration:** Access the "STM32H747_E2USD.ioc" file within STM32CubeIDE to return to the CubeMX configuration page.

15. **AI Tool Configuration:** Under software packs, configure the validation input and validation output in the Cube.AI tool's network settings. Select the dataset file and validate it on the target board.

16. **Model Deployment:** Run the AI model on the development board, and obtain a comprehensive evaluation report.

17. Congratulations, your AI model is now successfully deployed and operational on the STM32H747I-DISCO development board.



