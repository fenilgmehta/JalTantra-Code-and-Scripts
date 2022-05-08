# Jaltantra-Code-and-Scripts

Code and Scripts written for Jaltantra Project

- [Shared Drive Folder](https://drive.google.com/drive/folders/1meWna4CxTDQjtfJEhkcGVMDxywxMoutH?usp=sharing)
    - [Time vs Objective Function Value](https://docs.google.com/spreadsheets/d/1TBwoyL2dkQaxP-lT6kpZ0-uVaws9hfPB/edit?usp=sharing&ouid=101940378676875078716&rtpof=true&sd=true)
- Programs/Scripts
    - [auto_run_model_and_data_in_parallel.py](auto_run_model_and_data_in_parallel.py) - Automatically execute solver
      for multiple model and data files while monitoring the system resources
    - [htop_monitor_writter.sh](htop_monitor_writter.sh) - Monitor system resource usage using htop and save them to a
      file at regular interval
        - [htop_monitor_reader.sh](htop_monitor_reader.sh) - Read the output generated by the above monitoring program
    - [output_table_extractor_baron.sh](output_table_extractor_baron.sh) - Extract important values from the output
      generated by Baron solver
    - [output_table_extractor_octeract.sh](output_table_extractor_octeract.sh) - Extract important values from the
      output generated by Octeract solver
    - [CalculateNetworkCost.py](CalculateNetworkCost.py) - Automatically execute multiple Solvers and multiple Models on
      input graph/network (i.e. data/testcase file) and return the best solution
        - `python3 CalculateNetworkCost.py -p Files/Data/m1_m2/d9_HG_SP_4_2.dat --solver-models 'baron 1 2' 'octeract 1 2' --time '0:5:0' --debug`
        - To clean up the temporary file created by this program in the `/tmp` directory, execute the below commands
          ```shell
          # Make sure that the below GLOB does not match any other important file or
          # directory created by some someone other than CalculateNetworkCost.py
          # And, do not perform this clean up when any instance of CalculateNetworkCost.py is running
          rm -r /tmp/pid_* /tmp/at*octsol /tmp/baron_tmp*
          ```
    - [CalculateNetworkCost_ExtractResultFromAmplOutput.py](CalculateNetworkCost_ExtractResultFromAmplOutput.py) -
      Extract the following values from stdout/stderr logs of AMPL + Solver
        1. head for each node
        2. flow for each arc/edge
        3. pipe ID and pipe length for each arc/edge

### Overview

| Original thing    | Similar to                              |
|-------------------|-----------------------------------------|
| AMPL              | Shell                                   |
| Solver            | Java bytecode interpreter               |
| Model file        | Algorithm / Java function               |
| Network/Data file | Testcases / Input / Function parameters |

- AMPL is like shell because we use it to define variables pointing to data file and model file, and then execute the
  solver on them. Similar to
  ```sh
  solver --data-file DATA_FILE_PATH --model-file MODEL_FILE_PATH --solver-parameter1 value1 --solver-parameter2 value2 ...
  ```
- Solver is like bytecode interpreter because multiple bytecode interpreter exists and they work in their own way (
  Oracle interpreter, OpenJDK interpreter, Kotlin interpreter)
- Model file is like an algorithm
- Network/Data file has Graph data

### AMPL Commands

- AMPL execution commands are present in [Files/main.run](Files/main.run)

### Model Files

- Present at [Files/Models](Files/Models)
- These are written in R language
- According to the [AMPL Dev User Guide](https://optirisk-systems.com/wp-content/uploads/2018/05/AMPLDevExtract.pdf),
  these files should have `.mod` file extension. However, we have used `.R` so that we get syntax highlighting in text
  editors like VSCode and SublimeText

| Unique Name | Original Name    |
|-------------|------------------|
| m1          | Basic            |
| m2          | Basic2 Basic2_v2 |
| m3          | Descrete Segment |
| m4          | Parallel Links   |

### Data Files

- For model m1 and m2, data files are in [Files/Data/m1_m2](Files/Data/m1_m2)
- For model m3 and m4, data files are in [Files/Data/m3_m4](Files/Data/m3_m4)
- According to the [AMPL Dev User Guide](https://optirisk-systems.com/wp-content/uploads/2018/05/AMPLDevExtract.pdf),
  these files should have `.dat` file extension. However, we have used `.R` so that we get syntax highlighting in text
  editors like VSCode and SublimeText

| Unique Name | Original Name |
| ----------- | ------------- |
| d1          | Two Loop      |
| d2          | Cycle Hanoi   | 
| d3          | Double Hanoi  |
| d4          | Triple Hanoi  |
| d5          | Taichung      |
| d6          | HG_SP_1_4     |
| d7          | HG_SP_2_3     |
| d8          | HG_SP_3_4     |
| d9          | HG_SP_4_2     |
| d10         | HG_SP_5_5     |
| d11         | HG_SP_6_3     |

