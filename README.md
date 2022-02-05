# Jaltantra-Code-and-Scripts

Code and Scripts written for Jaltantra Project

### Overview

| Original thing | Similar to                              |
| -------------- | --------------------------------------- |
| Data file      | Testcases / Input / Function parameters |
| Model file     | Algorithm / Java function               | 
| Solver         | Java bytecode interpreter               |
| AMPL           | Shell                                   |

- Data file has Graph data
- Model file is like an algorithm
- Solver is like bytecode interpreter because multiple bytecode interpreter exists and they work in their own way (Oracle interpreter, OpenJDK interpreter, Kotlin interpreter)
- AMPL is like shell because we use it to define variables pointing to data file and model file, and then execute the solver on them. Similar to 
	```sh
	solver --data-file DATA_FILE_PATH --model-file MODEL_FILE_PATH --solver-parameter1 value1 --solver-parameter2 value2 ...
	```
- AMPL execution commands are present in [Files/main.run](Files/main.run)

### Model Files

- Present at [Files/Models](Files/Models)
- These are written in R language

| Unique Name | Original Name    |
|-------------|------------------|
| m1          | Basic            |
| m2          | Basic2 Basic2_v2 |
| m3          | Descrete Segment |
| m4          | Parallel Links   |

### Data Files

- For model m1 and m2, data files are in [Files/Data/m1_m2](Files/Data/m1_m2)
- For model m3 and m4, data files are in [Files/Data/m3_m4](Files/Data/m3_m4)

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

