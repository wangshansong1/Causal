This project was part of the work during a visit to the University of South Australia.

Project name: Causal  
env:  
	python==3.7  
	torch==1.2.0+cpu  
	matplotlib==2.2.2  
	bartpy.egg==info  
	pandas==0.24.2  
	bartpy==0.0.2  
	numpy==1.17.4  
	scikit_learn==0.22  
software:  
	JetBrains PyCharm Community Edition 2019.2.3 x64  

Directory Structure:  
	  
Causal  
|  
|--|requirements.txt  
|--|readme.md  
|--|Experimnets.pdf  
|--|dataset  
|--|--|criteo  
|--|--|--|test  
|--|--|--|train  
|--|--|--|D0.txt  
|--|--|--|D1.txt  
|--|--|file20   
|--|--|gerber_huber_2014_data  
|--|--|ihdp  
|--|--|job  
|--|--|MineThatData  
|--|--|twin  
|  
|--|src  
|--|--|weight_w0  
|--|--|weight_w1  
|--|--|*.py  
  
Note:
1.All raw data are divided into test and training sets according to the proportion of 20% and 80%.In the test and training sets, the data set is divided into "control" group and "treatment" group.
2.In these data sets, some of the data sets are real data, and some of them are only features.So we need to generate the data according to the DGP process in "Transfer Learning for Estimating Causal Effects using Neural Networks". In the src/  folder，Files ending with "_data" are used in the DGP process. Each data set has a corresponding "XX_data.py". Because some data sets are real data, these data sets may not use the corresponding "XX_data.py" To generate data, but I still prepared him.
