# Titanic
Predictions with a simple dataset using Machine Learning. This project aims to use basic ML and DL tools to be familiarized with. Moreover, in this README file I will upload notes and useful information regarding issues and solutions.


## Experimental flow

1. Data exploration
2. Data preprocessing
3. Models developement
4. Training
5. Evaluation
6. Testing
7. Results
8. Discussion


## 1. Data exploration

In this section the training dataset will be explored and analysed. The main point is looking for insights that can be useful for feature engineering, although interesting and curious facts are always welcomed

### "Sex" and "Survival" relationship analysis

Number of passangers: `891`

Number of men: `577`

Number of women: `314`

 
Deaths ALL: `549`

Deaths MEN: ``468``

Deaths WOMAN: ``81``

 
Survived ALL: ``342``

Survived MEN: ``109``

Survived WOMAN: ``233``

 
Survival: ``38.38 %``

 
Overall survival MEN: ``12.23 %``

Overall survival WOMEN: ``26.15 %``

 
Survival among MEN: ``18.89 %``

Survival among WOMEN: ``74.2 %``

 
Overall deaths MEN:``52.53 %``

Overall deaths WOMEN: ``9.09 %``

 
Deaths among MEN: ``81.11 %``

Deaths among WOMEN: ``25.8 %``


#### Useful insights

From the analysis above the following points can be extracted (some of them are useful for feature engineering and some are just interesting and curious):

- More than half of the people died (~61%)
- Among the men, 81 % died, 468 out of 549. **(This can be useful so as if 'Sex' is a man, 'Survival' is probably 0)**
- Among women, 74 % survived, 233 out of 314 **(This can be useful so as if 'Sex' is a woman, 'Survival' is probably 1)**