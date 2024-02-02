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


|                         | Number | Percentage |
|-------------------------|--------|------------|
| **Number of Passengers** | 891    |    -        |
| **Number of Men**        | 577    |   -         |
| **Number of Women**      | 314    |      -      |
| **Deaths (All)**         | 549    |     -       |
| **Deaths (Men)**         | 468    |     -       |
| **Deaths (Women)**       | 81     |      -      |
| **Survived (All)**       | 342    |       -     |
| **Survived (Men)**       | 109    |       -     |
| **Survived (Women)**     | 233    |      -      |
| **Survival Rate (All)**  | -      | 38.38%     |
| **Overall Survival (Men)**| -     | 12.23%     |
| **Overall Survival (Women)**| -   | 26.15%     |
| **Survival among Men**   | -      | 18.89%     |
| **Survival among Women** | -      | 74.2%      |
| **Overall Deaths (Men)** | -      | 52.53%     |
| **Overall Deaths (Women)**| -    | 9.09%       |
| **Deaths among Men**     | -      | 81.11%     |
| **Deaths among Women**   | -      | 25.8%      |



#### Useful insights

From the analysis above the following points can be extracted (some of them are useful for feature engineering and some are just interesting and curious):

- More than half of the people died (~61%)
- Among men, 81 % died, 468 out of 549. **(This can be useful so as if 'Sex' is a man, 'Survival' is probably 0)**
- Among women, 74 % survived, 233 out of 314 **(This can be useful so as if 'Sex' is a woman, 'Survival' is probably 1)**