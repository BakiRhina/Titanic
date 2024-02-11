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

### Age and survival analysis

In this analysis ages are separated in different classes. Since Titanic sank in 1912, age ranges are designed according to the same period:

- Infant: [0, 2] years old
- Child: (2, 12] years old
- Adolescent: (12, 18] years old
- Young: (18, 25] years old
- Middle-aged: (25, 40] years old
- Adult: (40, 60] years old 
- Elderly: (60+) years old



The following chart show the number of people in each class, the number of people who survived (in light green) and the number who didn't (in red). The percentage shows the survability of each class.

![image](https://github.com/BakiRhina/Titanic/assets/108484177/a5953e0a-e20e-43e2-bcf2-250771eaf42a)

**Important observation**: Most of the people falls into the cathegory of Middle Aged Young (25 to 35 y.o.). This number has been affected when dealing with Missing Values due to using the mean to substitute them. In this case, the mean is 29. Therefore, using the mean value to deal with NaN, added **177** people to the Middle Aged Young class. This observation is crucial, so as it means that without this 177 persons (~50%), the percentage of survival in that cathegory would change completely.

A future approach, in this context, could be equally distributing the ages over all cathegories, therefore not benefiting any class in particular.


#### Useful insights


- Most people is between 20 and 40 years old
- Elder people had less chance of survival while infants survived the most (in %, not number)
- Although in some cases it is possible to predict if the person survived or not with enough probability by just looking at the age (**elder** 23% of survival, 77% death or **Infant** 62% survival), other age ranges are more difficult to allocate with confidence.


### Data balance check

![image](https://github.com/BakiRhina/Titanic/assets/108484177/0610eca0-5a86-4a35-9715-817ef15cdcf7)


From the analysis above, it is clear that the data is not balanced (61.62% (0), 38.38% (1)). Since it is not too imbalanced, we will proceed with data preprocessing and supervised models, and after some results we will implement data balancing techniques such as data augmentation, SMOTE, among others.
