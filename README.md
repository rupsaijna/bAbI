# bAbI
Tsetlin Machine experiments on bAbI tasks

# Dataset
```
cd data
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -xvzf tasks_1-20_v1-2.tar.gz
```

# Task 1 : Single Supporting Fact
The file format for the task is as follows:
```
ID text
ID text
ID text
ID question[tab]answer[tab]supporting_fact ID.
```
Each sentence is provided with an ID. The IDs for a given “story” start at 1 and increase. When the IDs in a file reset back to 1 you can consider the following sentences as a new “story”. Supporting fact ID refer to the sentences within a “story”.
```
1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom        1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway         4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel?      hallway         4
```
Above example can be seen in **data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt**

## Requirements
Pandas 1.0.3
