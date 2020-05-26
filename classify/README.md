>python bow_features.py ../generated/generated1.txt

>python bow_classifyTM.py ../generated/generated1.txt 

>python bow_classifyTM_posonly.py ../generated/generated1.txt 

>python process_clauses.py ../generated/generated1.txt   ##uses _posonly output by default



###############

process_clauses requires package Sympy

```python

git clone https://github.com/sympy/sympy.git
cd sympy
git pull origin master
python setupegg.py develop

```

