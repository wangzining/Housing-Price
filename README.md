##House Price

Group Project by Wenxuan Wang, Wenda Zheng, Zining Wang

This project investigate how different features can affect house prices.

Usage
----------

* 2017.5 Version

Since we used Jupyter, we directly exported the files as a pdf file

To run:
0. install Python 2.7(may require environmental variable setups)
1. install annaconda that comes with python packages
2. Up to the version of annaconda you may need to update some packages, including sklearn(scikit-learn) and seaborn by doing:
	conda update scikit-learn
3. open jupyter inside of annaconda
4. open combined.ipynb
5. to run each block, select that block and do SHIFT + ENTER

Alternatively, we provide a regular file that has extension of .py. This could be run in terminal with
	python combined.py

* 2017.12 Version

We added xgboost+lasso(0.13013,baseline).ipynb file and generated test results in price_sol.csv. This final model gives us better prediction.