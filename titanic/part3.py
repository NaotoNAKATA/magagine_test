# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

if __name__ == '__main__':
	# バージョン確認
	print(pd.__name__, pd.__version__)
	print(np.__name__, np.__version__)
	print(matplotlib.__name__, matplotlib.__version__)
	print(sklearn.__name__, sklearn.__version__)

	# データセット読み込み
	train_set = pd.read_csv('data/train.csv')
	test_set = pd.read_csv('data/test.csv')

#	# 読み込み確認
#	train_set.head(2)
#	test_set.head(2)

	# Fontの設定
#	font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
#	font_prop = FontProperties(fname=font_path)

	#
	# 探索データ解析
	#

	# PClass(旅客等級)
#	fig1 = plt.figure(figsize=(12,4))
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	plot1 = train_set['Survived'].groupby(train_set['Pclass']).mean()
	ax1.bar(x=plot1.index, height=plot1.values)
	ax1.set_ylabel('Survival Rate')
	ax1.set_xlabel('PClass')
	ax1.set_xticks(plot1.index)
	ax1.set_yticks(np.arange(0, 1.1, .1))
	ax1.set_title('PClass and Survival Rate')
	fig1.savefig('01_PClass.png')

	# Sex(性別)
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	plot2 = train_set['Survived'].groupby(train_set['Sex']).mean()
	ax2.bar(x=plot2.index, height=plot2.values)
	ax2.set_ylabel('Survival Rate')
	ax2.set_xlabel('Gender')
	ax2.set_xticks(plot2.index)
	ax2.set_yticks(np.arange(0, 1.1, .1))
	ax2.set_title('Gender and Survival Rate')
	fig2.savefig('02_Sex.png')

	# SibSp(同乗中の兄弟/配偶者の数)
	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111)
	plot3 = train_set['Survived'].groupby(train_set['SibSp']).mean()
	ax3.bar(x=plot3.index, height=plot3.values)
	ax3.set_ylabel('Survival Rate')
	ax3.set_xlabel('Total Siblings')
	ax3.set_xticks(plot3.index)
	ax3.set_yticks(np.arange(0, 1.1, .1))
	ax3.set_title('Total Siblings and Survival Rate')
	fig3.savefig('03_Siblings.png')

	# Parch(同乗中の親/子供の数)
	fig4 = plt.figure()
	ax4 = fig4.add_subplot(111)
	plot4 = train_set['Survived'].groupby(train_set['Parch']).mean()
	ax4.bar(x=plot4.index, height=plot4.values, color='Teal')
	ax4.set_ylabel('Survival Rate')
	ax4.set_xlabel('Number of Parents and Children abroad')
	ax4.set_xticks(plot4.index)
	ax4.set_yticks(np.arange(0, 1.1, .1))
	ax4.set_title('Number of Parents and CHildren aborad and Survival Rate')
	fig4.savefig('04_Parch.png')



