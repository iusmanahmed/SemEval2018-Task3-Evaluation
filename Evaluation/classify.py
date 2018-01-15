import pandas
import numpy
from sklearn.externals import joblib
model = joblib.load("class.joblib.pkl")
data_file=pandas.read_csv('feature.csv')
X =numpy.array(data_file.iloc[:, 1:11])
r2= model.predict(X)
f=open("predictions-taskA.txt","a")
for u in range (0,len(r2)):
	f.write(str(r2[u]))
	f.write("\n")
f.close()

print "done"