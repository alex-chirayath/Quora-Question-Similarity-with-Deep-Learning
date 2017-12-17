from scipy import stats

#single, dual 25, dual 100, multinomial, logregression 
k=stats.kruskal([1-0.8662,1-0.8593,1-0.8578,1-0.8547],[1-0.8522,1-0.8537,1-0.8423,1-0.8458],[1-0.8980,1-0.8982,1-0.8921,1-0.8983],[1-0.7409,1-0.7409,1-0.7494,1-0.7554],[1-0.7525,1-0.753,1-0.7576,1-0.7661],[1-0.8280,1-0.8286,1-0.8332,1-0.8204],[1-0.7788,1-0.7839,1-0.7791,1-0.7816],[1-0.8361,1-0.8321,1-0.8260,1-0.8124],[1-0.7832,1-0.7791,1-0.7832,1-0.7729],[1-0.7756,1-0.7755,1-0.7784,1-0.7753],[1-0.7739,1-0.7794,1-0.7788,1-0.7725])
print k
if(k[1]<0.05):
	print("we reject the null that the mean error of models are the same and thus there does exist some model with mean error not the same as another")
else:
	print("\n we fail to reject the null that the mean error of models are the same\n")

#dual with 25 epochs and bilstm
t=stats.ttest_rel([0.8522,0.8537,0.8423,0.8458],[0.8280,0.8286,0.8332,0.8204])
print t
if(t[1]*2<0.05/11):
	print("we reject the null that the mean accuracy of dual with 25 epochs and blinear single siamese lstm are the same")
else:
	print("\n we fail to reject the null that the mean accuracy of dual with 25 epochs and blinear single siamese lstm are the same\n")
 
t=stats.ttest_rel([0.8361,0.8321,0.8260,0.8124],[0.7832,0.7791,0.7832,0.7729])
print t
if(t[1]*2<0.05/11):
	print("we reject the null that the mean accuracy of single siamese and LSTM with 1 hidden layer on reel (50 epochs on 200k data each) are the same")
else:
	print("\n we fail to reject the null that the mean accuracy of single siamese and LSTM with 1 hidden layer on reel (50 epochs on 200k data each) are the same\n")

t=stats.ttest_rel([0.8522,0.8537,0.8423,0.8458],[0.7409,0.7409,0.7494,0.7554])
print t
if(t[1]*2<0.05/11):
	print("we reject the null that the mean accuracy of dual siamese 25 and multinomial are the same")
else:
	print("\n we fail to reject the null that the mean accuracy of dual siamese 25 and multinomial are the same\n")
 


 
