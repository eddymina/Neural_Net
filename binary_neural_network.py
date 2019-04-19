from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
    
class MLP:

    def __init__(self, input_size, hidden_dims=32, alpha=.2):
        '''
        Instantiate MLP using given parameters. 
        
        You may assume that there is only a single hidden layer
        (i.e., you need not generalize to handle arbitrary numbers of
        hidden layers).
        \alpha is the learning rate.
        '''
        self.hidden_dims= hidden_dims 
        print("There are {} hidden dimensions".format(self.hidden_dims))
        self.w1 = np.random.rand(input_size,self.hidden_dims) #w1 = [input_size x hidden_dims]
        self.w2 = np.random.rand(self.hidden_dims,1) #w2 = [hidden_dims x 1]
        self.bias1= np.random.rand(1,self.hidden_dims) 
        self.bias2= np.random.rand(1,1) 
        self.learn_rate= alpha
        print ("\nInitialized Weight Shapes:: w2=",self.w2.shape, "w1=", self.w1.shape )
        print ("\nInitialized Bias Shapes:: b2=",self.bias2.shape, "b1=", self.bias1.shape )
        print (" ")
        

    def sigmoid(self,x):
        """
        Params:
        ---

        x: nxd numpy array 

        Output:
        ---
        Result of Sigmoid "Squishing" Function: Array of vals in rang {0,1}
        """
        return 1/(1 + np.exp(-x))
    
        
    def log_loss(self,y,prediction,eps=1e-15):
        """
        Logistic Regression Loss Function Defined 

        Params:
        ---

        y= actual class (nx1 array)
        prediction= predicted class (nx1 array)

        Returns:
        Average of the loss for a entire set (epoch of data)
        """
        #return -1.0*(y*np.log(prediction)+(1-y)*np.log(1-prediction))
        prediction= np.clip(prediction, eps, 1 - eps)
        if y == 1:
            return -np.log(prediction)
        else:
            return -np.log(1 - prediction)
    
    def predict(self,X):
        """
        Compute the y_hat given a set of weights 
        """
        #output = y_hat 
        self.hidden_layer = self.sigmoid(np.dot(X,self.w1)+self.bias1) # hidden layer= sig(xW_1)
        # reshape self.hidden_layer ?
        y_hat= self.sigmoid(np.dot(self.hidden_layer,self.w2)+self.bias2) #sig(sig(xW_1)*W_2)
        return y_hat

    def loss_plot(self,train_loss):
        """
        Show Log Loss Plot 
        """
        plt.figure(1)
        plt.plot(range(len(train_loss)),train_loss,color='g')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Loss vs. Epoch")
        plt.draw()
        
    def accuracy_plot(self,train_accuracy,test_accuracy,train_color='g',test_color='b'):
        """
        Show Acc Loss Plot 
        """
        plt.plot(range(len(test_accuracy)),train_accuracy,color=train_color,label='Train Acc w/ {} hidden dims'.format(self.hidden_dims))
        plt.plot(range(len(test_accuracy)),test_accuracy,color=test_color,label='Test Acc w/ {} hidden dims'.format(self.hidden_dims))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Acc vs. Epoch")
        plt.legend()
        plt.draw()
        
    def convert_to_class(self,a):
        """
        Convert probabilities to class:: if p < .5, class 0, else class 1
        """
        return np.array([int(np.around(x)) for x in a])

        
    def fit(self, X, y, epochs=100,random_state=42,loss_plot=False,verbose=False):
        '''
        Train the model via backprop for the specified number of epochs.
        '''

        print ("The shape of the data is: ",X.shape)
        print ("\n------------------------------")
        #Split the data 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=random_state)
        
        #store accuracy of train and test vals 
        training_loss=[]
        epoch_train_accuracy=[]
        self.epoch_train_accuracy=epoch_train_accuracy
        epoch_test_accuracy=[]
        self.epoch_test_accuracy=epoch_test_accuracy

        #for epoch 
        for epoch in range(epochs): #loop through all training data
            loss= []
            epoch_y_train=[]
            epoch_y_test=[]
            
            #iterate through each element in the batch
            for i,x in enumerate(X_train): 
                prediction= self.predict(x) #sig(weights*x) gives the class probaility 
                loss.append(self.log_loss(y_train[i],prediction)) #get the loss for row 
                
                #back prop 
                dw_2= (y_train[i]-prediction)*self.hidden_layer.T
                dw_1=np.dot(np.expand_dims(x,axis=1),((y_train[i]-prediction)*self.w2.T*self.hidden_layer*(1-self.hidden_layer)))
                db_2= y_train[i]-prediction
                db_1= ((y_train[i]-prediction)*self.w2.T*self.hidden_layer*(1-self.hidden_layer))
                
                #update weights (SGD)
                self.w1 += self.learn_rate*(dw_1/len(y_train))
                self.w2 += self.learn_rate*(dw_2/len(y_train))
                self.bias1 += self.learn_rate*(db_1/len(y_train))
                self.bias2 += self.learn_rate*(db_2/len(y_train))
       
                epoch_y_train.append(prediction) #get all the predictions for each item in training batch
        
            epoch_y_test= self.predict(X_test) #get all the predictions for each item in training batch
                
            self.epoch_train_accuracy.append(accuracy_score(y_train,self.convert_to_class(epoch_y_train)))
            self.epoch_test_accuracy.append(accuracy_score(y_test,self.convert_to_class(epoch_y_test)))
            
            training_loss.append(np.average(loss))
            if verbose == True:
                if epoch % 20 == 0:
                    print ("\n For Epoch:", epoch, "With Training Loss:", np.around(np.average(loss),decimals=6))
                    pass 
        
        if loss_plot == True:
            self.loss_plot(training_loss)
            pass 

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.metrics import classification_report

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,n_redundant = 2,
                               n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
    mlp=MLP(X.shape[1],32,.2)
    mlp.fit(X, y, epochs=100, loss_plot=False, verbose=True)
    print("\nResults from Numpy Built Classifier\n")
    y_preds= mlp.predict(X_test)
    y_preds=mlp.convert_to_class(y_preds)
    print("\nTrain Model Accuracy:",accuracy_score(y_train,mlp.convert_to_class(mlp.predict(X_train))))
    print("\nTest Model Accuracy:",accuracy_score(y_test,y_preds))
    print(classification_report(y_true=y_test, y_pred=y_preds))
    mlp.accuracy_plot(mlp.epoch_train_accuracy,mlp.epoch_test_accuracy)
    plt.show()


        