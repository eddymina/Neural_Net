import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd 
np.seterr(divide='ignore', invalid='ignore') #disregard initial confusion matrix

class MMLP:

    def __init__(self, input_size, output_size, hidden_dims=32, alpha=.2):
        '''
        Instantiate MLP using given parameters. 
        
        You may assume that there is only a single hidden layer
        (i.e., you need not generalize to handle arbitrary numbers of
        hidden layers).
        
        \alpha is the learning rate.
        '''
        assert(output_size > 1)
        self.hidden_dims= hidden_dims 
        print("There are {} hidden dimensions".format(self.hidden_dims))
        self.w1 = np.random.rand(input_size,self.hidden_dims) #w1 = [input_size x hidden_dims]
        self.w2 = np.random.rand(self.hidden_dims,output_size) #w2 = [hidden_dims x 1]
        self.bias1= np.random.rand(1,self.hidden_dims) 
        self.bias2= np.random.rand(1,output_size) 
        self.learn_rate= alpha
        print ("\nInitialized Weight Shapes:: w2=",self.w2.shape, "w1=", self.w1.shape )
        print ("\nInitialized Bias Shapes:: b2=",self.bias2.shape, "b1=", self.bias1.shape )
        print (" ")
        
        pass 

    def CrossEntropy(self, yHat, y,eps=1e-15):
        """
        Simplified Cross Entropy w.r. to each class 
        yHat is clipped for stability 
        """

        yHat= np.clip(yHat, eps, 1 - eps)
        return -np.sum(y * np.log(yHat))

    def sigmoid(self,x):
        """
        Params:
        ---

        x: nxd numpy array 

        Output:
        ---
        Result of Sigmoid "Squishing" Function: Array of vals in range {0,1}
        """
        return 1/(1 + np.exp(-x))
    
    def softmax(self,X):
        """
        Stabilized softmax function
        
        """
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
    
    
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
        
    def epoch_plot(self,train_results,test_results,result_topic, train_color='g',test_color='b'):
        """
        Show result_topic vs Loss Plot 
        result_topic= f1 score, precision, recall, etc. 
        """
        plt.plot(range(len(train_results)),train_results,color=train_color,
                       label='Train {} w/ {} hidden dims'.format(result_topic,self.hidden_dims))
        plt.plot(range(len(test_results)),test_results,color=test_color,
                 label='Test {} w/ {} hidden dims'.format(result_topic, self.hidden_dims))
        plt.xlabel('Epoch')
        plt.ylabel(str(result_topic))
        plt.title("{} vs. Epoch".format(result_topic))
        plt.legend()
        plt.draw()
    
    
    
    def one_hot(self,obj_class,num_classes=5):
        """
        obj_class:: current class of the object
        num_classes:: number of total classes 
        class --> encode class
        1 --> [0 1 0 0]
        
        """
        enc=np.zeros(num_classes)
        enc[obj_class]= 1
        return enc

    def convert_to_class(self,y_enc):
        """
        Convert to index encoding to class 
        [0 1 0 0] --> 1
        """
        return np.array([np.argmax((enc),axis=0) for enc in y_enc])
    

    def predict(self, x):
        '''
        Yield predictions \hat{y} for instances X. To match the data generation 
        process, return the index of the highest performing class.
        '''
        
        self.hidden_layer = self.sigmoid(np.dot(x,self.w1)+self.bias1) # hidden layer= sig(xW_1)
        y_hat= self.softmax(np.dot(self.hidden_layer,self.w2) + self.bias2) #sig(sig(xW_1)*W_2)
        return y_hat
    
    def confusion_matrix(self, actual, prediction):    
        
        """
        Calculate the confusion matrix; labels are numpy array of classification labels
        
        """
        num_classes= len(set(actual))
        cm = np.zeros((num_classes, num_classes))
        for a, p in zip(actual, prediction):
            cm[a][p] += 1
        return cm

    def precision(self,confusion_matrix,average = None):
        
        """
        From confusion matrix, determine precision. 
        If average is selected choose option for micro or macro 
        """

        if average== 'micro':
            return np.sum(np.diag(confusion_matrix)) / np.sum(np.sum(confusion_matrix, axis = 0))
        elif average== 'macro':
            return np.average(np.diag(confusion_matrix)/ np.sum(confusion_matrix, axis = 0))
        pass 
        return np.diag(confusion_matrix)/ np.sum(confusion_matrix, axis = 0)

    def recall(self,confusion_matrix,average = None):
        
        """
        From confusion matrix, determine recall. 
        If average is selected choose option for micro or macro 
        """
        
        if average== 'micro':
            return np.sum(np.diag(confusion_matrix)) / np.sum(np.sum(confusion_matrix, axis = 1))
        elif average== 'macro':
            return np.average(np.diag(confusion_matrix)/ np.sum(confusion_matrix, axis = 1))
        else: 
            return np.diag(confusion_matrix)/ np.sum(confusion_matrix, axis = 1)


    def f1(self,confusion_matrix,average = None):
        """
        From confusion matrix, determine f1 score. 
        If average is selected choose option for micro or macro 
        """

        if average== 'micro':
            p  = self.precision(confusion_matrix,average='micro')
            r = self.recall(confusion_matrix,average='micro')
        elif average== 'macro':
            p  = self.precision(confusion_matrix)
            r = self.recall(confusion_matrix)
            return np.average(np.nan_to_num( 2 * p * r / (p + r)))
        else:  
            p  = self.precision(confusion_matrix)
            r = self.recall(confusion_matrix)

        return np.nan_to_num(2 * p * r / (p + r))

    def create_report(self, actual, prediction):
        """
        From actual and predicition values, create sklearn equivalent  
        of classification_report 
        """
        
        cm = self.confusion_matrix(actual, prediction)
        print ("Confusion Matrix:\n------------------\n",cm,"\n")
        import pandas as pd
        ind = np.arange(len(set(actual))).tolist()
        ind.extend(['','micro avg','macro avg'])
        p_list= self.precision(cm).tolist()
        p_list.extend(['',self.precision(cm,'micro'),self.precision(cm,'macro')])
        r_list=self.recall(cm).tolist()
        r_list.extend(['',self.recall(cm,'micro'),self.recall(cm,'macro')])
        f_list= self.f1(cm).tolist()
        f_list.extend(['',self.f1(cm,'micro'),self.f1(cm,'macro')])
        
        print("Report:\n------------------\n")

        print(pd.DataFrame({"Precision:":p_list,
                      "Recall:":r_list,
                     "f1_score:":f_list},index=ind))
        
    def fit(self, X, y, epochs=100, random_state=42,loss_plot=False,verbose=False,print_every=10):
        '''
        Train the model via backprop for the specified number of epochs.
       
        '''
        print ("The shape of the data is: ",X.shape)
        print ("\n------------------------------")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=random_state)
        ##SGD 
        y_train_enc=np.array([self.one_hot(obj_class,num_classes=len(set(y))) for obj_class in y_train])
        #y_test_enc=np.array([self.one_hot(obj_class,num_classes=len(set(y))) for obj_class in y_test])
        
        training_loss=[]
        
        #accuracy test, train
        epoch_train_accuracy=[]
        self.epoch_train_accuracy=epoch_train_accuracy
        epoch_test_accuracy=[]
        self.epoch_test_accuracy=epoch_test_accuracy
        
        #precision test, train
        epoch_train_precision=[]
        self.epoch_train_precision=epoch_train_precision
        epoch_test_precision=[]
        self.epoch_test_precision=epoch_test_precision
        
        #recall test, train
        epoch_train_recall=[]
        self.epoch_train_recall=epoch_train_recall
        epoch_test_recall=[]
        self.epoch_test_recall=epoch_test_recall
        
        #f1 macro test, train
        epoch_train_f1_macro=[]
        self.epoch_train_f1_macro=epoch_train_f1_macro
        epoch_test_f1_macro=[]
        self.epoch_test_f1_macro=epoch_test_f1_macro
        
        #f1 macro test, train
        epoch_train_f1_micro=[]
        self.epoch_train_f1_micro=epoch_train_f1_micro
        epoch_test_f1_micro=[]
        self.epoch_test_f1_micro=epoch_test_f1_micro
        
        #for epoch 
        for epoch in range(epochs): #loop through all training data
            #run through each element in the batch
            loss= []
            epoch_y_train=[]
            epoch_y_test=[]
            for i,x in enumerate(X_train): 
        
                
                x= np.expand_dims(x,axis=0)
                
                
                
                prediction= self.predict(x) #sig(weights*x) gives the class probaility
                
                loss.append(self.CrossEntropy(y_train_enc[i],prediction)) #get the loss for row 

                #return None
                #Back Prop 
                dw_2= np.dot(self.hidden_layer.T,(y_train_enc[i]-prediction))
                dw_1=np.dot(x.T,(np.dot((y_train_enc[i]-prediction),self.w2.T)*self.hidden_layer*(1-self.hidden_layer)))
                db_2= y_train_enc[i]-prediction
                db_1= np.dot((y_train_enc[i]-prediction),self.w2.T)*self.hidden_layer*(1-self.hidden_layer)
          
                #update weights 
                self.w1 += self.learn_rate*(dw_1/len(y_train))
                self.w2 += self.learn_rate*(dw_2/len(y_train))
                self.bias1 += self.learn_rate*(db_1/len(y_train))
                self.bias2 += self.learn_rate*(db_2/len(y_train))
                
                #validation_loss.append(val_loss)
                epoch_y_train.append(self.convert_to_class(prediction)) #get all the predictions for each item in training batch
        
            epoch_y_test= self.predict(X_test) #get all the predictions for each item in training batch
                 
            #get confusion matrix 
            cm_train = self.confusion_matrix(y_train,epoch_y_train)
            cm_test = self.confusion_matrix(y_test, self.convert_to_class(epoch_y_test))
            
            
            #accuracy 
            self.epoch_train_accuracy.append(accuracy_score(y_train,epoch_y_train))
            self.epoch_test_accuracy.append(accuracy_score(y_test,self.convert_to_class(epoch_y_test)))
            
            #precision 
            self.epoch_train_precision.append(self.precision(cm_train,'macro'))
            self.epoch_test_precision.append(self.precision(cm_test,'macro'))
            
            #recall
            self.epoch_train_recall.append(self.recall(cm_train,'macro'))
            self.epoch_test_recall.append(self.recall(cm_test,'macro'))
        
            
            #f1 macro
            self.epoch_train_f1_macro.append(self.f1(cm_train,'macro'))
            self.epoch_test_f1_macro.append(self.f1(cm_test,'macro'))
            
            #f1 micro
            self.epoch_train_f1_micro.append(self.f1(cm_train,'micro'))
            self.epoch_test_f1_micro.append(self.f1(cm_test,'micro'))
            
            training_loss.append(np.average(loss))
            if verbose == True:
                if epoch % print_every == 0:
                    print ("\n For Epoch:", epoch, "With Training Loss:", np.around(np.average(loss),decimals=6))
                    pass 
        
        if loss_plot == True:
            self.loss_plot(training_loss)
            pass 

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.metrics import confusion_matrix

    X, y = make_classification(n_samples=2000, n_features=10, n_informative=8,n_redundant = 2,
                               n_classes=5) 


    multi_class_mlp=MMLP(X.shape[1], output_size=5, hidden_dims=32, alpha=.2)
    multi_class_mlp.fit(X, y, epochs=50, loss_plot=True, verbose=True, print_every=10)


    print("\nResults from Numpy Built Multi Class Classifier\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
    y_preds= multi_class_mlp.predict(X_test)
    y_preds= multi_class_mlp.convert_to_class(y_preds)
    print(multi_class_mlp.create_report(y_test,y_preds))

