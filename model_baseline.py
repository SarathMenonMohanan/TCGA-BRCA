from sklearn.linear_model import LogisticRegression
from util_components.FCN_performance_index import FCN_performance_index



def FCN_baseline_sim(x_train,y_train,x_test,y_test,iteration):
    model = FCN_baseline_training(x_train,y_train,iteration)
    y_pred = FCN_baseline_pred(model,x_test)
    
    return  FCN_performance_index(y_test,y_pred)

def FCN_baseline_training(x_train,y_train,iteration):
    model = LogisticRegression(max_iter = iteration , class_weight = "balanced")
    model.fit(x_train,y_train)
    return model

def FCN_baseline_pred(model,x_test):
    return model.predict(x_test)

    
    

    