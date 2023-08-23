__version__ = '0.1.dev0'

def graphFreq(dataframe, column, agg, title):
    import matplotlib.pyplot as plt
    dataframe[column].asfreq(agg).plot(linewidth=3)
    plt.title(title +' per '+agg)
    plt.xlabel(agg)
    plt.xlabel(title);

def checkNullVisual(dataFrame):
    import seaborn as sns
    sns.heatmap(dataFrame.isnull())

def epochHistoryGraph(epochhist):
    import matplotlib.pyplot as plt
    plt.plot(epochhist.history['loss'])
    plt.plot(epochhist.history['val_loss'])
    plt.title('Model loss progress during training')
    plt.xlabel('Epochs')
    plt.ylabel('Training and Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss' ])

def regressionEvaluation(y_test, y_predict, X_test):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from math import sqrt
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_predict)
    k = X_test.shape[1]
    n = len(X_test)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2:', r2)
    print('R2 Ajustado:', adj_r2)

def get_model_memory_usage(batch_size, model):
    
    features_mem = 0 # Initialize memory for features. 
    float_bytes = 4.0 #Multiplication factor as all values we store would be float32.
    
    for layer in model.layers:

        out_shape = layer.output_shape
        
        if type(out_shape) is list:   #e.g. input layer which is a list
            out_shape = out_shape[0]
        else:
            out_shape = [out_shape[1], out_shape[2], out_shape[3]]
            
        #Multiply all shapes to get the total number per layer.    
        single_layer_mem = 1 
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        
        single_layer_mem_float = single_layer_mem * float_bytes #Multiply by 4 bytes (float)
        single_layer_mem_MB = single_layer_mem_float/(1024**2)  #Convert to MB
        
        print("Memory for", out_shape, " layer in MB is:", single_layer_mem_MB)
        features_mem += single_layer_mem_MB  #Add to total feature memory count

def bubblechart(df):
    df_sorted = df = df.sort_values("Country", ascending = False)
