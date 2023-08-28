__version__ = '0.0.2'

def getversion():
    print(f'{__version__}')


# Bibliotecas - Gráficas

def gf_freq(dataframe, column, agg, title):
    import matplotlib.pyplot as plt
    dataframe[column].asfreq(agg).plot(linewidth=3)
    plt.title(title +' per '+agg)
    plt.xlabel(agg)
    plt.xlabel(title);

def gf_nullvalues(dataFrame):
    import seaborn as sns
    sns.heatmap(dataFrame.isnull())

def gf_epochhistory(epochhist):
    import matplotlib.pyplot as plt
    plt.plot(epochhist.history['loss'])
    plt.plot(epochhist.history['val_loss'])
    plt.title('Model loss progress during training')
    plt.xlabel('Epochs')
    plt.ylabel('Training and Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss' ])

def gf_bubblechart(df, x, y):
    import plotly.express as px
    df_sorted = df = df.sort_values(y, ascending = False)
    fig = px.scatter(
        df, y=y, x=x, color='Exports', size='Exports', size_max=20,
        color_continuous_scale = px.color.sequential.RbBu,
    )
    fig.update_layout(
        paper_bg_color='white',
        plot_bgcolor='white'
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_layout(height=500, width=1000)
    fig.update_coloraxes(colorbar=dict(title='Exports'))
    fig.update_traces(marker=dict(sizeref=0.09))
    fig.update_yaxes(title=y)
    fig.update_xaxes(title=x)
    fig.update_layout(showlegend=False)
    fig.show()

# Bibliotecas - Machine Learning

def ml_regressionevaluation(y_test, y_predict, X_test):
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


# Bibliotecas - Tensorflow

def tf_getModelMemoryUsage(batch_size, model):
    
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

# Image processing

def img_filtering(img_path, filter):
    import cv2
    import numpy as np
    
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    identity = np.array(([0, 0, 0],[0, 1, 0], [0, 0, 0]), np.float32)
    edge_1 = np.array(([0, -1, 0],[-1, 4, -1],[0, -1, 0]), np.float32)
    edge_2 = np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]), np.float32)
    sharpen = np.array(([0, -1, 0],[-1, 5, -1], [0, -1, 0]), np.float32)
    box_blur = np.array(([1, 1, 1],[1, 1, 1], [1, 1, 1]), np.float32)/9
    gaussing_blur_3 = np.array(([1, 2, 1],[2, 4, 2], [1, 2, 1]), np.float32)/16
    gaussing_blur_5 = np.array(([1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]), np.float32)/256

    if filter == 'edge_1':
        k = edge_1
    elif filter == 'edge_2':
        k = edge_2
    elif filter == 'sharpen':
        k = sharpen
    elif filter == 'box_blur':
        k = box_blur
    elif filter == 'gaussing_blur_3':
        k = gaussing_blur_3
    elif filter == 'gaussing_blur_5':
        k = gaussing_blur_5
    else:
        k = identity

    filtered_img = cv2.filter2D(img, -1, k)
    return filtered_img