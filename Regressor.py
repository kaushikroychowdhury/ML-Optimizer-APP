import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes


# Functions ............................................................................................................

def filedownload(df):

    """
    filedownload function converts the dataframe df into csv file and downloads it.

    :param df: dataframe containing max_feature, n_estimators, R^2.
    """

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href
def build_model_Adaboost_Regressor(df):

    """
    It builds a model using Adaboost regresion Algorithm.
    Takes input from streamlit web interface and use those inputs for building the model.
    Used GridSearchCV for Hyperparameter Tunning.
    Ploting the result using Plotly Framework.

    :param df: dataframe containing features and labels.
    """

    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    adaboost = AdaBoostRegressor(loss= loss, random_state= random_state)

    grid = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, n_jobs=n_jobs)
    grid.fit(X_train, Y_train)

    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info(mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info(mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info(mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        all = True
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info(mse)
        st.write('Root Mean Squared Error (RMSE):')
        rsme = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info(rsme)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info(mae)

    st.write("The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    # Grid Data .......
    grid_results = pd.concat(
        [pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],
        axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['learning_rate', 'n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['learning_rate', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('learning_rate', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # -----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Learning_rate')
        ))
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='Learning_Rate',
                          zaxis_title='R2'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    if all == True:
        criteria = ['RMSE', 'MSE', 'MAE']
        colors = {'RMSE': 'red',
                  'MSE': 'orange',
                  'MAE': 'lightgreen'}
        fig = go.Figure([go.Bar(x=criteria, y=[rsme, mse, mae])], marker={'color': colors[criteria]})


    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.show()

    # -----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x, y, z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)

def build_model_RandomForestRegressor(df):

    """
    It builds a model using Adaboost regresion Algorithm.
    Takes input from streamlit web interface and use those inputs for building the model.
    Used GridSearchCV for Hyperparameter Tunning.
    Ploting the result using Plotly Framework.

    :param df: dataframe containing features and labels.
    """

    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    # X_train.shape, Y_train.shape
    # X_test.shape, Y_test.shape

    rf = RandomForestRegressor(n_estimators=n_estimators,
                               random_state=random_state,
                               max_features=max_features,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap,
                               oob_score=oob_score,
                               n_jobs=n_jobs)

    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)


    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    if criterion == 'MSE':
        st.write('Mean Squared Error (MSE):')
        st.info(mean_squared_error(Y_test, Y_pred_test))
    if criterion == 'MAE':
        st.write('Mean Absolute Error (MAE):')
        st.info(mean_absolute_error(Y_test, Y_pred_test))
    if criterion == 'RMSE':
        st.write('Root Mean Squared Error (RMSE):')
        st.info(mean_squared_error(Y_test, Y_pred_test, squared=False))
    if criterion == 'All':
        all = True
        st.write('Mean Squared Error (MSE):')
        mse = mean_squared_error(Y_test, Y_pred_test)
        st.info(mse)
        st.write('Root Mean Squared Error (RMSE):')
        rmse = mean_squared_error(Y_test, Y_pred_test, squared=False)
        st.info(rmse)
        st.write('Mean Absolute Error (MAE):')
        mae = mean_absolute_error(Y_test, Y_pred_test)
        st.info(mae)

    st.write("The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    # Grid Data .......
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])], axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # -----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_features')
        ))
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning (Surface Plot)',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='max_features',
                          zaxis_title='R2'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))

    st.plotly_chart(fig)

    if all == True:
        criteria = ['RMSE', 'MSE', 'MAE']
        fig = go.Figure([go.Bar(x=criteria, y=[rmse, mse, mae])])

    st.plotly_chart(fig)

    # -----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x, y, z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)


# Page Layout ( Streamlit web Interface )
st.set_page_config(page_title="HyperParameter Optimization")

st.write("""
# Regressor Hyperparameter Optimizer
""")

# Sidebar ..............................................

# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

st.sidebar.header("Parameter Configuration")
split_size = st.sidebar.slider('Data Split Ratio (training set)', 10,90,80,5)

st.sidebar.header("Select Regressor")
reg = st.sidebar.selectbox("Choose Regression Algorithm", options=['Random Forest Regressor', 'Adaboost Regressor'])
if reg == 'Random Forest Regressor':
    st.sidebar.subheader('Learning Parameters')
    n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10, 50), 50)
    n_estimators_step = st.sidebar.number_input('Step size for n_estimators (n_estimators_step)', 10)
    st.sidebar.write('---')
    max_features = st.sidebar.slider('Max features', 1, 50, (1, 3), 1)
    max_features_step = st.sidebar.number_input('Step Size for max Features', 1)
    st.sidebar.write('---')
    min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)',
                                         1, 10, 2, 1)

    st.sidebar.subheader('General Parameters')
    random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])
    bootstrap = st.sidebar.selectbox('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    oob_score = st.sidebar.selectbox('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)',
                                     options=[False, True])
    n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(n_estimators[0], n_estimators[1] + n_estimators_step, n_estimators_step)
    max_features_range = np.arange(max_features[0], max_features[1] + max_features_step, max_features_step)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

if reg == 'Adaboost Regressor':
    st.sidebar.subheader('Learning Parameters')
    n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10, 50), 50)
    n_estimators_step = st.sidebar.number_input('Step size for n_estimators (n_estimators_step)', 10)
    st.sidebar.write('---')
    criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['All', 'MSE', 'MAE', 'RMSE'])
    lr = [0.0001, 0.001, 0.01, 0.1]
    learning_rate = st.sidebar.select_slider('Range of Learning Rate (learning_rate)',
                                             options=[0.0001, 0.001, 0.01, 0.1], value=(0.0001, 0.01))
    l = lr.index(learning_rate[0])
    r = lr.index(learning_rate[1])
    learning_rate_range = lr[l:r + 1]

    st.sidebar.write('---')

    st.sidebar.header("Loss")
    loss = st.sidebar.selectbox("Choose Loss",options=['linear', 'square', 'exponential'])

    st.sidebar.subheader('General Parameters')
    random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)

    n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(n_estimators[0], n_estimators[1] + n_estimators_step, n_estimators_step)

    param_grid = dict(learning_rate = learning_rate_range, n_estimators=n_estimators_range)

# main Body ...............................................................................................

st.subheader('Dataset')




if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    if reg == 'Random Forest Regressor':
        build_model_RandomForestRegressor(df)
    if reg == 'Adaboost Regressor':
        build_model_Adaboost_Regressor(df)

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)

        st.markdown('The **Diabetes** dataset is used as the example.')
        st.write(df.head(5))

        if reg == 'Random Forest Regressor':
            build_model_RandomForestRegressor(df)

        if reg == 'Adaboost Regressor':
            build_model_Adaboost_Regressor(df)