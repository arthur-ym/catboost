import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(df, target_variable):
    """
    Plot a correlation heatmap of the features, including the target variable.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing features and the target variable.
    - target_variable: str - The name of the target variable.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()


def plot_heat_map(X, y, step=10):
    """
    Plots correlation heatmaps for every `step` features in the DataFrame `X`,
    including the target variable from the Series `y` in each plot.

    Parameters:
    - X: pd.DataFrame - The input DataFrame with features.
    - y: pd.Series - The Series containing the target variable.
    - step: int - The number of features to include in each heatmap.
    """
    num_features = X.shape[1]  # Number of columns in the dataset

    for i in range(0, num_features, step):
        # Select a subset of features
        subset = X.iloc[:, i:i+step]
        
        # Add the target variable (Series) to the subset
        subset['target_variable'] = y
        
        # Calculate the correlation matrix
        correlation_matrix = subset.corr()
        
        # Plot the heatmap
        plt.figure(figsize=(7, 4))
        sns.heatmap(correlation_matrix, annot=True,
                    cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Heatmap for Features {i+1} to {min(i+step, num_features)} (including target variable)")
        plt.show()

def print_cv_summary(cv_data):

    best_value = cv_data['test-Logloss-mean'].min()
    best_iter = cv_data['test-Logloss-mean'].values.argmin()

    print('Best validation Logloss score : {:.4f}±{:.4f} on step {}'.format(
        best_value,
        cv_data['test-Logloss-std'][best_iter],
        best_iter)
    )

def plot_traing_process(cv_data):
    train_loss = cv_data['train-Logloss-mean']
    test_loss = cv_data['test-Logloss-mean']
    iterations =cv_data['iterations']
 
    plt.figure(figsize=(7, 4))
    plt.plot(iterations, train_loss, label='Training Loss', color='blue')
    plt.plot(iterations, test_loss, label='Validation Loss', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('CatBoost Training Progress')
    plt.legend()
    plt.grid()
    plt.show()
    

def check_missing_values(df):
    """
    Check for missing values in the DataFrame and return the percentage of nulls for every column.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame to check.
    
    Returns:
    - pd.DataFrame - A DataFrame showing columns with their percentage of missing values.
    """
    # Calculate the number of missing values per column
    missing_values = df.isnull().sum()
    
    # Calculate the percentage of missing values per column
    missing_percentage = 100 * missing_values / len(df)
    
    # Create a DataFrame to summarize missing values and their percentages
    missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
    
    # Return all columns with their percentage of missing values, sorted by percentage
    return missing_df.sort_values(by='Percentage', ascending=False)



def plot_feature_distributions(df, target_variable):
    """
    Plot the distribution of each feature split by the target variable.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing features and the target variable.
    - target_variable: str - The name of the target variable.
    """
    features = df.drop(columns=[target_variable]).columns
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x=feature, hue=target_variable, kde=True, element="step")
        plt.title(f'Distribution of {feature} by {target_variable}')
        plt.show()

def check_class_balance(df, target_variable):
    """
    Check the balance of classes in the target variable.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing the target variable.
    - target_variable: str - The name of the target variable.
    
    Returns:
    - pd.Series - A series showing the count of each class.
    """
    class_balance = df[target_variable].value_counts()
    print("Class Balance:\n", class_balance)
    return class_balance


def plot_pairplot(df, target_variable):
    """
    Create pair plots to visualize interactions between features for different classes.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing features and the target variable.
    - target_variable: str - The name of the target variable.
    """
    sns.pairplot(df, hue=target_variable, diag_kind='kde')
    plt.show()

def summary_statistics(df, target_variable):
    """
    Provide summary statistics for each feature, grouped by the target variable.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing features and the target variable.
    - target_variable: str - The name of the target variable.
    
    Returns:
    - pd.DataFrame - A DataFrame showing summary statistics grouped by the target variable.
    """
    return df.groupby(target_variable).describe().T


def detect_outliers(df):
    """
    Detect outliers in the DataFrame using the IQR method.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame to check for outliers.
    
    Returns:
    - pd.DataFrame - A DataFrame showing outlier counts for each feature.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    return pd.DataFrame({'Outliers': outliers})
