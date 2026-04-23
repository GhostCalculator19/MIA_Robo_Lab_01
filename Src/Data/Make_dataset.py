from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
import yaml


def main() -> int:

    # Load configuration
    with open('Config/paramts.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Load raw data
    df = pd.read_csv(config['data']['raw_dataset_csv'])

    # Text cols encoding
    

    for col in ['gender', 'lunch', 'school_type', 'teaching_method']:
        lb = LabelBinarizer()
        df[col] = lb.fit_transform(df[col])

    for col in ['school_setting']:
        ohe = OneHotEncoder()
        encoded = ohe.fit_transform(df[[col]]).toarray()
        df = df.join(pd.DataFrame(encoded, columns=[f"{col}_{c}" for c in ohe.categories_[0]], index=df.index))
        df.drop(col, axis=1, inplace=True)   

    # Numeric cols scaling
    scaler = StandardScaler()
    df[['n_student','pretest']] = scaler.fit_transform(df[['n_student','pretest']])
    
    #Drop cols
    df.drop(['student_id', 'school', 'classroom'], axis=1, inplace=True)

    # Save data all
    df.to_csv(config['data']['processed_dataset_csv'], index=False)

    # Generate report
    profile = ProfileReport(df, title="Data Profiling Report")
    profile.to_file(config['reports']['ydata_report_path'])
    
    # Save data to Numpy
    data_x, data_y = np.array(df.drop('posttest', axis=1)), np.array(df['posttest'])
    np.save(config['data']['dataset_x_path_np'], data_x)
    np.save(config['data']['dataset_y_path_np'], data_y)
    
    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=config['base']['test_size'],
        random_state=config['base']['random_state']
    )

    # Save splitted data
    train_df.to_csv(config['data']['train_path'], index=False)
    test_df.to_csv(config['data']['test_path'], index=False)

    return 0

if __name__ == '__main__':
    main()