import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define feature groups for each position
POSITION_FEATURES = {
    "Back": [
        "clearances", "interceptions", "defensive_duels", "defensive_duels_won", 
        "aerial_duels", "aerial_duels_won", "recoveries", "recoveries_in_own_half",
        "fouls", "fouls_drawn", "passes_into_final_third", "back_passes", "shots"
    ],
    "Midfielder": [
        "total_passes", "total_passes_completed", "progressive_runs_attempted", 
        "passes_into_final_third", "passes_into_box", "assists", "dribbles", 
        "successful_dribbles", "shots", "goals", "aerial_duels_won", "loose_ball_duels"
    ],
    "Forward": [
        "dribbles", "successful_dribbles", "crosses", "accurate_crosses", 
        "touches_inside_box", "passes_into_box", "shots", "goals", "xg", 
        "passes_into_final_third", "aerial_duels", "fouls_drawn"
    ]
}

def prepare_data(df, features=POSITION_FEATURES):
    """
    Prepare data for model training
    
    Args:
        df (pd.DataFrame): Input dataframe with player statistics
        features (dict): Dictionary of features for each position
    
    Returns:
        tuple: Scaled features and target variable
    """
    # Combine all possible features
    selected_columns = ['player_name', 'Simple_Position', 'team']
    selected_columns += [col for role in features.values() for col in role]
    
    # Select relevant columns
    df = df[selected_columns].reset_index(drop=True)
    
    # Separate features and target
    X = df.drop(['player_name', 'Simple_Position', 'team'], axis=1)
    y = df['Simple_Position']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler

def train_position_classifier_with_cv(X, y):
    """
    Train Random Forest Classifier with cross-validation
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
    
    Returns:
        dict: Cross-validation results and trained model
    """
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, 
        X, y, 
        cv=cv,
        scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    )
    
    # Fit final model on entire dataset
    model.fit(X, y)
    
    return {
        'model': model,
        'cv_scores': {
            'accuracy': cv_results['test_accuracy'].mean(),
            'f1': cv_results['test_f1_weighted'].mean(),
            'precision': cv_results['test_precision_weighted'].mean(),
            'recall': cv_results['test_recall_weighted'].mean()
        }
    }

def predict_optimal_lineup(df_test, model, scaler, formation='4-3-3', features=POSITION_FEATURES):
    """
    Predict optimal lineup based on player position probabilities
    
    Args:
        df_test (pd.DataFrame): Players to predict positions for
        model (RandomForestClassifier): Trained position classifier
        scaler (MinMaxScaler): Feature scaler
        formation (str): Tactical formation
    
    Returns:
        dict: Predicted lineup with players for each position
    """
    # Formation requirements
    formation_requirements = {
        '4-3-3': {'Back': 4, 'Midfielder': 3, 'Forward': 3},
        '4-4-2': {'Back': 4, 'Midfielder': 4, 'Forward': 2},
        '3-5-2': {'Back': 3, 'Midfielder': 5, 'Forward': 2}
    }
    
    selected_columns = ['player_name', 'Simple_Position', 'team']
    feature_columns = [col for role in features.values() for col in role]
    selected_columns += feature_columns
    
    df_test = df_test[selected_columns].reset_index(drop=True)

    y_test = df_test['Simple_Position']

    # Prepare test data
    X_test = scaler.transform(
        df_test.drop(['player_name', 'Simple_Position', 'team'], axis=1))
    X_test_scaled = pd.DataFrame(X_test,
                                 columns=feature_columns)
    
    # Get position probabilities
    position_probs = model.predict_proba(X_test_scaled)
    y_pred = model.predict(X_test_scaled)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'player_name': df_test['player_name'],
        'Back_prob': position_probs[:, 0],
        'Midfielder_prob': position_probs[:, 1],
        'Forward_prob': position_probs[:, 2]
    })

    condensed_predictions = predictions_df.groupby('player_name')[
        ['Back_prob', 'Midfielder_prob', 'Forward_prob']
    ].mean().reset_index()

    # Select top players for each position without duplication
    lineup = {}
    selected_players = set()  # Keep track of already selected players

    for position in ['Back', 'Midfielder', 'Forward']:
        n_players = formation_requirements[formation][position]
        
        # Filter out already selected players
        available_players = condensed_predictions[~condensed_predictions['player_name'].isin(selected_players)]
        
        # Select top players for this position
        top_players = available_players.nlargest(
            n_players, 
            f'{position}_prob'
        )['player_name'].tolist()
        
        lineup[position] = top_players
        selected_players.update(top_players)  # Add selected players to the exclusion set
        print(top_players)

    return {'lineup': lineup,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'accuracy': model.score(X_test_scaled, y_test)
    }

def main():
    # Load data
    df = pd.read_csv('player-data/cleaned_player.csv')

    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['year'] = df['Date'].dt.year
    df = df[df['Simple_Position'] != 'Goalkeeper'].reset_index(drop = True)

    df_test = df[(df['team'].str.contains('Northwestern') == True) & (df['year'] == 2024)]
    df_train = df[(df['team'].str.contains('Northwestern') == False) & (df['year'] != 2024)]
    
    # Prepare data
    X_train, y_train, scaler = prepare_data(df_train)
    
    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    
    # Train model
    cv_results = train_position_classifier_with_cv(X_train, y_train)
    model = cv_results['model']
    print("Cross-Validation Metrics:")
    for metric, score in cv_results['cv_scores'].items():
        print(f"{metric.capitalize()}: {score:.4f}")
    

    # Predict lineup
    optimal_lineup = predict_optimal_lineup(df_test, model, scaler, formation='4-3-3')
    print("\nOptimal Lineup:")
    for position, players in optimal_lineup['lineup'].items():
        print(f"{position}: {players}")

    print("Model Performance:")
    print(optimal_lineup['classification_report'])
    


if __name__ == "__main__":
    main()