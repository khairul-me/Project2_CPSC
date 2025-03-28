import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import os

class CovidHMM:
    """
    A Hidden Markov Model implementation for COVID-19 case prediction.
    Hidden states represent infection levels, observable evidence is month of the year.
    """
    
    def __init__(self, file_path):
        """
        Initialize the HMM with file path to COVID data.
        
        Args:
            file_path (str): Path to the CSV file containing COVID data
        """
        self.file_path = file_path
        self.data = None
        self.processed_data = None
        
        # Define 10 hidden states representing infection levels
        self.states = [
            "level_0_1k",     # New cases per million < 1k
            "level_1k_2k",    # 1k - 2k
            "level_2k_3k",    # 2k - 3k
            "level_3k_4k",    # 3k - 4k
            "level_4k_5k",    # 4k - 5k
            "level_5k_6k",    # 5k - 6k
            "level_6k_7k",    # 6k - 7k
            "level_7k_8k",    # 7k - 8k
            "level_8k_9k",    # 8k - 9k
            "level_9k_plus"   # > 9k
        ]
        
        # Observable evidence: 12 months
        self.observations = list(range(1, 13))  # 1-12 representing Jan-Dec
        
        # Initialize transition probabilities (will be learned from data)
        self.transition_prob = np.zeros((len(self.states), len(self.states)))
        
        # Initialize emission probabilities (will be learned from data)
        self.emission_prob = np.zeros((len(self.states), len(self.observations)))
        
        # Initialize initial state probabilities
        self.initial_prob = np.zeros(len(self.states))
        
        # Load and preprocess the data
        self.load_data()
        self.process_monthly_data()
        self.train_hmm()
        
    def load_data(self):
        """Load and preprocess the COVID data from CSV."""
        # Read the CSV file
        self.data = pd.read_csv(self.file_path)
        
        # Convert date column to datetime
        self.data['date'] = pd.to_datetime(self.data['date'], format='%m/%d/%Y')
        
        # Sort data by date
        self.data = self.data.sort_values('date')
        
        # Rename columns for better readability
        self.data = self.data.rename(columns={
            'new_cases_per_million': 'daily_cases',
            'total_cases_per_million': 'cumulative_cases',
            'new_cases_per_million_7_day_avg_right': '7day_avg'
        })
        
        # Fill any missing values in the 7-day average column
        self.data['7day_avg'] = self.data['7day_avg'].fillna(self.data['daily_cases'].rolling(window=7).mean())
        
        print(f"Data loaded successfully. Time range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Total records: {len(self.data)}")
    
    def process_monthly_data(self):
        """Process data to monthly sums and determine state levels."""
        # Extract year and month
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        
        # Group by year and month and sum daily cases
        monthly_data = self.data.groupby(['year', 'month'])['daily_cases'].sum().reset_index()
        monthly_data['date'] = pd.to_datetime(monthly_data['year'].astype(str) + '-' + monthly_data['month'].astype(str) + '-01')
        
        # Determine the hidden state for each month based on case levels
        monthly_data['state'] = pd.cut(
            monthly_data['daily_cases'], 
            bins=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, float('inf')],
            labels=self.states,
            include_lowest=True
        )
        
        self.processed_data = monthly_data
        print(f"Data processed into {len(monthly_data)} monthly records.")
    
    def train_hmm(self):
        """Train the HMM by learning transition and emission probabilities from the data."""
        # Count initial states
        initial_counts = np.zeros(len(self.states))
        state_map = {state: i for i, state in enumerate(self.states)}
        
        for state in self.processed_data.iloc[0:1]['state']:
            initial_counts[state_map[state]] += 1
        
        # Normalize to get initial probabilities
        self.initial_prob = initial_counts / initial_counts.sum()
        
        # Count transitions between states
        transition_counts = np.zeros((len(self.states), len(self.states)))
        
        for i in range(len(self.processed_data) - 1):
            current_state = self.processed_data.iloc[i]['state']
            next_state = self.processed_data.iloc[i + 1]['state']
            
            if pd.notnull(current_state) and pd.notnull(next_state):
                transition_counts[state_map[current_state], state_map[next_state]] += 1
        
        # Normalize rows to get transition probabilities
        for i in range(len(self.states)):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_prob[i] = transition_counts[i] / row_sum
            else:
                # If no transitions from this state, use uniform distribution
                self.transition_prob[i] = np.ones(len(self.states)) / len(self.states)
        
        # Count emissions (state to month)
        emission_counts = np.zeros((len(self.states), len(self.observations)))
        
        for _, row in self.processed_data.iterrows():
            if pd.notnull(row['state']):
                state_idx = state_map[row['state']]
                month_idx = int(row['month']) - 1  # Convert 1-12 to 0-11 for indexing
                emission_counts[state_idx, month_idx] += 1
        
        # Normalize rows to get emission probabilities
        for i in range(len(self.states)):
            row_sum = emission_counts[i].sum()
            if row_sum > 0:
                self.emission_prob[i] = emission_counts[i] / row_sum
            else:
                # If no emissions from this state, use uniform distribution
                self.emission_prob[i] = np.ones(len(self.observations)) / len(self.observations)
        
        print("HMM trained successfully.")
    
    def filter_current_month(self, month):
        """
        Filter the level of infection for the current month.
        
        Args:
            month (int): Month number (1-12)
            
        Returns:
            list: Probability distribution over states for the given month
        """
        month_idx = month - 1  # Convert 1-12 to 0-11 for indexing
        
        # Filter calculation (simple multiplication with emission probabilities)
        filtered_probs = np.zeros(len(self.states))
        
        for state_idx in range(len(self.states)):
            filtered_probs[state_idx] = self.initial_prob[state_idx] * self.emission_prob[state_idx, month_idx]
        
        # Normalize
        if filtered_probs.sum() > 0:
            filtered_probs = filtered_probs / filtered_probs.sum()
        
        return filtered_probs
    
    def predict_next_months(self, month, num_months=3):
        """
        Predict the next n months' level of infection given the current month.
        
        Args:
            month (int): Current month number (1-12)
            num_months (int): Number of months to predict (default: 3)
            
        Returns:
            list: List of probability distributions for each predicted month
        """
        current_probs = self.filter_current_month(month)
        predictions = []
        
        for _ in range(num_months):
            # Predict next state using transition matrix
            next_probs = np.zeros(len(self.states))
            
            for next_state in range(len(self.states)):
                for current_state in range(len(self.states)):
                    next_probs[next_state] += current_probs[current_state] * self.transition_prob[current_state, next_state]
            
            # Update current state for next iteration
            current_probs = next_probs
            predictions.append(next_probs)
            
            # Update month for next iteration
            month = month % 12 + 1
        
        return predictions
    
    def viterbi_algorithm(self, start_month, end_month):
        """
        Use the Viterbi algorithm to find the most likely sequence of infection levels.
        
        Args:
            start_month (int): Starting month number (1-12)
            end_month (int): Ending month number (1-12)
            
        Returns:
            list: The most likely sequence of infection levels
        """
        # Calculate number of months
        num_months = (end_month - start_month) % 12
        if num_months == 0 and start_month != end_month:
            num_months = 12
        num_months += 1  # Include end month
        
        # Initialize Viterbi variables
        V = np.zeros((len(self.states), num_months))
        backpointer = np.zeros((len(self.states), num_months), dtype=int)
        
        # Initialize first month
        month_idx = start_month - 1  # Convert 1-12 to 0-11 for indexing
        for s in range(len(self.states)):
            V[s, 0] = np.log(self.initial_prob[s] + 1e-10) + np.log(self.emission_prob[s, month_idx] + 1e-10)
        
        # Recursion step
        for t in range(1, num_months):
            month_idx = (start_month + t - 1) % 12  # Calculate month index for each step
            if month_idx == 0:  # Handle December to January transition
                month_idx = 12 - 1
                
            for s in range(len(self.states)):
                max_val = -np.inf
                max_state = 0
                
                for prev_s in range(len(self.states)):
                    val = V[prev_s, t-1] + np.log(self.transition_prob[prev_s, s] + 1e-10)
                    if val > max_val:
                        max_val = val
                        max_state = prev_s
                
                V[s, t] = max_val + np.log(self.emission_prob[s, month_idx] + 1e-10)
                backpointer[s, t] = max_state
        
        # Termination step
        last_state = np.argmax(V[:, -1])
        
        # Backtracking
        path = [last_state]
        for t in range(num_months - 1, 0, -1):
            last_state = backpointer[last_state, t]
            path.insert(0, last_state)
        
        # Convert state indices to state names
        state_path = [self.states[s] for s in path]
        
        return state_path
    
    def visualize_hmm(self, save_path=None):
        """
        Visualize the HMM components (transition and emission probabilities).
        
        Args:
            save_path (str, optional): Path to save the visualizations
        """
        # Create a directory for visualizations if it doesn't exist
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Plot transition probabilities as a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.transition_prob, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=[s.replace("level_", "") for s in self.states],
                   yticklabels=[s.replace("level_", "") for s in self.states])
        plt.title('Transition Probabilities Between Infection Levels', fontsize=16)
        plt.xlabel('To State', fontsize=12)
        plt.ylabel('From State', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/transition_probabilities.png")
            print(f"Transition probabilities visualization saved to {save_path}/transition_probabilities.png")
        else:
            plt.show()
        
        # Plot emission probabilities as a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.emission_prob, annot=True, fmt=".2f", cmap="YlOrRd",
                   xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   yticklabels=[s.replace("level_", "") for s in self.states])
        plt.title('Emission Probabilities: Infection Levels to Months', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Infection Level', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/emission_probabilities.png")
            print(f"Emission probabilities visualization saved to {save_path}/emission_probabilities.png")
        else:
            plt.show()
        
        # Plot the initial state probabilities
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.states)), self.initial_prob)
        plt.xticks(range(len(self.states)), [s.replace("level_", "") for s in self.states], rotation=45)
        plt.title('Initial State Probabilities', fontsize=16)
        plt.xlabel('Infection Level', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/initial_probabilities.png")
            print(f"Initial probabilities visualization saved to {save_path}/initial_probabilities.png")
        else:
            plt.show()
    
    def visualize_viterbi_path(self, start_month, input_month, save_path=None):
        """
        Visualize the most likely path of infection levels from start_month to input_month.
        
        Args:
            start_month (int): Starting month number (1-12)
            input_month (int): Ending month number (1-12)
            save_path (str, optional): Path to save the visualization
        """
        state_path = self.viterbi_algorithm(start_month, input_month)
        
        # Create month labels for the x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = []
        current_month = start_month - 1  # Convert to 0-indexed
        
        for _ in range(len(state_path)):
            months.append(month_names[current_month])
            current_month = (current_month + 1) % 12
        
        # Convert state names to numeric values for plotting
        state_values = []
        for state in state_path:
            if state == "level_0_1k":
                state_values.append(0.5)
            elif state == "level_1k_2k":
                state_values.append(1.5)
            elif state == "level_2k_3k":
                state_values.append(2.5)
            elif state == "level_3k_4k":
                state_values.append(3.5)
            elif state == "level_4k_5k":
                state_values.append(4.5)
            elif state == "level_5k_6k":
                state_values.append(5.5)
            elif state == "level_6k_7k":
                state_values.append(6.5)
            elif state == "level_7k_8k":
                state_values.append(7.5)
            elif state == "level_8k_9k":
                state_values.append(8.5)
            elif state == "level_9k_plus":
                state_values.append(9.5)
        
        plt.figure(figsize=(12, 6))
        plt.plot(months, state_values, 'o-', linewidth=2, markersize=8)
        plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                  ['0-1k', '1k-2k', '2k-3k', '3k-4k', '4k-5k', '5k-6k', '6k-7k', '7k-8k', '8k-9k', '9k+'])
        plt.grid(True, alpha=0.3)
        plt.title(f'Most Likely Infection Level Path from {month_names[start_month-1]} to {month_names[input_month-1]}', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Infection Level (cases per million)', fontsize=12)
        
        # Highlight start and end months
        plt.plot(months[0], state_values[0], 'go', markersize=10, label='Start')
        plt.plot(months[-1], state_values[-1], 'ro', markersize=10, label='End')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/viterbi_path.png")
            print(f"Viterbi path visualization saved to {save_path}/viterbi_path.png")
        else:
            plt.show()
    
    def run_interactive_demo(self):
        """Run an interactive demo of the HMM functionality."""
        print("\n========== COVID-19 HMM INTERACTIVE DEMO ==========\n")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Filter level of infection for a current month")
            print("2. Predict next 3 months level of infection")
            print("3. Use Viterbi algorithm to explain infection levels")
            print("4. Visualize HMM components")
            print("5. Exit")
            
            choice = input("Enter your choice (1-5): ")
            
            if choice == '1':
                month = int(input("Enter month number (1-12): "))
                if 1 <= month <= 12:
                    filtered_probs = self.filter_current_month(month)
                    print(f"\nFiltered probabilities for month {month}:")
                    for i, state in enumerate(self.states):
                        print(f"{state}: {filtered_probs[i]:.4f}")
                    
                    # Find the most likely state
                    most_likely = np.argmax(filtered_probs)
                    print(f"\nMost likely infection level for month {month}: {self.states[most_likely]}")
                else:
                    print("Invalid month. Please enter a number between 1 and 12.")
            
            elif choice == '2':
                month = int(input("Enter current month number (1-12): "))
                if 1 <= month <= 12:
                    predictions = self.predict_next_months(month)
                    
                    print(f"\nPredictions starting from month {month}:")
                    for i, pred in enumerate(predictions):
                        next_month = (month + i) % 12 + 1
                        most_likely = np.argmax(pred)
                        print(f"Month {next_month}: Most likely level is {self.states[most_likely]} (probability: {pred[most_likely]:.4f})")
                else:
                    print("Invalid month. Please enter a number between 1 and 12.")
            
            elif choice == '3':
                start_month = int(input("Enter starting month (1-12): "))
                end_month = int(input("Enter ending month (1-12): "))
                
                if 1 <= start_month <= 12 and 1 <= end_month <= 12:
                    path = self.viterbi_algorithm(start_month, end_month)
                    
                    print(f"\nMost likely infection level path from month {start_month} to {end_month}:")
                    current_month = start_month
                    for i, state in enumerate(path):
                        print(f"Month {current_month}: {state}")
                        current_month = current_month % 12 + 1
                    
                    # Visualize the path
                    self.visualize_viterbi_path(start_month, end_month)
                else:
                    print("Invalid month. Please enter numbers between 1 and 12.")
            
            elif choice == '4':
                self.visualize_hmm()
            
            elif choice == '5':
                print("\nExiting demo. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")


# Main execution
if __name__ == "__main__":
    try:
        # Initialize the HMM with the COVID data file
        hmm = CovidHMM('covid_data.csv')
        
        # Generate visualizations
        output_dir = 'covid_hmm_output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        hmm.visualize_hmm(output_dir)
        
        # Example: Find the most likely path from January to December
        path = hmm.viterbi_algorithm(1, 12)
        print("\nMost likely infection level path from January to December:")
        for i, state in enumerate(path):
            print(f"Month {i+1}: {state}")
        
        # Example: Visualize Viterbi path
        hmm.visualize_viterbi_path(1, 12, output_dir)
        
        # Run the interactive demo
        hmm.run_interactive_demo()
        
    except Exception as e:
        print(f"An error occurred: {e}")