from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as RT
import pandas as pd
import numpy as np
import time
from multiprocessing import Manager, Process

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


df = pd.read_csv("all_busloads.csv", index_col=None, header=None)
data = df.values
reshaped_data = data.ravel(order='F')
reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((366, 99, 288)).transpose((0, 2, 1))
expanded_x = newdata

newy = pd.read_csv("y1.csv", index_col=None, header=None)
newy = newy.values
newy = newy.ravel(order='F')
newy = np.array(newy)
newy = newy.reshape((366, 288, 54))

expanded_x = expanded_x.reshape(366 * 288, 99)
newy = newy.reshape(366 * 288, 54)

# Delineate training and test sets with a ratio of 8:2
X_train, X_test, y_train, y_test = train_test_split(expanded_x, newy, test_size=0.2, random_state=42, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


input_dim = X_train.shape[1]
hidden_dim = 128
code_dim = 54
autoencoder = Autoencoder(input_dim, hidden_dim, code_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)


epochs = 200
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

for epoch in range(epochs):
    autoencoder.train()
    optimizer.zero_grad()
    encoded, decoded = autoencoder(X_train_tensor)
    loss = criterion(decoded, X_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Extract the coding features of the self-encoder
autoencoder.eval()
with torch.no_grad():
    encoded_train_features = autoencoder.encoder(X_train_tensor).cpu().numpy()
    encoded_test_features = autoencoder.encoder(X_test_tensor).cpu().numpy()

# Decision tree generates leaf node features
tree = DecisionTreeRegressor(max_depth=8)
tree.fit(X_train, y_train)
leaf_node_features_train = tree.apply(X_train).reshape(-1, 1)
leaf_node_features_test = tree.apply(X_test).reshape(-1, 1)
# Merge self-encoder features and decision tree features
combined_train_features = np.concatenate((encoded_train_features, leaf_node_features_train), axis=1)
combined_test_features = np.concatenate((encoded_test_features, leaf_node_features_test), axis=1)


decoder_weights = autoencoder.decoder[0].weight.data.abs().mean(dim=0).cpu().numpy()

# Normalised weights
# feature_importance_autoencoder = decoder_weights / decoder_weights.sum()
feature_importance_autoencoder =decoder_weights

# tree_weights = 2.0
feature_importance_tree = np.array([0.08])
feature_importance = np.concatenate([feature_importance_autoencoder, feature_importance_tree])
assert len(feature_importance) == combined_train_features.shape[1], "False"


weighted_train_features = combined_train_features * feature_importance[np.newaxis, :]
weighted_test_features = combined_test_features * feature_importance[np.newaxis, :]

# Create new DMatrix object with feature weights
dtrain_weighted = RT.DMatrix(weighted_train_features, label=y_train, feature_weights=feature_importance)
dtest_weighted = RT.DMatrix(weighted_test_features, feature_weights=feature_importance)


params = {
    'objective': 'reg:squarederror',
    'max_depth': 50,
    'eta': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse'
}

# Train RT with weighted features and new parameters
bst = RT.train(params, dtrain_weighted, num_boost_round=150)

# Prediction using weighted features
start_time = time.time()
predictions = bst.predict(dtest_weighted)
end_time = time.time()

predictions[np.abs(predictions) < 1] = 0


mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = sqrt(mse)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Time taken: {(end_time - start_time)/73.2:.4f} s/day')

# Create DataFrame of predicted and actual values
predictions_df = pd.DataFrame(predictions, columns=[f'Pred_{i}' for i in range(predictions.shape[1])])
actual_values_df = pd.DataFrame(y_test, columns=[f'Actual_{i}' for i in range(y_test.shape[1])])


predictions_df['Actual_Sum'] = actual_values_df.sum(axis=1)


results = pd.concat([predictions_df, actual_values_df], axis=1)


results.to_csv('./ST.csv', index=False)
print('Predictions and actual values have been saved to Proposed model predictions.csv')

# Load physical constraints
constraints_df = pd.read_csv('tunit.csv')
min_power = constraints_df['Pmin'].values
max_power = constraints_df['Pmax'].values
ramp_up = constraints_df['RampUp'].values
ramp_down = constraints_df['RampDown'].values
# min_up_time = constraints_df['MinUpTime'].values  #
# min_down_time = constraints_df['MinDownTime'].values  #
min_up_time = np.full(54, 3)
min_down_time = np.full(54, 3)
# of loaded cost parameters
cost_params_df = pd.read_csv('tunit_price.csv')
cost_params = cost_params_df[['NETCOST', 'INCCOST', 'INCCOSTEND']].values
factor=0
# Reading forecast and actual data
df = pd.read_csv('./ST.csv')
# Global transmission limit constants
GLOBAL_TRANSMISSION_LIMIT = 100000
# Read connectivity from line.xlsx file
line_df = pd.read_excel('line.xlsx', header=None)
line_df.columns = ['From', 'To']
edge_list = list(zip(line_df['From'], line_df['To']))


full_neighbors = {}
for node1, node2 in edge_list:
    full_neighbors.setdefault(node1, set()).add(node2)
    full_neighbors.setdefault(node2, set()).add(node1)

generator_units = set([1, 4, 6, 8, 10, 11, 12, 15, 18, 19,
                       24, 25, 26, 27, 31, 32, 34, 36, 40, 42,
                       46, 49, 54, 55, 56, 62, 63, 64, 65, 66,
                       68, 70, 72, 73, 74, 76, 77, 81, 85, 87,
                       89, 90, 91, 92, 99, 100, 103, 104, 105, 107,
                       110, 112, 113, 116])

generator_neighbors = {}
for unit in generator_units:
    neighbors_set = full_neighbors.get(unit, set())
    filtered_neighbors = neighbors_set & generator_units
    generator_neighbors[unit] = list(filtered_neighbors)

unit_to_index = {unit: idx for idx, unit in enumerate(sorted(generator_units))}
index_to_unit = {idx: unit for unit, idx in unit_to_index.items()}

neighbors = {}
for unit, neighbor_units in generator_neighbors.items():
    unit_idx = unit_to_index[unit]
    neighbor_indices = [unit_to_index[neighbor_unit] for neighbor_unit in neighbor_units]
    neighbors[unit_idx] = neighbor_indices



class UnitStatus:
    def __init__(self, num_units):
        self.up_time = np.zeros(num_units)
        self.down_time = np.zeros(num_units)
        self.status = np.zeros(num_units)

    def update(self, power_output):
        new_status = (power_output >= min_power).astype(int)

        for i in range(len(power_output)):
            if new_status[i] == 1:
                if self.status[i] == 1:
                    self.up_time[i] += 1
                    self.down_time[i] = 0
                else:
                    self.up_time[i] = 1
                    self.down_time[i] = 0
            else:
                if self.status[i] == 0:
                    self.down_time[i] += 1
                    self.up_time[i] = 0
                else:
                    self.down_time[i] = 1
                    self.up_time[i] = 0

        self.status = new_status

    def check_constraints(self, proposed_power, min_up_time, min_down_time):
        proposed_status = (proposed_power >= min_power).astype(int)
        valid = np.ones(len(proposed_power), dtype=bool)

        for i in range(len(proposed_power)):
            if self.status[i] != proposed_status[i]:
                if self.status[i] == 1:
                    if self.up_time[i] < min_up_time[i]:
                        valid[i] = False
                else:
                    if self.down_time[i] < min_down_time[i]:
                        valid[i] = False

        return valid


# Cost function
def calculate_cost(p, a, b, c):
    return 0.5 * a * (p ** 2) + b * p + c


# Power update function
def update_power_with_neighbors(current_power, min_power, max_power, ramp_up, ramp_down,
                              demand, cost_params, previous_power, neighbors, unit_status):
    adjusted_power = current_power.copy()
    """
        Update generator outputs considering neighborhood relationships and various constraints

        Note:
        1. Regarding transmission constraints: While PTDF matrices are commonly used in theoretical 
           research for accurate power flow calculation and constraint handling, in China's power 
           system operation, system security is typically ensured by increasing transmission line 
           capacity rather than relying on generation-side adjustment.
        2. This code adopts a simplified power difference limit between adjacent units to simulate 
           transmission constraints. Although this approach is less precise than using PTDF matrices, 
           it aligns with the actual operational characteristics of the system.
        """
    # Special handling of the first moment
    if previous_power is None:
        # For the first moment, the original power is maintained and only obvious violations of the constraint are handled
        for i in range(len(adjusted_power)):
            if adjusted_power[i] > 0:
                # If power is close to minimum power, adjust to minimum power
                if adjusted_power[i] < min_power[i]:
                    if adjusted_power[i] >= min_power[i] * 0.8:  # 80%的容差
                        adjusted_power[i] = min_power[i]
                    else:
                        adjusted_power[i] = 0
                # If maximum power is exceeded, adjust to maximum power
                if adjusted_power[i] > max_power[i]:
                    adjusted_power[i] = max_power[i]
    else:

        valid_changes = unit_status.check_constraints(adjusted_power, min_up_time, min_down_time)
        for i in range(len(adjusted_power)):
            if not valid_changes[i]:
                adjusted_power[i] = previous_power[i]


        zero_indices = np.where(adjusted_power < min_power)[0]
        adjusted_power[zero_indices] = 0


    adjusted_power = np.clip(adjusted_power, 0, max_power)


    if previous_power is not None:
        for i in range(len(adjusted_power)):
            if adjusted_power[i] == 0:
                continue
            adjusted_power[i] = np.clip(
                adjusted_power[i],
                max(previous_power[i] - ramp_down[i], 0),
                min(previous_power[i] + ramp_up[i], max_power[i])
            )


    for i in range(len(adjusted_power)):
        if adjusted_power[i] == 0:
            continue
        neighbor_indices = neighbors.get(i, [])
        if neighbor_indices:
            active_neighbors = [n for n in neighbor_indices if adjusted_power[n] > 0]
            if active_neighbors:
                neighbor_powers = adjusted_power[active_neighbors]
                neighbor_avg_power = np.mean(neighbor_powers)
                adjustment_factor=factor
                adjusted_power[i] += adjustment_factor * (neighbor_avg_power - adjusted_power[i])
                adjusted_power[i] = np.clip(adjusted_power[i], min_power[i], max_power[i])
                if previous_power is not None:
                    adjusted_power[i] = np.clip(
                        adjusted_power[i],
                        max(previous_power[i] - ramp_down[i], min_power[i]),
                        min(previous_power[i] + ramp_up[i], max_power[i])
                    )

    for i in range(len(adjusted_power)):
        if adjusted_power[i] > 0:
            neighbor_indices = neighbors.get(i, [])
            for neighbor in neighbor_indices:
                if adjusted_power[neighbor] > 0:
                    power_difference = abs(adjusted_power[i] - adjusted_power[neighbor])
                    if power_difference > GLOBAL_TRANSMISSION_LIMIT:
                        if adjusted_power[i] > adjusted_power[neighbor]:
                            adjusted_power[i] = adjusted_power[neighbor] + GLOBAL_TRANSMISSION_LIMIT
                        else:
                            adjusted_power[i] = adjusted_power[neighbor] - GLOBAL_TRANSMISSION_LIMIT

                        adjusted_power[i] = np.clip(adjusted_power[i], min_power[i], max_power[i])
                        if previous_power is not None:
                            adjusted_power[i] = np.clip(
                                adjusted_power[i],
                                max(previous_power[i] - ramp_down[i], min_power[i]),
                                min(previous_power[i] + ramp_up[i], max_power[i])
                            )

    total_power = np.sum(adjusted_power)
    power_difference = demand - total_power


    if power_difference > 0:
        deficit = power_difference

        cost_efficiency = [cost_params[i][0] * adjusted_power[i] + cost_params[i][1]
                           if adjusted_power[i] > 0 else float('inf')
                           for i in range(len(adjusted_power))]
        for i in np.argsort(cost_efficiency):
            if deficit <= 0:
                break
            if adjusted_power[i] == 0:
                continue
            possible_increase = min(max_power[i] - adjusted_power[i],
                                    ramp_up[i] if previous_power is not None else max_power[i])
            increase = min(deficit, possible_increase)
            adjusted_power[i] += increase
            deficit -= increase


    elif power_difference < 0:
        surplus = -power_difference

        cost_efficiency = [cost_params[i][0] * adjusted_power[i] + cost_params[i][1]
                           if adjusted_power[i] > 0 else -float('inf')
                           for i in range(len(adjusted_power))]
        for i in np.argsort(cost_efficiency)[::-1]:
            if surplus <= 0:
                break
            if adjusted_power[i] == 0:
                continue
            possible_decrease = min(adjusted_power[i] - min_power[i],
                                    ramp_down[i] if previous_power is not None else adjusted_power[i])
            decrease = min(surplus, possible_decrease)
            adjusted_power[i] -= decrease
            surplus -= decrease


    adjusted_power = np.clip(adjusted_power, 0, max_power)


    total_cost = sum(calculate_cost(adjusted_power[i], *cost_params[i])
                     for i in range(len(adjusted_power)))

    return adjusted_power, total_cost


#
def compute_adjusted_power_multi(start_idx, end_idx, df, previous_power, return_dict, idx, neighbors):
    unit_status = UnitStatus(54)
    adjusted_values = []

    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        initial_power = row[[f'Pred_{i}' for i in range(54)]].values
        demand_power = row['Actual_Sum']
        current_power = initial_power.copy()

        iterations =5
        for _ in range(iterations):
            current_power, total_cost = update_power_with_neighbors(
                current_power, min_power, max_power, ramp_up, ramp_down,
                demand_power, cost_params, previous_power, neighbors, unit_status
            )

        unit_status.update(current_power)
        adjusted_values.append(current_power)
        previous_power = current_power

    return_dict[idx] = adjusted_values


if __name__ == '__main__':
    manager = Manager()
    return_dict = manager.dict()

    num_processes = 104
    processes = []
    chunk_size = len(df) // num_processes
    previous_power = None
    start_time2 = time.time()

    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_processes - 1 else len(df)
        p = Process(target=compute_adjusted_power_multi, args=(
            start_idx, end_idx, df, previous_power, return_dict, i, neighbors))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time2 = time.time()


    adjusted_values = []
    for i in range(num_processes):
        adjusted_values.extend(return_dict[i])

    adjusted_values = np.array(adjusted_values)
    adjusted_df = pd.DataFrame(adjusted_values, columns=[f'Adjust_{i}' for i in range(54)])

    combined_df = pd.concat([
        df[[f'Pred_{i}' for i in range(54)]].reset_index(drop=True),
        df[[f'Actual_{i}' for i in range(54)]].reset_index(drop=True),
        adjusted_df.reset_index(drop=True)
    ], axis=1)

    combined_df['Pred_Sum'] = combined_df[[f'Pred_{i}' for i in range(54)]].sum(axis=1)
    combined_df['Actual_Sum'] = combined_df[[f'Actual_{i}' for i in range(54)]].sum(axis=1)
    combined_df['Adjust_Sum'] = adjusted_df.sum(axis=1)

    combined_df['Pred_Cost_Sum'] = combined_df[[f'Pred_{i}' for i in range(54)]].apply(
        lambda x: sum(calculate_cost(x[i], *cost_params[i]) for i in range(len(x))), axis=1)
    combined_df['Actual_Cost_Sum'] = combined_df[[f'Actual_{i}' for i in range(54)]].apply(
        lambda x: sum(calculate_cost(x[i], *cost_params[i]) for i in range(len(x))), axis=1)
    combined_df['Adjust_Cost_Sum'] = adjusted_df.apply(
        lambda x: sum(calculate_cost(x[i], *cost_params[i]) for i in range(len(x))), axis=1)

    combined_df.to_csv('ST-GCCA.csv', index=False)

    print(f"Processing time: {(end_time2 - start_time2)/73.2} s/day")
    t1=(end_time-start_time)/73.2
    t2=(end_time2 - start_time2)/73.2
    print(f"all time:{t1+t2}s/day")
    print("Power adjustments and cost calculations successfully completed and saved to file.")
