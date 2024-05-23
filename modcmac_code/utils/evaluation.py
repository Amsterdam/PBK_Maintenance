import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_scoring_table(file_path):
    """Reads the scoring table CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the scoring table.
    """
    return pd.read_csv(file_path)


def extract_component_states(df):
    """Extracts component states and action columns from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the scoring table.

    Returns:
        tuple: A tuple containing:
            - component_states (dict): Mapping of component numbers to their states.
            - action_columns (dict): Mapping of component numbers to their action columns.
    """
    component_states = {}
    action_columns = {}
    for col in df.columns:
        if col.startswith("state_comp_"):
            comp_num, state_num = col.split("_")[2], col.split("_")[4]
            if comp_num not in component_states:
                component_states[comp_num] = []
            component_states[comp_num].append((col, state_num))
        elif col.startswith("action_comp_"):
            comp_num = col.split("_")[2]
            action_columns[comp_num] = col
    return component_states, action_columns


def create_timestep_df(df, component_states, action_columns, action_map):
    """Creates a DataFrame for each timestep from the original DataFrame.

    Args:
        df (pd.DataFrame): Original DataFrame containing the scoring table.
        component_states (dict): Mapping of component numbers to their states.
        action_columns (dict): Mapping of component numbers to their action columns.
        action_map (dict): Mapping of action codes to action descriptions.

    Returns:
        pd.DataFrame: DataFrame containing states and actions for each timestep.
    """
    output_dfs = []
    for index, row in df.iterrows():
        timestep_states = {}
        for comp_num, states_list in component_states.items():
            comp_state = None
            comp_action = None
            for state_col, state_num in states_list:
                if row[state_col] == 1:
                    comp_state = state_num
                    comp_action = action_map[row[action_columns[comp_num]]]
                    break
            timestep_states[f"Component {comp_num} State"] = comp_state
            timestep_states[f"Component {comp_num} Action"] = comp_action
        output_dfs.append(pd.DataFrame(timestep_states, index=[0]))
    return pd.concat(output_dfs, ignore_index=True)


def count_failures_per_component(output_df):
    """Counts failures per component and logs timesteps of failures within the first 7 timesteps.

    Args:
        output_df (pd.DataFrame): DataFrame containing states and actions for each timestep.

    Returns:
        pd.DataFrame: DataFrame containing failure statistics for each component.
    """
    failures_per_component = {}
    for col in output_df.columns:
        if col.endswith("State"):
            comp_num = col.split(" ")[1]
            failure_count = output_df[col].eq('4').sum()
            avg_failures_per_timestep = failure_count / len(output_df)
            first_7_failures_count = output_df.iloc[:7][col].eq('4').sum()
            failures_per_component[comp_num] = {
                'Failure Count': failure_count,
                'Average Failures per Timestep': avg_failures_per_timestep,
                'Failures within First 7 Timesteps': first_7_failures_count
            }
    failures_df = pd.DataFrame.from_dict(failures_per_component, orient='index')
    failures_df['Component Number'] = failures_df.index
    failures_df = failures_df[
        ['Component Number', 'Failure Count', 'Average Failures per Timestep', 'Failures within First 7 Timesteps']]
    return failures_df


def calculate_cumulative_actions(output_df):
    """Calculates cumulative actions for each component-action pair.

    Args:
        output_df (pd.DataFrame): DataFrame containing states and actions for each timestep.

    Returns:
        pd.DataFrame: DataFrame containing cumulative actions for each component-action pair.
    """
    cumulative_actions = {}
    comp_action_counts = {(comp_num, action): 0 for comp_num in range(13) for action in ['Replace', 'Repair']}
    prev_counts = comp_action_counts.copy()
    for index, row in output_df.iloc[1:].iterrows():
        for comp_num in range(13):
            action = row[f'Component {comp_num} Action']
            if action in ['Repair', 'Replace']:
                comp_action_counts[(comp_num, action)] += 1
        count_changes = {k: v - prev_counts.get(k, 0) for k, v in comp_action_counts.items()}
        prev_counts = comp_action_counts.copy()
        cumulative_actions[index] = count_changes
    cumulative_actions_df = pd.DataFrame(cumulative_actions).fillna(0).astype(int)
    return cumulative_actions_df


def reformat_cumulative_actions(cumulative_actions_df):
    """Reformats the cumulative actions DataFrame.

    Args:
        cumulative_actions_df (pd.DataFrame): DataFrame containing cumulative actions for each component-action pair.

    Returns:
        pd.DataFrame: Reformatted DataFrame with specified column names.
    """
    final_dfs = []
    for (comp_num, action), counts in cumulative_actions_df.iterrows():
        data = {'Component Number': [comp_num], 'Action Type': [action]}
        data.update({f'Y{index}': [count] for index, count in counts.items()})
        df = pd.DataFrame(data)
        final_dfs.append(df)
    final_df = pd.concat(final_dfs, ignore_index=True).fillna(0)
    final_df = final_df.sort_values(by='Component Number')
    final_df['Y15'] = final_df.loc[:, 'Y11':'Y15'].sum(axis=1)
    final_df['Y20'] = final_df.loc[:, 'Y16':'Y20'].sum(axis=1)
    final_df['Y30'] = final_df.loc[:, 'Y21':'Y30'].sum(axis=1)
    final_df['Y40'] = final_df.loc[:, 'Y31':'Y40'].sum(axis=1)
    final_df['Y50'] = final_df.loc[:, 'Y41':'Y50'].sum(axis=1)
    final_df = final_df[
        ['Component Number', 'Action Type', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y15', 'Y20',
         'Y30', 'Y40', 'Y50']]
    return final_df


def plot_failure_counts(failures_df):
    """Plots a histogram of failure counts per component.

    Args:
        failures_df (pd.DataFrame): DataFrame containing failure statistics for each component.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(failures_df['Component Number'], failures_df['Failure Count'], color='skyblue')
    plt.xlabel('Component Number')
    plt.ylabel('Failure Count')
    plt.title('Failure Counts per Component')
    plt.xticks(failures_df['Component Number'])
    plt.gca().set_facecolor('white')
    plt.show()


def plot_avg_failures_per_timestep(failures_df):
    """Plots a histogram of average failures per timestep.

    Args:
        failures_df (pd.DataFrame): DataFrame containing failure statistics for each component.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(failures_df['Component Number'], failures_df['Average Failures per Timestep'], color='lightgreen')
    plt.xlabel('Component Number')
    plt.ylabel('Average Failures per Year')
    plt.title('Average Failures per Year')
    plt.xticks(failures_df['Component Number'])
    plt.gca().set_facecolor('white')
    plt.show()


def plot_total_failures_comparison(total_failures_7_years, total_avg_failures_modcmac):
    """Plots a comparison of total failures for Reactive and MO-DCMAC models over 7 years.

    Args:
        total_failures_7_years (int): Total failures over 7 years for the reactive model.
        total_avg_failures_modcmac (float): Average failures per year for the MO-DCMAC model.
    """
    plt.figure(figsize=(10, 5))
    bars = plt.bar(['Reactive Model', 'MO-DCMAC Model'], [total_failures_7_years, total_avg_failures_modcmac],
                   color=['skyblue', 'lightgreen'])
    plt.xlabel('Model')
    plt.ylabel('Total Failures over 7 years')
    plt.title('Comparison of Total Failures for Reactive and MO-DCMAC Models over 7 years')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, '%d' % int(height), ha='center', va='bottom')

    plt.show()


def plot_avg_failures_comparison(total_avg_failures_reactive, total_avg_failures_modcmac):
    """Plots a comparison of average failures per year for Reactive and MO-DCMAC models.

    Args:
        total_avg_failures_reactive (float): Average failures per year for the reactive model.
        total_avg_failures_modcmac (float): Average failures per year for the MO-DCMAC model.
    """
    plt.figure(figsize=(10, 5))
    bars = plt.bar(['Reactive Model', 'MO-DCMAC Model'], [total_avg_failures_reactive, total_avg_failures_modcmac],
                   color=['skyblue', 'lightgreen'])
    plt.xlabel('Model')
    plt.ylabel('Average Failures per Year')
    plt.title('Comparison of Average Failures per Year for Reactive and MO-DCMAC Models')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, '%.3f' % height, ha='center', va='bottom')

    plt.show()


def plot_failures_by_component_type(failures_df):
    """Plots total failures over 7 years for MO-DCMAC model by component type.

    Args:
        failures_df (pd.DataFrame): DataFrame containing failure statistics for each component.
    """
    component_types = {
        'Poles': range(9),
        'Kesps': range(9, 12),
        'Floor': [12]
    }

    failures_per_component_type = {component_type: 0 for component_type in component_types}

    for index, row in failures_df.iterrows():
        component_number = int(index)
        component_failures = row['Failure Count']
        for component_type, component_nums in component_types.items():
            if component_number in component_nums:
                failures_per_component_type[component_type] += component_failures
                break

    plt.figure(figsize=(10, 5))
    plt.bar(failures_per_component_type.keys(), failures_per_component_type.values(),
            color=['skyblue', 'lightgreen', 'salmon'])
    plt.xlabel('Component Type')
    plt.ylabel('Total Failures over 7 Years')
    plt.title('Total Failures over 7 Years for MO-DCMAC Model by Component Type')
    plt.show()


def plot_cumulative_actions(final_df):
    """Plots cumulative actions for each component-action pair.

    Args:
        final_df (pd.DataFrame): DataFrame containing cumulative actions for each component-action pair.
    """
    data = final_df.drop(columns=['Component Number', 'Action Type']).values
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    repair_data = final_df[final_df['Action Type'] == 'Repair'].drop(columns=['Component Number', 'Action Type']).values
    axs[0].pcolormesh(repair_data, cmap='RdYlGn_r', edgecolors='k', linewidth=2, vmin=repair_data.min(),
                      vmax=repair_data.max())

    for i in range(repair_data.shape[0]):
        for j in range(repair_data.shape[1]):
            if repair_data[i, j] != 0:
                axs[0].text(j + 0.5, i + 0.5, repair_data[i, j], color='black', ha='center', va='center')

    repair_row_totals = repair_data.sum(axis=1)
    for i, total in enumerate(repair_row_totals):
        axs[0].text(repair_data.shape[1] + 0.5, i + 0.5, total, color='black', ha='center', va='center')

    axs[0].set_title('Repair Action')
    axs[0].set_ylabel('Component Number')
    axs[0].set_yticks(np.arange(repair_data.shape[0]) + 0.5)
    axs[0].set_yticklabels(final_df[final_df['Action Type'] == 'Repair']['Component Number'], fontsize=8)
    axs[0].invert_yaxis()

    replace_data = final_df[final_df['Action Type'] == 'Replace'].drop(
        columns=['Component Number', 'Action Type']).values
    axs[1].pcolormesh(replace_data, cmap='RdYlGn_r', edgecolors='k', linewidth=2, vmin=replace_data.min(),
                      vmax=replace_data.max())

    for i in range(replace_data.shape[0]):
        for j in range(replace_data.shape[1]):
            if replace_data[i, j] != 0:
                axs[1].text(j + 0.5, i + 0.5, replace_data[i, j], color='black', ha='center', va='center')

    replace_row_totals = replace_data.sum(axis=1)
    for i, total in enumerate(replace_row_totals):
        axs[1].text(replace_data.shape[1] + 0.5, i + 0.5, total, color='black', ha='center', va='center')

    axs[1].set_title('Replace Action')
    axs[1].set_ylabel('Component Number')
    axs[1].set_yticks(np.arange(replace_data.shape[0]) + 0.5)
    axs[1].set_yticklabels(final_df[final_df['Action Type'] == 'Replace']['Component Number'], fontsize=8)
    axs[1].invert_yaxis()

    for ax in axs:
        ax.set_xticks(np.arange(len(final_df.columns[2:])) + 0.5)
        ax.set_xticklabels(final_df.columns[2:], rotation=45, ha='right')
        ax.set_xlabel('Timestep')

    plt.tight_layout()
    plt.savefig('component_actions.png', bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.show()

def main1():
    file_path = '../MODCMAC/results/020524/MO_DCMAC_scoring.csv'
    action_map = {0: 'Nothing', 1: 'Repair', 2: 'Replace'}

    df = read_scoring_table(file_path)
    component_states, action_columns = extract_component_states(df)
    output_df = create_timestep_df(df, component_states, action_columns, action_map)
    failures_df = count_failures_per_component(output_df)
    cumulative_actions_df = calculate_cumulative_actions(output_df)
    final_df = reformat_cumulative_actions(cumulative_actions_df)

    print(failures_df)
    print(final_df)

def main2():
    file_path = '../MODCMAC/results/020524/MO_DCMAC_scoring.csv'
    action_map = {0: 'Nothing', 1: 'Repair', 2: 'Replace'}

    df = read_scoring_table(file_path)
    component_states, action_columns = extract_component_states(df)
    output_df = create_timestep_df(df, component_states, action_columns, action_map)
    failures_df = count_failures_per_component(output_df)
    cumulative_actions_df = calculate_cumulative_actions(output_df)
    final_df = reformat_cumulative_actions(cumulative_actions_df)

    plot_failure_counts(failures_df)
    plot_avg_failures_per_timestep(failures_df)
    plot_total_failures_comparison(47, failures_df['Average Failures per Timestep'].mean())
    plot_avg_failures_comparison(47 / 7, failures_df['Average Failures per Timestep'].mean())
    plot_failures_by_component_type(failures_df)
    plot_cumulative_actions(final_df)
