from ortools.sat.python import cp_model
import pandas as pd
import colorcet as cc
import seaborn as sns
from matplotlib import pyplot as plt


def plot_day_schedule(schedule):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(w=2 * 9.5, h=2 * 5)
    fig.tight_layout(pad=1.7)

    resources = set(schedule['anesthetist_id'])
    resources = sorted(resources, key=lambda x: (len(x), x), reverse=True)
    resource_mapping = {resource: i for i, resource in enumerate(resources)}

    intervals_start = (schedule.start_time - schedule.start_time.dt.floor('d')).dt.total_seconds().div(3600)
    intervals_end = (schedule.end_time - schedule.start_time.dt.floor('d')).dt.total_seconds().div(3600)

    intervals = list(zip(intervals_start, intervals_end))

    palette = sns.color_palette(cc.glasbey_dark, n_colors=len(schedule))
    palette = [(color[0] * 0.9, color[1] * 0.9, color[2] * 0.9) for color in palette]
    cases_colors = {case_id: palette[i] for i, case_id in enumerate(set(schedule['room_id']))}

    for i, (resource_on_block_id, resource, evt) in enumerate(
            zip(schedule['room_id'], schedule['anesthetist_id'], intervals)):
        txt_to_print = str(i)
        ax.barh(resource_mapping[resource], width=evt[1] - evt[0], left=evt[0], linewidth=1, edgecolor='black',
                color=cases_colors[resource_on_block_id])
        ax.text((evt[0] + evt[1] - 0.07 * len(str(txt_to_print))) / 2, resource_mapping[resource], txt_to_print,
                name='Arial', color='white', va='center')

    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels([f'{resource}' for resource in resources])

    ax.set_ylabel('anesthetist_id'.replace('_', ' '))

    ax.title.set_text(f'Total {len(set(schedule["anesthetist_id"]))} anesthetists')


# Load and preprocess data
df = pd.read_csv('C:\\Users\\coral\\Downloads\\surgeries.csv', index_col=0)
df['start'] = pd.to_datetime(df['start'], dayfirst=True)
df['end'] = pd.to_datetime(df['end'], dayfirst=True)
df = df.dropna(subset=['start', 'end'])
# Convert start and end times to a numerical format (e.g., minutes since midnight)
df['start_minutes'] = (df['start'].dt.hour * 60 + df['start'].dt.minute).astype(int)
df['end_minutes'] = (df['end'].dt.hour * 60 + df['end'].dt.minute).astype(int)
df['Duration'] = (df['end'] - df['start']).dt.total_seconds() / 3600  # Convert duration to hours

all_surgeries = range(len(df))
# Assuming the worst-case scenario where each surgery could potentially require a different anesthesiologist
num_anesthesiologists = len(df)
all_anesthesiologists = range(num_anesthesiologists)
num_rooms = 20  # 20 operating rooms available
all_rooms = range(num_rooms)

model = cp_model.CpModel()

# assigned[(s, r)] is True if surgery s is assigned to room r
assigned_room = {}
for s in all_surgeries:
    for r in all_rooms:
        assigned_room[(s, r)] = model.NewBoolVar(f"assigned_s{s}_r{r}")

# Ensure each surgery is assigned to exactly one operating room
for s in all_surgeries:
    model.Add(sum(assigned_room[(s, r)] for r in all_rooms) == 1)

#  ensure that no two surgeries assigned to the same room overlap in time
for r in all_rooms:
    for i in all_surgeries:
        for j in range(i + 1, len(df)):
            # Check if surgeries i and j overlap
            if not (df.iloc[i]['end'] <= df.iloc[j]['start'] or df.iloc[j]['end'] <= df.iloc[i]['start']):
                # Ensure surgeries i and j are not assigned to the same room if they overlap
                model.Add(assigned_room[(i, r)] + assigned_room[(j, r)] <= 1)

# [(a, s)] is True if anesthesiologist a is assigned to surgery s
assigned = {}
for a in all_anesthesiologists:
    for s in all_surgeries:
        assigned[(a, s)] = model.NewBoolVar(f"assigned_a{a}_s{s}")

# Ensure each surgery is assigned to at one anesthesiologist
for s in all_surgeries:
    model.Add(sum(assigned[(a, s)] for a in all_anesthesiologists) == 1)

# preventing an anesthesiologist from being assigned to overlapping surgeries
for a in all_anesthesiologists:
    for i in all_surgeries:
        for j in range(i + 1, len(df)):
            # Check if surgeries i and j overlap
            if not (df.iloc[i]['end'] <= df.iloc[j]['start'] or df.iloc[j]['end'] <= df.iloc[i]['start']):
                # Ensure anesthesiologist a is not assigned to both overlapping surgeries
                model.Add(assigned[(a, i)] + assigned[(a, j)] <= 1)

# Initialize variables for the earliest start and latest end of surgeries for each anesthesiologist
earliest_start_time = {a: model.NewIntVar(0, int(max(df['end_minutes'])), f'earliest_start_a{a}') for a in all_anesthesiologists}
latest_end_time = {a: model.NewIntVar(0, int(max(df['end_minutes'])), f'latest_end_a{a}') for a in all_anesthesiologists}

# For each anesthesiologist, find the earliest start time and latest end time among assigned surgeries
for a in all_anesthesiologists:
    for s in all_surgeries:
        is_assigned = assigned[(a, s)]
        start_time = df.iloc[s]['start_minutes']
        end_time = df.iloc[s]['end_minutes']
        # Convert pandas series values to integer (if they are not already)
        start_time = int(start_time) if not isinstance(start_time, int) else start_time
        end_time = int(end_time) if not isinstance(end_time, int) else end_time
        # Create the constraints
        model.Add(earliest_start_time[a] <= start_time).OnlyEnforceIf(is_assigned)
        model.Add(latest_end_time[a] >= end_time).OnlyEnforceIf(is_assigned)

shift_duration = {}
shift_cost = {}

# For each anesthesiologist, calculate the shift cost
for a in all_anesthesiologists:
    # Variables to represent if an anesthesiologist's shift duration is above 5 hours (minimum) and 9 hours (overtime)
    is_shift_above_min = model.NewBoolVar(f'is_shift_above_min_{a}')
    is_shift_above_overtime = model.NewBoolVar(f'is_shift_above_overtime_{a}')

    # The duration of the shift in hours
    shift_hours = model.NewIntVar(0, 24, f'shift_hours_{a}')
    shift_duration[a] = model.NewIntVar(0, max(df['end_minutes']) - min(df['start_minutes']), f'shift_duration_a{a}')
    model.Add(shift_hours == shift_duration[a])

    # If the shift duration is less than or equal to 5 hours, the cost is 5
    # If the shift duration is more than 5 hours, the cost is the shift duration
    # This is handled by setting is_shift_above_min to True when shift_duration[a] > 5 * 60 (5 hours)
    model.Add(shift_duration[a] > 5 * 60).OnlyEnforceIf(is_shift_above_min)
    model.Add(shift_duration[a] <= 5 * 60).OnlyEnforceIf(is_shift_above_min.Not())

    # The base cost is either 5 (minimum cost) or the actual shift duration in hours (if above 5 hours)
    base_cost = model.NewIntVar(0, 24, f'base_cost_{a}')
    model.Add(base_cost == 5).OnlyEnforceIf(is_shift_above_min.Not())
    model.Add(base_cost == shift_hours).OnlyEnforceIf(is_shift_above_min)

    # If the shift duration is more than 9 hours, there's an additional overtime cost
    # Multiply the duration and 9 hours by 2 (to avoid fractions), and then the result of subtraction by 0.5 will be integer division
    scaled_duration = model.NewIntVar(0, 2 * max(df['end_minutes']), f'scaled_duration_a{a}')
    model.Add(scaled_duration == 2 * shift_duration[a])

    additional_hours_scaled = model.NewIntVar(0, 2 * (24 - 9), f'additional_hours_scaled_{a}')
    model.Add(additional_hours_scaled == scaled_duration - (9 * 2 * 60)).OnlyEnforceIf(is_shift_above_overtime)
    model.Add(additional_hours_scaled == 0).OnlyEnforceIf(is_shift_above_overtime.Not())

    # Since we scaled by 2, dividing by 2 now is the same as multiplying by 0.5 in the original formula
    overtime_cost = model.NewIntVar(0, (24 - 9), f'overtime_cost_{a}')
    model.AddDivisionEquality(overtime_cost, additional_hours_scaled, 2)

    # The total cost for an anesthesiologist's shift is the sum of the base cost and the overtime cost
    total_cost_a = model.NewIntVar(0, 5 + (24 - 9),
                                   f'total_cost_{a}')  # The total cost has a maximum based on the longest possible shift
    model.Add(total_cost_a == base_cost + overtime_cost)


# Minimize the total cost across all anesthesiologists
model.Minimize(sum(total_cost_a for a in all_anesthesiologists))

# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Optimal or feasible solution found. Preparing the schedule for plotting...")
    schedule_data = {
        'surgery_id': [],
        'anesthetist_id': [],
        'room_id': [],
        'start_time': [],
        'end_time': []
    }

    for s in all_surgeries:
        for r in all_rooms:
            if solver.Value(assigned_room[(s, r)]):
                # Assuming a surgery can only be assigned to one anesthetist, find which one
                for a in all_anesthesiologists:
                    if solver.Value(assigned[(a, s)]):
                        anesthetist_id = str(a)
                        break

                schedule_data['surgery_id'].append(str(s))
                schedule_data['anesthetist_id'].append(str(anesthetist_id))
                schedule_data['room_id'].append(str(r))
                start_time = df.iloc[s]['start']
                end_time = df.iloc[s]['end']
                schedule_data['start_time'].append(start_time)
                schedule_data['end_time'].append(end_time)


    schedule_df = pd.DataFrame(schedule_data)
    schedule_df['anesthetist_id'] = 'anesthetist - ' + schedule_df['anesthetist_id'].astype(str)
    schedule_df['room_id'] = 'room - ' + schedule_df['room_id'].astype(str)
    plot_day_schedule(schedule_df)
    # Save to CSV
    output_file_path = 'C:\\Users\\coral\\Downloads\\sol.csv'  # Adjust the file path as needed
    print(schedule_df.head())
    schedule_df.to_csv(output_file_path, index=False)
    plt.show()
else:
    print("No solution found.")


