import pandas as pd

# Defining the criteria and activities as provided
criteria = [
    "water quality",
    "noise",
    "odor",
    "air quality",
    "soil quality",
    "erosion",
    "sedimentation",
    "ecosystem",
    "endangered species",
    "fauna",
    "flora",
    "land use",
    "employment & lifestyle",
    "economic development",
    "community upgradation",
    "communication",
    "accident and health risk"
]

activities = [
    "Site preparation and clearing",
    "Earthworks (excavation, soil movement)",
    "Foundation works (concrete pouring, foundation stabilization)",
    "Structural construction (tank, building, pipeline installation)",
    "Road and access route construction",
    "Drainage and wastewater treatment infrastructure installation",
    "Electrical and mechanical systems installation",
    "Temporary facilities (e.g., construction offices, material storage)"
]

# Example values assigned to each activity for each criterion, based on hypothetical impact assessment
# These values are illustrative and may not reflect actual impact assessments
# For an actual Leopold matrix, these should be determined by an environmental expert



# Using the provided criteria and activities, we will assign hypothetical impact values to a larger number of items.
# The format is degree/significance, following the provided rules.
# If an activity does not impact a criterion, we will indicate this with "없다!"

# Initialize the matrix with blank values
matrix = {activity: {criterion: '' for criterion in criteria} for activity in activities}

# Hypothetical example of assigning values
for activity in activities:
    for criterion in criteria:
        if activity == "Site preparation and clearing":
            # Just as an example, assuming site preparation has a widespread negative impact on most criteria
            matrix[activity][criterion] = "-3 / 4" if criterion in ["soil quality", "erosion", "land use"] else "없다!"
        elif activity == "Earthworks (excavation, soil movement)":
            matrix[activity][criterion] = "-2 / 3" if criterion in ["water quality", "sedimentation"] else "없다!"
        elif activity == "Foundation works (concrete pouring, foundation stabilization)":
            matrix[activity][criterion] = "-1 / 2" if criterion in ["noise", "air quality"] else "없다!"
        elif activity == "Structural construction (tank, building, pipeline installation)":
            matrix[activity][criterion] = "-2 / 3" if criterion in ["fauna", "flora"] else "없다!"
        elif activity == "Road and access route construction":
            matrix[activity][criterion] = "-1 / 2" if criterion in ["noise", "traffic"] else "없다!"
        elif activity == "Drainage and wastewater treatment infrastructure installation":
            matrix[activity][criterion] = "2 / 3" if criterion in ["water quality", "ecosystem"] else "없다!"
        elif activity == "Electrical and mechanical systems installation":
            matrix[activity][criterion] = "-1 / 1" if criterion == "noise" else "없다!"
        elif activity == "Temporary facilities (e.g., construction offices, material storage)":
            matrix[activity][criterion] = "1 / 2" if criterion in ["employment & lifestyle", "economic development"] else "없다!"
        else:
            # Assuming the rest of the activities have no impact on the rest of the criteria
            matrix[activity][criterion] = "없다!"

# Convert the matrix into a pandas DataFrame for display
leopold_matrix_df = pd.DataFrame(matrix)

print(matrix)
print(leopold_matrix_df)

leopold_matrix_df.to_excel('output.xlsx', index=False)