import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_name = 'Sample_Data_for_Activity.csv'
data = pd.read_csv(file_name)

# Display the data for reference
print("Sample Data:")
print(data.head())

# Data Description
print("\nData Summary:")
print(data.describe())

# Visualization: Create a combined plot for all data types
plt.figure(figsize=(9, 6))

# Normal Distribution: Line Plot
plt.subplot(2, 2, 1)
plt.plot(data['Normal_Distribution'], marker='o', color='blue')
plt.title('Normal Distribution')
plt.xlabel('Index')
plt.ylabel('Normal_Distribution')

# Uniform Distribution: Line Plot
plt.subplot(2, 2, 2)
plt.plot(data['Uniform_Distribution'], marker='s', color='orange')
plt.title('Uniform Distribution')
plt.xlabel('Index')
plt.ylabel('Uniform_Distribution')

# Exponential Distribution: Line Plot
plt.subplot(2, 2, 3)
plt.plot(data['Exponential_Distribution'], marker='^', color='green')
plt.title('Exponential Distribution')
plt.xlabel('Index')
plt.ylabel('Exponential_Distribution')

# Poisson Distribution: Bar Plot
plt.subplot(2, 2, 4)
plt.bar(data.index, data['Poisson_Distribution'], color='red')
plt.title('Poisson Distribution')
plt.xlabel('Index')
plt.ylabel('Poisson_Distribution')

# Adjust layout and save
plt.tight_layout()
plt.savefig('combined_distributions_visualization.png', dpi=300)
plt.show()

# Story for the Data:
"""
The dataset represents a study of patterns observed in different scenarios of a modern urban ecosystem, each reflected through statistical distributions:

1. **Normal Distribution**:
   This data captures the natural rhythm of daily city activities, such as average commute times. The values cluster around a central mean, with minor deviations caused by fluctuations like traffic or weather conditions. This showcases the predictability of regular urban life.

2. **Uniform Distribution**:
   Representing resource allocation, such as the equal distribution of parking spaces across districts, this data ensures fairness. The uniform spread highlights balanced accessibility in city planning and infrastructure development.

3. **Exponential Distribution**:
   This reflects time intervals between successive events, such as the arrival of buses at a station. The steep decay indicates efficient scheduling, with shorter waiting times dominating the timeline, ensuring smooth public transport operations.

4. **Poisson Distribution**:
   Modeling discrete events like emergency service calls within an hour, this dataset highlights the frequency of rare but critical occurrences. Understanding these patterns aids in resource preparedness and dynamic response strategies.

Together, these visualizations narrate a cohesive story of a city’s heartbeat—a blend of predictable routines, equitable systems, efficient processes, and rare events. By analyzing these distributions, urban planners and decision-makers can fine-tune systems to ensure a harmonious and well-functioning urban environment.
"""